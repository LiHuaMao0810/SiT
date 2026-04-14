import argparse
import os
from time import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm
from torchvision.utils import save_image

from download import find_model
from grpo_utils import compute_elbo_ratio, compute_grpo_advantage, ppo_clip_loss
from models import SiT_models
from reward_utils import ClassifierReward
from sit_sampler import student_ode_sample, teacher_ode_sample
from train_utils import parse_transport_args
from transport import create_transport


def build_model(model_name, latent_size, num_classes, ckpt_path, device):
    model = SiT_models[model_name](input_size=latent_size, num_classes=num_classes).to(device)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    return model


def teacher_sample_chunked(teacher, transport, vae, y, num_steps, cfg_scale, latent_size, device, chunk_size, return_pixel):
    if chunk_size <= 0 or y.shape[0] <= chunk_size:
        return teacher_ode_sample(
            teacher,
            transport,
            vae,
            y,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            latent_size=latent_size,
            device=device,
            return_pixel=return_pixel,
        )

    latent_chunks = []
    pixel_chunks = []
    for start in range(0, y.shape[0], chunk_size):
        end = min(start + chunk_size, y.shape[0])
        out = teacher_ode_sample(
            teacher,
            transport,
            vae,
            y[start:end],
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            latent_size=latent_size,
            device=device,
            return_pixel=return_pixel,
        )
        if return_pixel:
            lat, pix = out
            latent_chunks.append(lat)
            pixel_chunks.append(pix)
        else:
            latent_chunks.append(out)

    latents = torch.cat(latent_chunks, dim=0)
    if not return_pixel:
        return latents
    pixels = torch.cat(pixel_chunks, dim=0)
    return latents, pixels


def per_sample_loss_chunked(model, x1, model_kwargs, transport, t, noise, chunk_size):
    if chunk_size <= 0 or x1.shape[0] <= chunk_size:
        return transport.per_sample_loss(model, x1, model_kwargs, t=t, noise=noise)
    loss_chunks = []
    for start in range(0, x1.shape[0], chunk_size):
        end = min(start + chunk_size, x1.shape[0])
        loss_chunks.append(
            transport.per_sample_loss(
                model,
                x1[start:end],
                {"y": model_kwargs["y"][start:end]},
                t=t[start:end],
                noise=noise[start:end],
            )
        )
    return torch.cat(loss_chunks, dim=0)


def main(args):
    assert torch.cuda.is_available(), "This script requires CUDA."
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    sample_dir = os.path.join(args.results_dir, "samples")
    if args.sample_every > 0:
        os.makedirs(sample_dir, exist_ok=True)

    latent_size = args.image_size // 8
    teacher = build_model(args.model, latent_size, args.num_classes, args.ckpt, device)
    student = build_model(args.model, latent_size, args.num_classes, args.ckpt, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    reward_fn = ClassifierReward(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0, foreach=False)
    scaler = GradScaler("cuda", enabled=args.amp)
    student.train()

    running = {
        "reward_mean": 0.0,
        "reward_std": 0.0,
        "ratio": 0.0,
        "clip_fraction": 0.0,
        "l_grpo": 0.0,
        "l_mse": 0.0,
        "k_actual": 0.0,
        "count": 0,
    }
    tic = time()

    pbar = tqdm(range(1, args.total_iters + 1), disable=args.no_tqdm, dynamic_ncols=True, desc="elbo_grpo")
    for step in pbar:
        b = args.batch_size
        g = args.G
        bg = b * g

        base_y = torch.randint(args.num_classes, size=(b,), device=device)
        cond_g = base_y.repeat_interleave(g)
        model_kwargs_g = dict(y=cond_g)

        # Phase 1: collect frozen data
        with torch.no_grad():
            x1_exp, x1_exp_pixel = teacher_sample_chunked(
                teacher,
                transport,
                vae,
                cond_g,
                num_steps=args.T_exp,
                cfg_scale=args.cfg_scale,
                latent_size=latent_size,
                device=device,
                chunk_size=args.bg_chunk_size,
                return_pixel=True,
            )
            rewards = reward_fn(x1_exp_pixel, cond_g).view(b, g)
            advantage = compute_grpo_advantage(rewards, adv_clip=args.adv_clip).reshape(bg)

            tau_mc = torch.rand(bg, args.N_mc, device=device)
            eps_mc = torch.randn(bg, args.N_mc, 4, latent_size, latent_size, device=device)
            ell_t_list = []
            for j in range(args.N_mc):
                ell_j = per_sample_loss_chunked(
                    teacher,
                    x1_exp,
                    model_kwargs_g,
                    transport,
                    t=tau_mc[:, j],
                    noise=eps_mc[:, j],
                    chunk_size=args.bg_chunk_size,
                )
                ell_t_list.append(ell_j)
            ell_t_cached = torch.stack(ell_t_list, dim=1)

            x1_hq = teacher_ode_sample(
                teacher,
                transport,
                vae,
                base_y,
                num_steps=args.T_hq,
                cfg_scale=args.cfg_scale,
                latent_size=latent_size,
                device=device,
            )
            tau_hq = torch.rand(b, device=device)
            eps_hq = torch.randn_like(x1_hq)
            u_hq = x1_hq - eps_hq
            x_tau_hq = tau_hq.view(-1, 1, 1, 1) * x1_hq + (1.0 - tau_hq).view(-1, 1, 1, 1) * eps_hq

        # Phase 2: K updates with frozen phase-1 tensors
        k_actual = 0
        last_ratio = torch.ones(bg, device=device)
        last_l_grpo = torch.tensor(0.0, device=device)
        last_l_mse = torch.tensor(0.0, device=device)

        for k in range(args.K):
            with autocast("cuda", enabled=args.amp):
                log_ratio, ratio = compute_elbo_ratio(
                    student,
                    x1_exp,
                    {**model_kwargs_g, "_chunk_size": args.bg_chunk_size},
                    transport,
                    tau_mc,
                    eps_mc,
                    ell_t_cached,
                )
                mean_ratio = ratio.mean().item()
                if mean_ratio > args.ratio_stop or mean_ratio < 1.0 / args.ratio_stop:
                    break

                l_grpo = ppo_clip_loss(ratio, advantage, eps_clip=args.eps_clip)
                pred_hq = student(x_tau_hq.detach(), tau_hq, y=base_y)
                l_mse = F.mse_loss(pred_hq, u_hq.detach())
                loss = l_grpo + args.lambda_mse * l_mse

            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            k_actual += 1
            last_ratio = ratio.detach()
            last_l_grpo = l_grpo.detach()
            last_l_mse = l_mse.detach()

        running["reward_mean"] += rewards.mean().item()
        running["reward_std"] += rewards.std().item()
        running["ratio"] += last_ratio.mean().item()
        running["clip_fraction"] += ((last_ratio < (1 - args.eps_clip)) | (last_ratio > (1 + args.eps_clip))).float().mean().item()
        running["l_grpo"] += last_l_grpo.item()
        running["l_mse"] += last_l_mse.item()
        running["k_actual"] += float(k_actual)
        running["count"] += 1

        if step % args.log_every == 0:
            dt = max(time() - tic, 1e-6)
            c = running["count"]
            avg_loss = running["l_grpo"] / c + args.lambda_mse * (running["l_mse"] / c)
            print(
                f"[step {step:05d}] "
                f"reward={running['reward_mean']/c:.4f}±{running['reward_std']/c:.4f} "
                f"ratio={running['ratio']/c:.4f} clip_frac={running['clip_fraction']/c:.4f} "
                f"k={running['k_actual']/c:.2f} "
                f"L_grpo={running['l_grpo']/c:.6f} L_mse={running['l_mse']/c:.6f} "
                f"it/s={args.log_every/dt:.2f}"
            )
            pbar.set_postfix(loss=f"{avg_loss:.4f}", ratio=f"{running['ratio']/c:.3f}", ips=f"{args.log_every/dt:.2f}")
            running = {k: 0.0 for k in running}
            running["count"] = 0
            tic = time()

        if args.sample_every > 0 and step % args.sample_every == 0:
            student.eval()
            with torch.no_grad():
                vis_labels = torch.randint(args.num_classes, size=(16,), device=device)
                _, vis_pixels = student_ode_sample(
                    student,
                    transport,
                    vae,
                    vis_labels,
                    num_steps=1,
                    cfg_scale=args.cfg_scale,
                    latent_size=latent_size,
                    device=device,
                    return_pixel=True,
                )
            save_image(
                vis_pixels,
                os.path.join(sample_dir, f"step_{step:06d}.png"),
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
            student.train()

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            torch.save(
                {"model": student.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args), "step": step},
                os.path.join(args.results_dir, f"step_{step:06d}.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/elbo_grpo")
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=list(SiT_models.keys()))
    parser.add_argument("--vae", type=str, default="mse", choices=["ema", "mse"])
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--T-exp", type=int, default=5)
    parser.add_argument("--T-hq", type=int, default=50)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--N-mc", type=int, default=4)
    parser.add_argument("--eps-clip", type=float, default=0.05)
    parser.add_argument("--adv-clip", type=float, default=2.0)
    parser.add_argument("--lambda-mse", type=float, default=0.1)
    parser.add_argument("--ratio-stop", type=float, default=1.5)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--bg-chunk-size", type=int, default=4, help="Chunk size for BG forwards to reduce VRAM.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total-iters", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=0, help="<=0 disables sample image saving")
    parser.add_argument("--ckpt-every", type=int, default=0, help="<=0 disables checkpoint saving")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training for lower VRAM.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
