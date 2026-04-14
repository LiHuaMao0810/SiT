import argparse
import os
from time import time

import torch
from torch.amp import GradScaler, autocast
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm
from torchvision.utils import save_image

from download import find_model
from models import SiT_models
from sit_sampler import student_ode_sample, teacher_ode_sample
from train_utils import parse_transport_args
from transport import create_transport


def build_model(model_name, latent_size, num_classes, ckpt_path, device):
    model = SiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        learn_sigma=True,
    ).to(device)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    return model


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
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # foreach=False avoids extra temporary buffers during optimizer.step, reducing peak VRAM.
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.0,
        foreach=False,
    )
    scaler = GradScaler("cuda", enabled=args.amp)
    student.train()

    running_loss = 0.0
    tic = time()

    pbar = tqdm(range(1, args.total_iters + 1), disable=args.no_tqdm, dynamic_ncols=True, desc="distill")
    for step in pbar:
        y = torch.randint(args.num_classes, size=(args.batch_size,), device=device)

        with torch.no_grad():
            x1_hq = teacher_ode_sample(
                teacher,
                transport,
                vae,
                y,
                num_steps=args.T_hq,
                cfg_scale=args.cfg_scale,
                latent_size=latent_size,
                device=device,
            )
            tau_hq = torch.rand(args.batch_size, device=device)
            eps_hq = torch.randn_like(x1_hq)
            u_hq = x1_hq - eps_hq
            x_tau_hq = tau_hq.view(-1, 1, 1, 1) * x1_hq + (1.0 - tau_hq).view(-1, 1, 1, 1) * eps_hq

        with autocast("cuda", enabled=args.amp):
            pred = student(x_tau_hq.detach(), tau_hq, y=y)
            loss = torch.nn.functional.mse_loss(pred, u_hq.detach())

        optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item()

        if step % args.log_every == 0:
            dt = max(time() - tic, 1e-6)
            avg_loss = running_loss / args.log_every
            print(f"[step {step:05d}] loss={avg_loss:.6f} it/s={args.log_every / dt:.2f}")
            pbar.set_postfix(loss=f"{avg_loss:.4f}", ips=f"{args.log_every / dt:.2f}")
            running_loss = 0.0
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
            ckpt_path = os.path.join(args.results_dir, f"step_{step:06d}.pt")
            torch.save(
                {
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "step": step,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/distill_only")
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=list(SiT_models.keys()))
    parser.add_argument("--vae", type=str, default="mse", choices=["ema", "mse"])
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--T-hq", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
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
