import argparse
import os

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import make_grid, save_image

from download import find_model
from models import SiT_models
from reward_utils import ClassifierReward
from sit_sampler import student_ode_sample
from train_utils import parse_transport_args
from transport import create_transport


def load_student(ckpt_path, model_name, latent_size, num_classes, device):
    model = SiT_models[model_name](input_size=latent_size, num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main(args):
    assert torch.cuda.is_available(), "This script requires CUDA."
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    latent_size = args.image_size // 8
    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    model = load_student(args.ckpt, args.model, latent_size, args.num_classes, device)
    reward_fn = ClassifierReward(device)

    reward_sum = 0.0
    reward_n = 0
    vis_pixels = []

    with torch.no_grad():
        remaining = args.num_samples
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            y = torch.randint(args.num_classes, size=(bs,), device=device)
            _, pixels = student_ode_sample(
                model,
                transport,
                vae,
                y,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                latent_size=latent_size,
                device=device,
                return_pixel=True,
            )
            rewards = reward_fn(pixels, y)
            reward_sum += rewards.sum().item()
            reward_n += bs
            if len(vis_pixels) < 16:
                vis_pixels.append(pixels[: max(0, 16 - len(vis_pixels))])
            remaining -= bs

    reward_mean = reward_sum / max(reward_n, 1)
    print(f"reward_mean={reward_mean:.6f}")
    with open(os.path.join(args.out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"reward_mean={reward_mean:.6f}\n")

    if vis_pixels:
        vis = torch.cat(vis_pixels, dim=0)[:16]
        grid = make_grid(vis, nrow=4, normalize=True, value_range=(-1, 1))
        save_image(grid, os.path.join(args.out_dir, "samples_grid.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="results/eval")
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=list(SiT_models.keys()))
    parser.add_argument("--vae", type=str, default="mse", choices=["ema", "mse"])
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
