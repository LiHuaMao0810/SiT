"""
Sampling wrappers for teacher/student SiT models.
"""
import torch
from torch.amp import autocast

from transport import Sampler


def _ode_sample(model, transport, vae, y, num_steps, cfg_scale=1.5, latent_size=32, device="cuda", z=None, return_pixel=False):
    batch = y.shape[0]
    if z is None:
        z = torch.randn(batch, 4, latent_size, latent_size, device=device)
    else:
        z = z.to(device=device)
    y = y.to(device=device, dtype=torch.long)

    z_cfg = torch.cat([z, z], dim=0)
    y_null = torch.full((batch,), 1000, device=device, dtype=torch.long)
    y_cfg = torch.cat([y, y_null], dim=0)

    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(num_steps=num_steps)
    # ODE integration with adaptive solver (e.g., dopri5) is numerically fragile in fp16.
    # Force fp32 for sampler/model forward here, even if caller enables AMP globally.
    with autocast("cuda", enabled=False):
        samples = sample_fn(
            z_cfg.float(),
            model.forward_with_cfg,
            y=y_cfg,
            cfg_scale=cfg_scale,
        )[-1]
    samples = samples[:batch]
    if not return_pixel:
        return samples
    pixels = vae.decode(samples / 0.18215).sample
    return samples, pixels


@torch.no_grad()
def teacher_ode_sample(
    teacher,
    transport,
    vae,
    y,
    num_steps,
    cfg_scale=1.5,
    latent_size=32,
    device="cuda",
    z=None,
    return_pixel=False,
):
    return _ode_sample(
        teacher,
        transport,
        vae,
        y,
        num_steps,
        cfg_scale=cfg_scale,
        latent_size=latent_size,
        device=device,
        z=z,
        return_pixel=return_pixel,
    )


@torch.no_grad()
def student_ode_sample(
    student,
    transport,
    vae,
    y,
    num_steps=1,
    cfg_scale=1.5,
    latent_size=32,
    device="cuda",
    z=None,
    return_pixel=False,
):
    return _ode_sample(
        student,
        transport,
        vae,
        y,
        num_steps,
        cfg_scale=cfg_scale,
        latent_size=latent_size,
        device=device,
        z=z,
        return_pixel=return_pixel,
    )
