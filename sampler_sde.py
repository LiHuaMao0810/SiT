"""
SDE baseline utilities.
"""
import math

import torch


def _forward_with_cfg(model, x, t, y, cfg_scale):
    n = x.shape[0]
    x_cfg = torch.cat([x, x], dim=0)
    y_null = torch.full((n,), 1000, device=y.device, dtype=torch.long)
    y_cfg = torch.cat([y, y_null], dim=0)
    t_cfg = torch.cat([t, t], dim=0)
    out_cfg = model.forward_with_cfg(x_cfg, t_cfg, y_cfg, cfg_scale)
    return out_cfg[:n]


def _forward_with_cfg_chunked(model, x, t, y, cfg_scale, chunk_size):
    if chunk_size is None or chunk_size <= 0 or x.shape[0] <= chunk_size:
        return _forward_with_cfg(model, x, t, y, cfg_scale)
    out = []
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out.append(_forward_with_cfg(model, x[start:end], t[start:end], y[start:end], cfg_scale))
    return torch.cat(out, dim=0)


def _gaussian_log_prob(x_next, mu, std):
    diff = x_next - mu
    flat = diff.flatten(start_dim=1)
    dim = flat.shape[1]
    quad = flat.pow(2).sum(dim=1) / (std ** 2)
    const = dim * math.log(2.0 * math.pi * (std ** 2))
    return -0.5 * (quad + const)


@torch.no_grad()
def sde_sample_with_logprob(
    model,
    z,
    y,
    num_steps,
    cfg_scale=1.5,
    sigma=0.1,
    chunk_size=None,
    return_noises=False,
):
    """
    Roll out SDE trajectory and accumulate path log-probability.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    x = z
    batch = z.shape[0]
    log_prob = torch.zeros(batch, device=z.device, dtype=torch.float32)
    saved_noises = []
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device, dtype=z.dtype)

    for i in range(num_steps):
        t_cur = torch.full((batch,), ts[i], device=z.device, dtype=z.dtype)
        dt = ts[i + 1] - ts[i]  # negative
        std = sigma * math.sqrt(float(abs(dt)))
        v = _forward_with_cfg_chunked(model, x, t_cur, y, cfg_scale, chunk_size)
        mu = x + v * dt
        noise = torch.randn_like(x)
        x = mu + std * noise
        log_prob = log_prob + _gaussian_log_prob(x, mu, std).float()
        if return_noises:
            saved_noises.append(noise.detach())

    if return_noises:
        return x, log_prob, saved_noises
    return x, log_prob


def sde_logprob_recompute(
    model,
    z,
    saved_noises,
    y,
    num_steps,
    cfg_scale=1.5,
    sigma=0.1,
    chunk_size=None,
):
    """
    Recompute trajectory log-probability with fixed saved noises.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")
    if len(saved_noises) != num_steps:
        raise ValueError("saved_noises length must equal num_steps.")

    x = z
    batch = z.shape[0]
    log_prob = torch.zeros(batch, device=z.device, dtype=torch.float32)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=z.device, dtype=z.dtype)

    for i in range(num_steps):
        t_cur = torch.full((batch,), ts[i], device=z.device, dtype=z.dtype)
        dt = ts[i + 1] - ts[i]
        std = sigma * math.sqrt(float(abs(dt)))
        v = _forward_with_cfg_chunked(model, x, t_cur, y, cfg_scale, chunk_size)
        mu = x + v * dt
        noise = saved_noises[i].to(device=z.device, dtype=z.dtype)
        x = mu + std * noise
        log_prob = log_prob + _gaussian_log_prob(x, mu, std).float()

    return log_prob
