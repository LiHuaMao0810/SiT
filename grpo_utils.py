"""
ELBO ratio, GRPO advantage, and PPO-clip utilities.
"""
import torch


def _slice_model_kwargs(model_kwargs, start, end):
    sliced = {}
    for key, value in model_kwargs.items():
        if key.startswith("_"):
            continue
        if torch.is_tensor(value) and value.shape[0] >= end:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def compute_elbo_ratio(student, x1_exp, model_kwargs, transport, tau_mc, eps_mc, ell_T_cached):
    """
    Compute ELBO ratio with cached teacher losses.

    Returns:
        log_ratio: (BG,)
        ratio: (BG,)
    """
    x1_exp = x1_exp.detach()
    batch = x1_exp.shape[0]
    n_mc = tau_mc.shape[1]
    ell_eta_mean = torch.zeros(batch, device=x1_exp.device, dtype=x1_exp.dtype)
    chunk_size = model_kwargs.get("_chunk_size", None)
    if chunk_size is None or chunk_size <= 0:
        chunk_size = batch
    for j in range(n_mc):
        chunk_losses = []
        for start in range(0, batch, chunk_size):
            end = min(start + chunk_size, batch)
            chunk_losses.append(
                transport.per_sample_loss(
                    student,
                    x1_exp[start:end],
                    _slice_model_kwargs(model_kwargs, start, end),
                    t=tau_mc[start:end, j],
                    noise=eps_mc[start:end, j],
                )
            )
        ell_eta_j = torch.cat(chunk_losses, dim=0)
        ell_eta_mean = ell_eta_mean + ell_eta_j / n_mc
    log_ratio = ell_T_cached.mean(dim=-1) - ell_eta_mean
    ratio = log_ratio.clamp(-5.0, 5.0).exp()
    return log_ratio, ratio


def compute_grpo_advantage(rewards, adv_clip=2.0):
    """
    Group-normalized advantage for GRPO.
    rewards: (B, G)
    """
    r_mean = rewards.mean(dim=1, keepdim=True)
    r_std = rewards.std(dim=1, keepdim=True)
    advantage = torch.where(
        r_std < 1e-8,
        torch.zeros_like(rewards),
        (rewards - r_mean) / (r_std + 1e-8),
    )
    return advantage.clamp(-adv_clip, adv_clip)


def ppo_clip_loss(ratio, advantage, eps_clip=0.05):
    """
    PPO-clip surrogate loss (negative for gradient descent optimizer).
    """
    if ratio.shape != advantage.shape:
        raise ValueError(f"ratio shape {ratio.shape} must match advantage shape {advantage.shape}")
    clipped = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip)
    loss = -torch.min(ratio * advantage, clipped * advantage).mean()
    return loss
