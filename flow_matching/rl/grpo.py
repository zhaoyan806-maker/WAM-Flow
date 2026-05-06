from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class GRPOLossConfig:
    """Runtime GRPO loss settings.

    beta and clip_epsilon are intentionally required at runtime because the
    paper/config notes do not provide fixed values.
    """

    beta: float
    clip_epsilon: float
    normalize_advantages: bool = True
    eps: float = 1e-6


def grouped_advantages(
    rewards: torch.Tensor,
    group_size: int = 3,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute group-relative advantages for GRPO.

    Rewards are expected in flattened order: [scene0_sample0, scene0_sample1,
    scene0_sample2, scene1_sample0, ...] when group_size is 3.
    """

    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length ({rewards.numel()}) must be divisible by group_size ({group_size})"
        )

    grouped_rewards = rewards.reshape(-1, group_size)
    advantages = grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)
    if normalize:
        advantages = advantages / (grouped_rewards.std(dim=1, keepdim=True).clamp_min(eps))
    return advantages.reshape_as(rewards)


def sequence_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Gather summed log-probabilities for masked target tokens."""

    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    masked_token_log_probs = token_log_probs * mask.to(token_log_probs.dtype)
    return masked_token_log_probs.sum(dim=-1)


def clipped_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    config: GRPOLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the clipped GRPO objective with a reference-model KL penalty."""

    ratio = torch.exp(current_logprobs - old_logprobs.detach())
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - config.clip_epsilon,
        max=1.0 + config.clip_epsilon,
    )
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    kl_loss = (current_logprobs - reference_logprobs.detach()).mean()
    loss = policy_loss + config.beta * kl_loss

    logs = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "ratio_mean": float(ratio.detach().mean().cpu()),
    }
    return loss, logs
