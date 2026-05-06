from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class RolloutOutput:
    """Container for sampled token sequences and rollout metadata."""

    tokens: torch.Tensor
    old_logprobs: torch.Tensor
    denoise_steps: torch.Tensor


def expand_for_group(batch: torch.Tensor, group_size: int) -> torch.Tensor:
    """Repeat each batch row group_size times while preserving scene grouping."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return batch.repeat_interleave(group_size, dim=0)


def repeated_denoise_steps(
    denoise_steps: Sequence[int], num_scenes: int, group_size: int, device: torch.device
) -> torch.Tensor:
    """Assign denoise-step variants to each grouped candidate."""

    if len(denoise_steps) == 0:
        raise ValueError("denoise_steps must not be empty")
    step_values = [denoise_steps[i % len(denoise_steps)] for i in range(group_size)]
    return torch.tensor(step_values, device=device).repeat(num_scenes)


def expand_data_info(data_info: dict, batch_size: int, group_size: int) -> dict:
    """Repeat tensor values in data_info for grouped rollouts."""

    expanded = {}
    for key, value in data_info.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
            expanded[key] = expand_for_group(value, group_size)
        else:
            expanded[key] = value
    return expanded


def build_text_x_init(input_ids: torch.Tensor, text_token_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Create flow-matching initial tokens by noising only assistant text tokens."""

    random_tokens = torch.randint(vocab_size, input_ids.shape, device=input_ids.device, dtype=input_ids.dtype)
    return random_tokens * text_token_mask.to(input_ids.dtype) + input_ids * (1 - text_token_mask.to(input_ids.dtype))


@torch.no_grad()
def sample_grouped_rollouts(
    solver,
    input_ids: torch.Tensor,
    data_info: dict,
    vocab_size: int,
    group_size: int = 3,
    denoise_steps: Sequence[int] = (1, 3, 5),
    dtype_categorical: torch.dtype = torch.float32,
) -> RolloutOutput:
    """Sample grouped trajectories from the WAM-Flow discrete solver.

    This helper keeps the rollout policy in project code; NAVSIM metric scoring
    and log-prob recomputation are handled by train_grpo.py / navsim_runner.py.
    """

    batch_size = input_ids.shape[0]
    grouped_input_ids = expand_for_group(input_ids, group_size)
    grouped_data_info = expand_data_info(data_info, batch_size, group_size)

    x_init = build_text_x_init(grouped_input_ids, grouped_data_info["text_token_mask"], vocab_size)
    assigned_steps = repeated_denoise_steps(denoise_steps, batch_size, group_size, input_ids.device)

    samples = []
    for idx, steps in enumerate(assigned_steps.tolist()):
        sample = solver.sample(
            x_init=x_init[idx : idx + 1],
            step_size=1.0 / steps,
            return_intermediates=False,
            div_free=0,
            dtype_categorical=dtype_categorical,
            datainfo={k: v[idx : idx + 1] if isinstance(v, torch.Tensor) and v.shape[:1] == x_init.shape[:1] else v for k, v in grouped_data_info.items()},
            cfg_scale=0,
        )
        samples.append(sample)

    tokens = torch.cat(samples, dim=0)
    old_logprobs = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.float32)
    return RolloutOutput(tokens=tokens, old_logprobs=old_logprobs, denoise_steps=assigned_steps)
