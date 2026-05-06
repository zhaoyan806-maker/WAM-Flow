from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Union

import torch


@dataclass
class RewardWeights:
    """Weights used by the NAVSIM GRPO reward."""

    safety_metrics: Sequence[str] = field(default_factory=lambda: ("NC", "DAC"))
    performance_weights: Mapping[str, float] = field(
        default_factory=lambda: {"EP": 5.0, "TTC": 5.0, "Comfort": 2.0}
    )


def _to_tensor(value, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def compute_navsim_reward(
    metrics: Mapping[str, Union[torch.Tensor, float, Sequence[float]]],
    weights: Optional[RewardWeights] = None,
) -> torch.Tensor:
    """Compute the simulated NAVSIM GRPO reward.

    The reward follows the user-provided setting:
    safety metrics are NC and DAC, while performance terms are weighted as
    EP:5, TTC:5, Comfort:2. Safety is used as a multiplicative gate so that
    unsafe candidates are strongly suppressed before policy optimization.
    """

    weights = weights or RewardWeights()
    device = None
    for value in metrics.values():
        if isinstance(value, torch.Tensor):
            device = value.device
            break

    safety_gate = None
    for name in weights.safety_metrics:
        metric_value = _to_tensor(metrics.get(name, 0.0), device=device)
        safety_gate = metric_value if safety_gate is None else safety_gate * metric_value

    if safety_gate is None:
        safety_gate = _to_tensor(1.0, device=device)

    performance_reward = None
    for name, weight in weights.performance_weights.items():
        metric_value = _to_tensor(metrics.get(name, 0.0), device=device)
        weighted_value = float(weight) * metric_value
        performance_reward = weighted_value if performance_reward is None else performance_reward + weighted_value

    if performance_reward is None:
        performance_reward = _to_tensor(0.0, device=device)

    return safety_gate * performance_reward
