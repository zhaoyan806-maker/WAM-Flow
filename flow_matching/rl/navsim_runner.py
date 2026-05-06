import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch



@dataclass
class ParsedTrajectory:
    trajectory: "Trajectory"
    numbers: list[float]
    parse_failed: bool


def extract_trajectory_numbers(text: str, expected_count: int = 16) -> tuple[list[float], bool]:
    """Extract and pad trajectory numbers from model text output."""

    numbers = [float(n) for n in re.findall(r"[-+]?[0-9]*\.?[0-9]+", text.strip())]
    parse_failed = len(numbers) < expected_count

    if len(numbers) == 0:
        numbers = [0.0] * expected_count
    elif len(numbers) < expected_count:
        last_pair = numbers[-2:] if len(numbers) >= 2 else [numbers[-1], numbers[-1]]
        while len(numbers) < expected_count:
            numbers.extend(last_pair)
        numbers = numbers[:expected_count]
    elif len(numbers) > expected_count:
        numbers = numbers[:expected_count]

    return numbers, parse_failed


def tokens_to_navsim_trajectory(tokenizer, token_ids: torch.Tensor, heading_model=None) -> ParsedTrajectory:
    """Decode sampled tokens and convert them to a NAVSIM Trajectory."""

    from navsim.common.dataclasses import Trajectory

    text = tokenizer.decode(token_ids.detach().cpu().tolist(), skip_special_tokens=True)
    numbers, parse_failed = extract_trajectory_numbers(text)
    xy = torch.tensor(numbers, dtype=torch.float32).reshape(1, 8, 2)

    if heading_model is None:
        headings = np.zeros((8, 1), dtype=np.float32)
    else:
        with torch.no_grad():
            model_device = next(heading_model.parameters()).device
            headings = heading_model(xy.to(model_device)).detach().cpu().reshape(8, 1).numpy()

    poses = np.concatenate([xy.squeeze(0).numpy(), headings], axis=1).astype(np.float32)
    return ParsedTrajectory(trajectory=Trajectory(poses), numbers=numbers, parse_failed=parse_failed)


def proxy_navsim_metrics_from_numbers(numbers: list[float]) -> dict[str, float]:
    """Build lightweight proxy metrics when the NAVSIM simulator is unavailable.

    This keeps GRPO development runnable before wiring the expensive PDM scorer.
    Replace these proxies with metric-cache-backed NC/DAC/EP/TTC/Comfort values
    for final experiments.
    """

    xy = np.asarray(numbers, dtype=np.float32).reshape(8, 2)
    x = xy[:, 0]
    y = xy[:, 1]

    nc = 1.0
    dac = float(np.all(np.abs(y) <= 6.0) and np.all(x >= -2.0))
    ep = float(np.clip(x[-1] / 50.0, 0.0, 1.0))
    ttc = 1.0

    velocity = np.diff(xy, axis=0)
    acceleration = np.diff(velocity, axis=0) if len(velocity) > 1 else np.zeros((1, 2), dtype=np.float32)
    accel_norm = float(np.linalg.norm(acceleration, axis=1).mean()) if len(acceleration) else 0.0
    comfort = float(1.0 / (1.0 + accel_norm))

    return {"NC": nc, "DAC": dac, "EP": ep, "TTC": ttc, "Comfort": comfort}


def collect_metric_tensors(metric_rows: Iterable[Mapping[str, float]], device=None) -> dict[str, torch.Tensor]:
    """Convert NAVSIM metric dictionaries to tensors keyed by metric name."""

    rows = list(metric_rows)
    keys = {key for row in rows for key in row.keys()}
    return {
        key: torch.tensor([float(row.get(key, 0.0)) for row in rows], device=device, dtype=torch.float32)
        for key in keys
    }
