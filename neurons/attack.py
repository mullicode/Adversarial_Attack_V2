from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torchvision.transforms import Normalize

from perturbnet.model import PREPROCESS

# Derived from PREPROCESS (EfficientNetV2-L IMAGENET1K_V1): resize/crop 480, [0,1], normalize.
_NORM_TRANSFORM: Normalize | None = None
for _transform in getattr(PREPROCESS, "transforms", ()):
    if isinstance(_transform, Normalize):
        _NORM_TRANSFORM = _transform
        break

NORM_MEAN = tuple(_NORM_TRANSFORM.mean) if _NORM_TRANSFORM is not None else (0.5, 0.5, 0.5)
NORM_STD = tuple(_NORM_TRANSFORM.std) if _NORM_TRANSFORM is not None else (0.5, 0.5, 0.5)

# Perturb in raw [0,1] space; validator measures pixel diffs in this space.
PIXEL_STEP_RAW = 1.0 / 255.0
# Equivalent step after (x - 0.5) / 0.5 normalization.
PIXEL_STEP_NORM = PIXEL_STEP_RAW / float(NORM_STD[0])


@dataclass(frozen=True)
class PixelChange:
    channel: int
    y: int
    x: int
    direction: int  # -1 or +1 for one ±1/255 raw step


@dataclass
class AcceptedGroup:
    competitor_idx: int
    feature_cell: tuple[int, int]
    image_box: tuple[int, int, int, int]
    pixels: list[PixelChange]
    gap_before: float
    gap_after: float
    gain: float
    gain_per_pixel: float
    round_id: int


@dataclass
class AttackState:
    x_raw_base: torch.Tensor  # [1,3,H,W] float32 in [0,1], pre-normalize
    delta_raw: torch.Tensor  # [1,3,H,W] float32, values in {-1/255, 0, +1/255}
    changed_mask: torch.Tensor  # [1,3,H,W] bool
    true_idx: int
    current_competitor_idx: int
    current_logits: torch.Tensor  # [num_classes] float32, detached
    current_gap: float  # logit_true - logit_current_competitor_idx
    accepted_groups: list[AcceptedGroup] = field(default_factory=list)

    @property
    def current_x_raw(self) -> torch.Tensor:
        return (self.x_raw_base + self.delta_raw).clamp(0.0, 1.0)


def _logits_row(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits[0]
    return logits


def best_competitor_idx(logits: torch.Tensor, true_idx: int) -> int:
    """Index of the strongest non-true class."""
    row = _logits_row(logits).clone()
    row[true_idx] = float("-inf")
    return int(row.argmax().item())


def competitor_gap(logits: torch.Tensor, true_idx: int, competitor_idx: int) -> float:
    """gap_k = logit_true - logit_competitor (positive means true class still leads)."""
    row = _logits_row(logits)
    return float((row[true_idx] - row[competitor_idx]).item())


def untargeted_gap(logits: torch.Tensor, true_idx: int) -> float:
    """logit_true - max_other_logit; equals gap vs best competitor."""
    competitor_idx = best_competitor_idx(logits, true_idx)
    return competitor_gap(logits, true_idx, competitor_idx)


def is_flip_success(logits: torch.Tensor, true_idx: int) -> bool:
    return untargeted_gap(logits, true_idx) < 0.0


def can_apply_pixel_change(x_raw: torch.Tensor, change: PixelChange) -> bool:
    """Skip saturated pixels so ±1/255 steps are real changes, not clamp no-ops."""
    value = float(x_raw[0, change.channel, change.y, change.x].item())
    if change.direction == 1:
        return value <= 1.0 - PIXEL_STEP_RAW + 1e-9
    if change.direction == -1:
        return value >= PIXEL_STEP_RAW - 1e-9
    return False


def chw_to_raw_base(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Map decoded CHW image to raw model input in [0,1] (resize/crop, stop before Normalize)."""
    x = image_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    transforms = getattr(PREPROCESS, "transforms", None)
    if transforms is None:
        raise RuntimeError("PREPROCESS has no .transforms; cannot split raw vs normalize pipeline.")
    with torch.no_grad():
        for transform in transforms:
            if isinstance(transform, Normalize):
                break
            x = transform(x)
    return x.clamp(0.0, 1.0)


def raw_to_model_input(x_raw: torch.Tensor, delta_raw: torch.Tensor | None = None) -> torch.Tensor:
    """Apply EfficientNetV2-L normalize from PREPROCESS; delta is in raw [0,1] space."""
    if _NORM_TRANSFORM is None:
        raise RuntimeError("EfficientNet normalize transform is unavailable.")
    x = x_raw if delta_raw is None else (x_raw + delta_raw).clamp(0.0, 1.0)
    return _NORM_TRANSFORM(x)


def logits_from_raw(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    delta_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward pass on raw-space tensor without re-running resize/center-crop."""
    return model(raw_to_model_input(x_raw=x_raw, delta_raw=delta_raw))


def init_attack_state(
    model: torch.nn.Module,
    image_chw: torch.Tensor,
    true_idx: int,
    device: torch.device,
) -> AttackState:
    """Beginning step: build raw-base tensors and baseline logits/gap for the attack loop."""
    x_raw_base = chw_to_raw_base(image_chw=image_chw, device=device)
    delta_raw = torch.zeros_like(x_raw_base)
    changed_mask = torch.zeros_like(x_raw_base, dtype=torch.bool)
    with torch.no_grad():
        logits_batched = logits_from_raw(model=model, x_raw=x_raw_base)
    current_logits = _logits_row(logits_batched).detach().to(dtype=torch.float32)
    competitor_idx = best_competitor_idx(current_logits, true_idx)
    gap = competitor_gap(current_logits, true_idx, competitor_idx)
    return AttackState(
        x_raw_base=x_raw_base,
        delta_raw=delta_raw,
        changed_mask=changed_mask,
        true_idx=true_idx,
        current_competitor_idx=competitor_idx,
        current_logits=current_logits,
        current_gap=gap,
    )
