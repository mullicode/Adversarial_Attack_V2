from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Union

import torch
from PIL import Image
from torchvision.transforms import Normalize

from perturbnet.model import LABELS, PREPROCESS, WEIGHTS, load_efficientnet_v2_l

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE = 480
EXPECTED_FEATURE_MAP_SHAPE = (1, 1280, 15, 15)
EXPECTED_RAW_BASE_SHAPE = (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

# Validator/RMSE constraint: each raw RGB channel may change by at most one ±1/255 step.
PIXEL_STEP_RAW = 1.0 / 255.0

ImageInput = Union[Image.Image, torch.Tensor]


def _get_preprocess_mean_std(preprocess: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean = getattr(preprocess, "mean", None)
    std = getattr(preprocess, "std", None)
    if mean is not None and std is not None:
        return tuple(float(v) for v in mean), tuple(float(v) for v in std)

    for transform in getattr(preprocess, "transforms", ()):
        if isinstance(transform, Normalize):
            return (
                tuple(float(v) for v in transform.mean),
                tuple(float(v) for v in transform.std),
            )

    raise RuntimeError("Could not extract mean/std from PREPROCESS.")


NORM_MEAN, NORM_STD = _get_preprocess_mean_std(PREPROCESS)


def _mean_std_tensors(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(NORM_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


@dataclass(frozen=True)
class ModelSetup:
    model: torch.nn.Module
    weights: Any
    preprocess: Any
    categories: list[str]
    feature_shape: tuple[int, ...]


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
    if change.direction not in (-1, 1):
        return False
    if x_raw.ndim != 4:
        return False

    _, channel_count, height, width = x_raw.shape
    if not (0 <= change.channel < channel_count):
        return False
    if not (0 <= change.y < height):
        return False
    if not (0 <= change.x < width):
        return False

    value = float(x_raw[0, change.channel, change.y, change.x].item())
    if change.direction == 1:
        return value <= 1.0 - PIXEL_STEP_RAW + 1e-9
    return value >= PIXEL_STEP_RAW - 1e-9


def normalize_raw(x_raw: torch.Tensor) -> torch.Tensor:
    """Map raw [0,1] tensor to model input: (x_raw - mean) / std."""
    mean, std = _mean_std_tensors(device=x_raw.device, dtype=x_raw.dtype)
    return (x_raw - mean) / std


def denormalize(x_norm: torch.Tensor) -> torch.Tensor:
    """Recover raw [0,1] tensor from normalized model input."""
    mean, std = _mean_std_tensors(device=x_norm.device, dtype=x_norm.dtype)
    return x_norm * std + mean


def _ensure_bchw_float(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = image.to(device=device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        x = x.float()
    elif x.dtype != torch.float32:
        x = x.to(dtype=torch.float32)
    if float(x.max()) > 1.0 + 1e-3:
        x = x / 255.0
    return x


def _preprocess_normalized(image: ImageInput, device: torch.device) -> torch.Tensor:
    """Run canonical PREPROCESS from perturbnet.model (same path as validator)."""
    if isinstance(image, Image.Image):
        x_in: Any = image.convert("RGB")
    else:
        x_in = _ensure_bchw_float(image, device=device)

    with torch.no_grad():
        x_norm = PREPROCESS(x_in)
    if x_norm.ndim == 3:
        x_norm = x_norm.unsqueeze(0)
    return x_norm.to(device=device, dtype=torch.float32)


def preprocess_to_raw_480(image: ImageInput, device: torch.device) -> torch.Tensor:
    """
    Convert PIL or tensor image to raw model input [1,3,480,480] before normalization.

    Uses PREPROCESS from perturbnet.model for resize/crop/normalize, then denormalize
    to recover the raw [0,1] tensor where ±1/255 perturbations are applied.
    """
    x_raw = denormalize(_preprocess_normalized(image=image, device=device))
    if tuple(x_raw.shape) != EXPECTED_RAW_BASE_SHAPE:
        raise RuntimeError(
            f"Unexpected raw base shape {tuple(x_raw.shape)}; expected {EXPECTED_RAW_BASE_SHAPE}"
        )
    return x_raw


def chw_to_raw_base(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Backward-compatible alias for decoded CHW images."""
    return preprocess_to_raw_480(image_chw, device=device)


def raw_to_model_input(
    x_raw: torch.Tensor,
    delta_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize raw tensor before model forward; clamp only when delta is applied."""
    x = x_raw if delta_raw is None else x_raw + delta_raw
    if delta_raw is not None:
        x = x.clamp(0.0, 1.0)
    return normalize_raw(x)


def logits_from_raw(
    model: torch.nn.Module,
    x_raw: torch.Tensor,
    delta_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward pass on raw-space tensor without re-running resize/center-crop."""
    return model(raw_to_model_input(x_raw=x_raw, delta_raw=delta_raw))


def verify_feature_map_shape(model: torch.nn.Module, device: torch.device) -> tuple[int, ...]:
    """Confirm model.features output for 480×480 input is [1, 1280, 15, 15]."""
    model.eval()
    with torch.no_grad():
        probe = raw_to_model_input(torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, device=device))
        features = model.features(probe)
    shape = tuple(int(dim) for dim in features.shape)
    if shape != EXPECTED_FEATURE_MAP_SHAPE:
        raise RuntimeError(
            f"Unexpected EfficientNetV2-L feature shape {shape}; "
            f"expected {EXPECTED_FEATURE_MAP_SHAPE}"
        )
    return shape


def setup_model(device: torch.device, *, verify_features: bool = True) -> ModelSetup:
    """Load EfficientNetV2-L, weights, transforms, and class labels for the attack pipeline."""
    model = load_efficientnet_v2_l(device)
    categories = list(LABELS)
    feature_shape: tuple[int, ...] = ()
    if verify_features:
        feature_shape = verify_feature_map_shape(model=model, device=device)
        logger.info(
            "setup_model verified feature map shape=%s for input=%sx%s norm_mean=%s norm_std=%s",
            feature_shape,
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
            NORM_MEAN,
            NORM_STD,
        )
    return ModelSetup(
        model=model,
        weights=WEIGHTS,
        preprocess=PREPROCESS,
        categories=categories,
        feature_shape=feature_shape,
    )


def init_attack_state(
    model: torch.nn.Module,
    image_chw: torch.Tensor,
    true_idx: int,
    device: torch.device,
) -> AttackState:
    """Beginning step: build raw-base tensors and baseline logits/gap for the attack loop."""
    x_raw_base = preprocess_to_raw_480(image=image_chw, device=device)
    delta_raw = torch.zeros_like(x_raw_base)
    changed_mask = torch.zeros_like(x_raw_base, dtype=torch.bool)
    with torch.no_grad():
        logits_batched = logits_from_raw(model=model, x_raw=x_raw_base)
    current_logits = _logits_row(logits_batched).detach().to(dtype=torch.float32)

    pred_idx = int(current_logits.argmax().item())
    if pred_idx != true_idx:
        logger.warning(
            "Provided true_idx=%s does not match clean model prediction=%s",
            true_idx,
            pred_idx,
        )

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
