from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from perturbnet.constants import MAX_LINF_DELTA, MIN_LINF_DELTA, MIN_PSNR_DB, MIN_SSIM, TIMEOUT_SECONDS
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import forward_logits_features, logits_for_images, predict_index, predict_label

logger = logging.getLogger(__name__)

FEATURE_MAP_SIZE = 15
MODEL_CROP_SIZE = 480
EXPECTED_FEATURE_MAP_SHAPE = (1, 1280, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
SPARSE_PIXEL_FRACTION = 0.05
DEFAULT_STEPS = 40

# Feature-cell box expansion in 480×480 crop space (one cell ≈ 32×32 px).
FEATURE_CELL_EXPAND_FIRST = 2.0
FEATURE_CELL_EXPAND_WIDE = 3.0
FEATURE_CELL_EXPAND_REFINE = 1.5

# Validator-visible pixel step in decoded [0, 1] image space.
PIXEL_STEP_RAW = 1.0 / 255.0

# Untargeted top-K competitor race (K=10 default for subnet timeout/model size).
DEFAULT_TOP_K = 10
TOP_K_FAST = 5
TOP_K_BALANCED = 10
TOP_K_STRONG = 20

# Region ranking defaults (validator timeout aware).
DEFAULT_TOP_REGIONS_PER_COMPETITOR = 6
TOP_REGIONS_FAST = 5
TOP_REGIONS_QUALITY = 15
REGION_PIXEL_GRAD_TOP_N = 32

# Discrete ±1/255 pixel selection inside ranked regions.
DEFAULT_TOP_PIXELS_PER_REGION = 8
DEFAULT_MAX_PIXELS_PER_STEP = 24
DEFAULT_MIN_GAIN_PER_PIXEL = 1e-6

# Region growing after a verified seed batch (step 13).
REGION_GROW_INITIAL_BATCH = 8
REGION_GROW_MAX_BATCH = 64
REGION_GROW_MIN_BATCH = 4
REGION_GROW_MAX_PIXELS_PER_REGION = 128
REGION_GROW_STRONG_GAIN_PER_PIXEL = 1e-4
REGION_GROW_CLOSE_TO_FLIP_GAP = 0.5
REGION_GROW_CLOSE_TO_FLIP_RATIO = 0.15
REGION_GROW_MAX_FAILURES = 2

# Beam search adapted to validator timeout (step 14).
DEFAULT_BEAM_WIDTH = 4
BEAM_WIDTH_FAST = 3
DEFAULT_BEAM_TOP_K = 10
DEFAULT_BEAM_TOP_REGIONS = 6
ATTACK_TIME_SEARCH_FRACTION = 0.60
ATTACK_TIME_PRUNE_FRACTION = 0.25
ATTACK_TIME_VALIDATE_FRACTION = 0.10
ATTACK_TIME_BUFFER_FRACTION = 0.05
ATTACK_SEARCH_ROUND_FRACTION = 0.75


@dataclass(frozen=True)
class CompetitorEntry:
    idx: int
    logit: float
    gap_k: float  # logit_true - logit_competitor


@dataclass(frozen=True)
class TopKCompetitorRace:
    true_idx: int
    true_logit: float
    competitors: tuple[CompetitorEntry, ...]

    @property
    def easiest(self) -> CompetitorEntry:
        return min(self.competitors, key=lambda entry: entry.gap_k)

    def is_success(self, logits: torch.Tensor) -> bool:
        row = _logits_row(logits)
        pred_idx = int(row.argmax().item())
        if pred_idx != self.true_idx:
            return True
        masked = row.clone()
        masked[self.true_idx] = float("-inf")
        max_other_logit = float(masked.max().item())
        return max_other_logit > self.true_logit


@dataclass(frozen=True)
class PixelChange:
    channel: int
    y: int
    x: int
    direction: int  # -1 or +1 for one ±1/255 step


@dataclass(frozen=True)
class PixelCandidate:
    channel: int
    y: int
    x: int
    direction: int  # -1 or +1
    grad_value: float
    predicted_gain: float  # |grad_value| * PIXEL_STEP_RAW

    def as_change(self) -> PixelChange:
        return PixelChange(
            channel=self.channel,
            y=self.y,
            x=self.x,
            direction=self.direction,
        )


@dataclass
class CompetitorGapGradients:
    """Gradients w.r.t. decoded raw image [3, H, W] through PREPROCESS."""

    competitor_idx: int
    gap_k: float
    logits: torch.Tensor  # [1, num_classes]
    features: torch.Tensor  # [1, 1280, 15, 15]
    pixel_grad_raw: torch.Tensor  # [3, H, W]
    feature_grad: torch.Tensor | None  # [1, 1280, 15, 15]


@dataclass(frozen=True)
class CompetitorGapCam:
    competitor_idx: int
    w_gap: torch.Tensor  # [1280]
    gap_cam: torch.Tensor  # [Hf, Wf] e.g. [15, 15]


@dataclass(frozen=True)
class PreprocessGeometry:
    """Approximate inverse of EfficientNetV2-L PREPROCESS resize + center-crop."""

    raw_h: int
    raw_w: int
    resized_h: int
    resized_w: int
    crop_top: float
    crop_left: float


@dataclass(frozen=True)
class CompetitorActivationMaps:
    """Activation-gradient maps for true-vs-competitor gap at feature resolution."""

    competitor_idx: int
    abs_map: torch.Tensor  # [Hf, Wf] general importance
    dir_map: torch.Tensor  # [Hf, Wf] signed gap sensitivity


@dataclass(frozen=True)
class RankedRegion:
    competitor_idx: int
    feature_cell: tuple[int, int]
    image_box: tuple[int, int, int, int]
    region_score: float
    gap_cam_term: float
    abs_map_term: float
    pixel_grad_density_term: float
    pixel_grad_density: float


@dataclass(frozen=True)
class TrialVerification:
    """Validator-style forward check for a trial delta batch."""

    accepted: bool
    flip_candidate: bool
    validator_would_pass: bool
    reason: str
    adv_try: torch.Tensor
    logits: torch.Tensor
    pred_idx: int
    norm: float
    rmse: float
    ssim: float
    psnr_db: float
    gap_before: float
    gap_after: float
    untargeted_gap_before: float
    untargeted_gap_after: float
    real_gain: float
    gain_per_pixel: float
    num_new_pixels: int


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
class RegionGrowSession:
    """Adaptive batch growth state for one ranked region."""

    region: RankedRegion
    batch_size: int = REGION_GROW_INITIAL_BATCH
    failure_count: int = 0
    pixels_applied: int = 0
    seed_accepted: bool = False
    stopped: bool = False


@dataclass
class RegionGrowResult:
    region: RankedRegion
    pixels_applied: int
    seed_accepted: bool
    stopped_reason: str
    best_verification: TrialVerification | None = None

    @property
    def flip_found(self) -> bool:
        return self.best_verification is not None and self.best_verification.flip_candidate


@dataclass(frozen=True)
class AttackTimeBudget:
    """Phase deadlines for search / prune / validate / buffer."""

    start: float
    search_end: float
    prune_end: float
    validate_end: float
    hard_end: float

    @classmethod
    def from_timeout(
        cls,
        timeout_seconds: float | int,
        *,
        search_fraction: float = ATTACK_TIME_SEARCH_FRACTION,
        prune_fraction: float = ATTACK_TIME_PRUNE_FRACTION,
        validate_fraction: float = ATTACK_TIME_VALIDATE_FRACTION,
        buffer_fraction: float = ATTACK_TIME_BUFFER_FRACTION,
    ) -> AttackTimeBudget:
        start = time.monotonic()
        total = float(timeout_seconds)
        search_end = start + total * float(search_fraction)
        prune_end = search_end + total * float(prune_fraction)
        validate_end = prune_end + total * float(validate_fraction)
        hard_end = start + total * (1.0 - float(buffer_fraction))
        return cls(
            start=start,
            search_end=search_end,
            prune_end=prune_end,
            validate_end=validate_end,
            hard_end=hard_end,
        )

    @property
    def search_round_end(self) -> float:
        """Stop launching new beam rounds after 75% of the search window."""
        search_span = self.search_end - self.start
        return self.start + search_span * float(ATTACK_SEARCH_ROUND_FRACTION)


@dataclass
class BeamNode:
    """One beam-search path with validator-aligned ranking metrics."""

    state: AttackState
    untargeted_gap: float
    changed_pixels: int
    recent_gain_per_pixel: float
    rmse: float
    norm: float
    flipped: bool
    pred_idx: int
    round_id: int = 0
    expansion_idx: int = 0

    @property
    def adv(self) -> torch.Tensor:
        return self.state.adv


@dataclass
class AttackState:
    clean: torch.Tensor  # [3, H, W] decoded challenge image in [0, 1]
    delta: torch.Tensor  # [3, H, W] sparse perturbation, typically {-1/255, 0, +1/255}
    changed_mask: torch.Tensor  # [3, H, W] bool — each channel-pixel changed at most once
    true_idx: int
    logits: torch.Tensor  # [num_classes] detached float32
    current_competitor_idx: int = -1
    top_k_competitors: list[CompetitorEntry] = field(default_factory=list)
    accepted_groups: list[AcceptedGroup] = field(default_factory=list)

    @property
    def adv(self) -> torch.Tensor:
        return (self.clean + self.delta).clamp(0.0, 1.0)

    @property
    def adv_raw(self) -> torch.Tensor:
        """Unclamped decoded adversarial tensor: clean + delta."""
        return self.clean + self.delta

    @property
    def linf(self) -> float:
        return float(self.delta.abs().max().item())


def _logits_row(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits[0]
    return logits


def inference_logits(model: torch.nn.Module, image_chw: torch.Tensor) -> torch.Tensor:
    """Canonical batched logits via perturbnet.model (includes PREPROCESS)."""
    return logits_for_images(model=model, image_bchw=image_chw.unsqueeze(0))


def inference_logits_features(
    model: torch.nn.Module,
    image_chw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and final feature map A for decoded CHW image (includes PREPROCESS)."""
    return forward_logits_features(model=model, image_bchw=image_chw.unsqueeze(0))


def inference_predict_index(model: torch.nn.Module, image_chw: torch.Tensor) -> int:
    """Canonical class index via perturbnet.model (includes PREPROCESS)."""
    return predict_index(model=model, image_chw=image_chw)


def inference_predict_label(model: torch.nn.Module, image_chw: torch.Tensor) -> str:
    """Canonical label string via perturbnet.model (includes PREPROCESS)."""
    return predict_label(model=model, image_chw=image_chw)


def is_prediction_flipped(model: torch.nn.Module, image_chw: torch.Tensor, true_idx: int) -> bool:
    return inference_predict_index(model=model, image_chw=image_chw) != true_idx


def _effective_max_delta(epsilon: float) -> float:
    return min(float(epsilon), float(MAX_LINF_DELTA))


def compute_top_k_competitors(
    logits: torch.Tensor,
    true_idx: int,
    k: int = DEFAULT_TOP_K,
) -> TopKCompetitorRace:
    """
    Rank top-K non-true classes by logit for dynamic untargeted competitor race.

    gap_k = logit_true - logit_competitor (positive while true class still leads).
    """
    row = _logits_row(logits).detach().float()
    true_logit = float(row[true_idx].item())

    masked = row.clone()
    masked[true_idx] = float("-inf")
    competitor_count = row.numel() - 1
    if competitor_count <= 0:
        return TopKCompetitorRace(true_idx=true_idx, true_logit=true_logit, competitors=())

    actual_k = min(max(int(k), 1), competitor_count)
    top_values, top_indices = torch.topk(masked, k=actual_k, largest=True)

    competitors = tuple(
        CompetitorEntry(
            idx=int(comp_idx),
            logit=float(comp_logit),
            gap_k=true_logit - float(comp_logit),
        )
        for comp_idx, comp_logit in zip(top_indices.tolist(), top_values.tolist())
    )
    return TopKCompetitorRace(true_idx=true_idx, true_logit=true_logit, competitors=competitors)


def refresh_competitor_race(state: AttackState, k: int = DEFAULT_TOP_K) -> TopKCompetitorRace:
    race = compute_top_k_competitors(logits=state.logits.unsqueeze(0), true_idx=state.true_idx, k=k)
    state.top_k_competitors = list(race.competitors)
    if race.competitors:
        state.current_competitor_idx = race.easiest.idx
    else:
        state.current_competitor_idx = -1
    return race


def best_competitor_idx(logits: torch.Tensor, true_idx: int) -> int:
    race = compute_top_k_competitors(logits=logits, true_idx=true_idx, k=1)
    if not race.competitors:
        return true_idx
    return race.competitors[0].idx


def competitor_gap(logits: torch.Tensor, true_idx: int, competitor_idx: int) -> float:
    row = _logits_row(logits)
    return float((row[true_idx] - row[competitor_idx]).item())


def untargeted_gap(logits: torch.Tensor, true_idx: int) -> float:
    race = compute_top_k_competitors(logits=logits, true_idx=true_idx, k=1)
    if not race.competitors:
        return 0.0
    return race.competitors[0].gap_k


def is_flip_success(logits: torch.Tensor, true_idx: int) -> bool:
    race = compute_top_k_competitors(logits=logits, true_idx=true_idx, k=1)
    return race.is_success(logits)


def init_attack_state(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
) -> AttackState:
    """
    Build attack state from decoded challenge image [3, H, W] in [0, 1].

    Do not resize to 480×480 — validator requires x_adv.shape == x_clean.shape.
    """
    if clean.ndim != 3:
        raise ValueError(f"Expected clean CHW tensor, got shape {tuple(clean.shape)}")

    delta = torch.zeros_like(clean)
    changed_mask = torch.zeros_like(clean, dtype=torch.bool)

    with torch.no_grad():
        logits = _logits_row(inference_logits(model=model, image_chw=clean))
        logits = logits.detach().to(dtype=torch.float32)

    pred_idx = inference_predict_index(model=model, image_chw=clean)
    if pred_idx != true_idx:
        logger.warning(
            "Provided true_idx=%s does not match clean model prediction=%s",
            true_idx,
            pred_idx,
        )

    state = AttackState(
        clean=clean.detach(),
        delta=delta,
        changed_mask=changed_mask,
        true_idx=true_idx,
        logits=logits,
    )
    refresh_competitor_race(state, k=DEFAULT_TOP_K)
    return state


def refresh_state_logits(model: torch.nn.Module, state: AttackState, *, top_k: int = DEFAULT_TOP_K) -> None:
    with torch.no_grad():
        state.logits = _logits_row(
            inference_logits(model=model, image_chw=state.adv)
        ).detach().to(dtype=torch.float32)
    refresh_competitor_race(state, k=top_k)


def pixel_direction_from_grad(grad_value: float) -> int:
    """
    Direction to reduce gap_k = logit_true - logit_competitor.

    Returns -1, 0, or +1. Zero means unusable (flat gradient).
    """
    if grad_value > 0.0:
        return -1
    if grad_value < 0.0:
        return 1
    return 0


def _raw_value_after_step(
    state: AttackState,
    channel: int,
    y: int,
    x: int,
    direction: int,
) -> float:
    return float(
        state.clean[channel, y, x].item()
        + state.delta[channel, y, x].item()
        + float(direction) * PIXEL_STEP_RAW
    )


def can_apply_pixel_change(state: AttackState, change: PixelChange) -> bool:
    """Reject already-changed, zero-direction, or clip-violating ±1/255 steps."""
    if change.direction not in (-1, 1):
        return False

    _, height, width = state.clean.shape
    if not (0 <= change.channel < 3):
        return False
    if not (0 <= change.y < height):
        return False
    if not (0 <= change.x < width):
        return False
    if state.changed_mask[change.channel, change.y, change.x]:
        return False

    next_raw = _raw_value_after_step(
        state=state,
        channel=change.channel,
        y=change.y,
        x=change.x,
        direction=change.direction,
    )
    if next_raw < -1e-9 or next_raw > 1.0 + 1e-9:
        return False
    return True


def apply_pixel_change(state: AttackState, change: PixelChange) -> bool:
    if not can_apply_pixel_change(state, change):
        return False

    step = float(change.direction) * PIXEL_STEP_RAW
    c, y, x = change.channel, change.y, change.x
    state.delta[c, y, x] = state.delta[c, y, x] + step
    state.changed_mask[c, y, x] = True
    return True


def propose_pixel_candidate(
    state: AttackState,
    pixel_grad_raw: torch.Tensor,
    channel: int,
    y: int,
    x: int,
) -> PixelCandidate | None:
    grad_value = float(pixel_grad_raw[channel, y, x].item())
    direction = pixel_direction_from_grad(grad_value)
    if direction == 0:
        return None

    change = PixelChange(channel=channel, y=y, x=x, direction=direction)
    if not can_apply_pixel_change(state, change):
        return None

    return PixelCandidate(
        channel=channel,
        y=y,
        x=x,
        direction=direction,
        grad_value=grad_value,
        predicted_gain=abs(grad_value) * PIXEL_STEP_RAW,
    )


def enumerate_region_pixel_candidates(
    state: AttackState,
    pixel_grad_raw: torch.Tensor,
    image_box: tuple[int, int, int, int],
) -> list[PixelCandidate]:
    """All valid ±1/255 pixel candidates inside one raw image box."""
    y1, y2, x1, x2 = image_box
    candidates: list[PixelCandidate] = []

    for channel in range(3):
        for y in range(y1, y2):
            for x in range(x1, x2):
                candidate = propose_pixel_candidate(
                    state=state,
                    pixel_grad_raw=pixel_grad_raw,
                    channel=channel,
                    y=y,
                    x=x,
                )
                if candidate is not None:
                    candidates.append(candidate)
    return candidates


def select_pixels_in_region(
    state: AttackState,
    pixel_grad_raw: torch.Tensor,
    image_box: tuple[int, int, int, int],
    *,
    top_n: int | None = DEFAULT_TOP_PIXELS_PER_REGION,
) -> list[PixelCandidate]:
    """Rank region candidates by |pixel_grad_raw| (strongest gap reduction first)."""
    candidates = enumerate_region_pixel_candidates(
        state=state,
        pixel_grad_raw=pixel_grad_raw,
        image_box=image_box,
    )
    candidates.sort(key=lambda candidate: abs(candidate.grad_value), reverse=True)
    if top_n is None or top_n <= 0:
        return candidates
    return candidates[:top_n]


def select_pixels_in_regions(
    state: AttackState,
    pixel_grad_raw: torch.Tensor,
    ranked_regions: list[RankedRegion],
    *,
    top_per_region: int = DEFAULT_TOP_PIXELS_PER_REGION,
) -> list[PixelCandidate]:
    """Collect top pixel candidates from each ranked region."""
    selected: list[PixelCandidate] = []
    for region in ranked_regions:
        selected.extend(
            select_pixels_in_region(
                state=state,
                pixel_grad_raw=pixel_grad_raw,
                image_box=region.image_box,
                top_n=top_per_region,
            )
        )
    selected.sort(key=lambda candidate: abs(candidate.grad_value), reverse=True)
    return selected


def compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    """Match validator SSIM gate on decoded [3, H, W] tensors."""
    if x_clean.ndim != 3 or x_adv.ndim != 3 or x_clean.shape != x_adv.shape:
        return 0.0

    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def build_delta_try_from_changes(
    state: AttackState,
    pixel_changes: list[PixelChange],
) -> torch.Tensor | None:
    """Build trial delta by applying a batch of ±1/255 changes on current state."""
    delta_try = state.delta.clone()
    changed_try = state.changed_mask.clone()

    for change in pixel_changes:
        if change.direction not in (-1, 1):
            return None
        if changed_try[change.channel, change.y, change.x]:
            return None

        next_raw = float(
            state.clean[change.channel, change.y, change.x].item()
            + delta_try[change.channel, change.y, change.x].item()
            + float(change.direction) * PIXEL_STEP_RAW
        )
        if next_raw < -1e-9 or next_raw > 1.0 + 1e-9:
            return None

        delta_try[change.channel, change.y, change.x] += float(change.direction) * PIXEL_STEP_RAW
        changed_try[change.channel, change.y, change.x] = True

    return delta_try


def verify_trial_delta(
    model: torch.nn.Module,
    state: AttackState,
    delta_try: torch.Tensor,
    true_idx: int,
    competitor_idx: int,
    *,
    gap_before: float,
    untargeted_gap_before: float,
    num_new_pixels: int,
    epsilon: float,
    min_delta: float,
    min_gain_per_pixel: float = DEFAULT_MIN_GAIN_PER_PIXEL,
) -> TrialVerification:
    """
    Forward-pass verification using validator-style metrics on decoded image space.

    adv_try = clamp(clean + delta_try, 0, 1)
    logits via logits_for_images (includes PREPROCESS).
    """
    adv_try = (state.clean + delta_try).clamp(0.0, 1.0)
    logits = inference_logits(model=model, image_chw=adv_try)
    row = _logits_row(logits)
    pred_idx = int(row.argmax().item())

    norm = float((adv_try - state.clean).abs().max().item())
    rmse = float(torch.sqrt(torch.mean((adv_try - state.clean) ** 2)).item())
    ssim = compute_ssim(state.clean, adv_try)
    psnr_db = compute_psnr_db(state.clean, adv_try)

    gap_after = competitor_gap(logits, true_idx, competitor_idx)
    untargeted_gap_after = untargeted_gap(logits, true_idx)
    real_gain = float(gap_before - gap_after)
    gain_per_pixel = real_gain / max(int(num_new_pixels), 1)

    max_norm = _effective_max_delta(epsilon)
    value_ok = float(adv_try.min().item()) >= -1e-9 and float(adv_try.max().item()) <= 1.0 + 1e-9
    norm_ok = norm <= max_norm + 1e-9
    untargeted_improved = untargeted_gap_after < untargeted_gap_before - 1e-9

    progress_accept = (
        real_gain > 0.0
        and gain_per_pixel >= float(min_gain_per_pixel)
        and untargeted_improved
        and norm_ok
        and value_ok
    )

    flipped = pred_idx != true_idx
    flip_candidate = (
        flipped
        and norm >= float(min_delta) - 1e-9
        and norm_ok
        and value_ok
        and ssim >= float(MIN_SSIM)
        and psnr_db >= float(MIN_PSNR_DB)
    )
    validator_would_pass = flip_candidate
    accepted = progress_accept or flip_candidate

    if flip_candidate:
        reason = "flip_candidate"
    elif progress_accept:
        reason = "progress_accept"
    elif not norm_ok:
        reason = "above_max_delta"
    elif not value_ok:
        reason = "value_out_of_range"
    elif real_gain <= 0.0:
        reason = "no_real_gain"
    elif not untargeted_improved:
        reason = "untargeted_gap_not_improved"
    elif gain_per_pixel < float(min_gain_per_pixel):
        reason = "weak_gain_per_pixel"
    else:
        reason = "rejected"

    return TrialVerification(
        accepted=accepted,
        flip_candidate=flip_candidate,
        validator_would_pass=validator_would_pass,
        reason=reason,
        adv_try=adv_try.detach(),
        logits=row.detach().to(dtype=torch.float32),
        pred_idx=pred_idx,
        norm=norm,
        rmse=rmse,
        ssim=ssim,
        psnr_db=psnr_db,
        gap_before=gap_before,
        gap_after=gap_after,
        untargeted_gap_before=untargeted_gap_before,
        untargeted_gap_after=untargeted_gap_after,
        real_gain=real_gain,
        gain_per_pixel=gain_per_pixel,
        num_new_pixels=int(num_new_pixels),
    )


def commit_verified_delta(
    state: AttackState,
    delta_try: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    state.delta = delta_try.detach()
    state.changed_mask = state.delta.abs() > 1e-12
    state.logits = logits.detach().to(dtype=torch.float32)


def try_accept_pixel_batch(
    model: torch.nn.Module,
    state: AttackState,
    pixel_changes: list[PixelChange],
    *,
    true_idx: int,
    competitor_idx: int,
    gap_before: float,
    untargeted_gap_before: float,
    epsilon: float,
    min_delta: float,
    region: RankedRegion | None = None,
    round_id: int = 0,
    min_gain_per_pixel: float = DEFAULT_MIN_GAIN_PER_PIXEL,
) -> tuple[bool, TrialVerification | None, AcceptedGroup | None]:
    delta_try = build_delta_try_from_changes(state=state, pixel_changes=pixel_changes)
    if delta_try is None:
        return False, None, None

    verification = verify_trial_delta(
        model=model,
        state=state,
        delta_try=delta_try,
        true_idx=true_idx,
        competitor_idx=competitor_idx,
        gap_before=gap_before,
        untargeted_gap_before=untargeted_gap_before,
        num_new_pixels=len(pixel_changes),
        epsilon=epsilon,
        min_delta=min_delta,
        min_gain_per_pixel=min_gain_per_pixel,
    )
    if not verification.accepted:
        return False, verification, None

    commit_verified_delta(state=state, delta_try=delta_try, logits=verification.logits)

    accepted_group = None
    if region is not None:
        accepted_group = AcceptedGroup(
            competitor_idx=competitor_idx,
            feature_cell=region.feature_cell,
            image_box=region.image_box,
            pixels=list(pixel_changes),
            gap_before=verification.gap_before,
            gap_after=verification.gap_after,
            gain=verification.real_gain,
            gain_per_pixel=verification.gain_per_pixel,
            round_id=round_id,
        )
        state.accepted_groups.append(accepted_group)

    return True, verification, accepted_group


def apply_verified_pixel_candidates(
    model: torch.nn.Module,
    state: AttackState,
    candidates: list[PixelCandidate],
    ranked_regions: list[RankedRegion],
    *,
    true_idx: int,
    competitor_idx: int,
    epsilon: float,
    min_delta: float,
    max_delta: float,
    max_pixels: int = DEFAULT_MAX_PIXELS_PER_STEP,
    round_id: int = 0,
) -> tuple[int, TrialVerification | None]:
    """
    Apply pixel candidates one-by-one with forward-pass verification.

    Returns count applied and the last verification (or best flip seen).
    """
    gap_before = competitor_gap(state.logits.unsqueeze(0), true_idx, competitor_idx)
    untargeted_gap_before = untargeted_gap(state.logits.unsqueeze(0), true_idx)
    applied = 0
    last_verification: TrialVerification | None = None
    best_flip: TrialVerification | None = None

    region_for_box: dict[tuple[int, int, int, int], RankedRegion] = {
        region.image_box: region for region in ranked_regions
    }

    for candidate in candidates:
        if applied >= max_pixels:
            break
        if state.linf + PIXEL_STEP_RAW > float(max_delta) + 1e-9:
            break

        change = candidate.as_change()
        region = None
        for image_box, ranked in region_for_box.items():
            y1, y2, x1, x2 = image_box
            if y1 <= change.y < y2 and x1 <= change.x < x2:
                region = ranked
                break

        accepted, verification, _ = try_accept_pixel_batch(
            model=model,
            state=state,
            pixel_changes=[change],
            true_idx=true_idx,
            competitor_idx=competitor_idx,
            gap_before=gap_before,
            untargeted_gap_before=untargeted_gap_before,
            epsilon=epsilon,
            min_delta=min_delta,
            region=region,
            round_id=round_id,
        )
        if verification is not None:
            last_verification = verification
            if verification.flip_candidate and (
                best_flip is None or verification.norm < best_flip.norm
            ):
                best_flip = verification

        if not accepted:
            continue

        applied += 1
        gap_before = verification.gap_after if verification else gap_before
        untargeted_gap_before = (
            verification.untargeted_gap_after if verification else untargeted_gap_before
        )

    return applied, best_flip or last_verification


def _clamp_grow_batch_size(batch_size: int) -> int:
    return max(REGION_GROW_MIN_BATCH, min(REGION_GROW_MAX_BATCH, int(batch_size)))


def _increase_grow_batch_size(batch_size: int) -> int:
    return _clamp_grow_batch_size(max(batch_size + 4, batch_size * 2))


def _decrease_grow_batch_size(batch_size: int) -> int:
    return _clamp_grow_batch_size(max(REGION_GROW_MIN_BATCH, batch_size // 2))


def is_gap_close_to_flip(gap_after: float, gap_before: float) -> bool:
    """True when competitor gap is near crossing zero (flip imminent)."""
    if gap_after <= float(REGION_GROW_CLOSE_TO_FLIP_GAP):
        return True
    if gap_before > 1e-9:
        return gap_after <= float(gap_before) * float(REGION_GROW_CLOSE_TO_FLIP_RATIO)
    return gap_after <= float(REGION_GROW_CLOSE_TO_FLIP_GAP)


def classify_region_gain_strength(real_gain: float, gain_per_pixel: float) -> str:
    if real_gain <= 0.0:
        return "zero"
    if gain_per_pixel >= float(REGION_GROW_STRONG_GAIN_PER_PIXEL):
        return "strong"
    return "weak"


def should_accept_region_growth_batch(
    verification: TrialVerification,
    *,
    gap_before: float,
    seed_phase: bool,
    epsilon: float,
) -> tuple[bool, str]:
    """
    Region-growing acceptance (step 13), layered on forward verification metrics.

    Seed phase uses step-12 progress/flip gates. Growth phase uses marginal-gain rules.
    """
    max_norm = _effective_max_delta(epsilon)
    value_ok = float(verification.adv_try.min().item()) >= -1e-9 and float(
        verification.adv_try.max().item()
    ) <= 1.0 + 1e-9
    norm_ok = verification.norm <= max_norm + 1e-9

    if verification.flip_candidate:
        return True, "flip_candidate"

    if seed_phase:
        if verification.accepted:
            return True, verification.reason
        return False, verification.reason

    if not value_ok:
        return False, "value_out_of_range"
    if not norm_ok:
        return False, "above_max_delta"

    strength = classify_region_gain_strength(
        verification.real_gain,
        verification.gain_per_pixel,
    )
    if strength == "zero":
        return False, "no_real_gain"
    if strength == "strong":
        return True, "strong_gain"
    if is_gap_close_to_flip(verification.gap_after, gap_before):
        return True, "weak_gain_close_to_flip"
    return False, "weak_gain_not_close_to_flip"


def select_unused_pixels_in_region(
    state: AttackState,
    pixel_grad_raw: torch.Tensor,
    image_box: tuple[int, int, int, int],
    *,
    top_n: int | None,
) -> list[PixelCandidate]:
    """Unused ±1/255 candidates inside one region, ranked by |pixel_grad_raw|."""
    return select_pixels_in_region(
        state=state,
        pixel_grad_raw=pixel_grad_raw,
        image_box=image_box,
        top_n=top_n,
    )


def _region_growth_budget_ok(state: AttackState, batch_len: int, max_delta: float) -> bool:
    """Batch adds ±1/255 pixels; Linf is max abs delta, not a sum over batch size."""
    if batch_len <= 0:
        return False
    projected_linf = max(state.linf, PIXEL_STEP_RAW)
    return projected_linf <= float(max_delta) + 1e-9


def try_region_growth_batch(
    model: torch.nn.Module,
    state: AttackState,
    pixel_changes: list[PixelChange],
    *,
    region: RankedRegion,
    true_idx: int,
    competitor_idx: int,
    gap_before: float,
    untargeted_gap_before: float,
    epsilon: float,
    min_delta: float,
    seed_phase: bool,
    round_id: int,
) -> tuple[bool, TrialVerification | None, AcceptedGroup | None, str]:
    delta_try = build_delta_try_from_changes(state=state, pixel_changes=pixel_changes)
    if delta_try is None:
        return False, None, None, "invalid_batch"

    verification = verify_trial_delta(
        model=model,
        state=state,
        delta_try=delta_try,
        true_idx=true_idx,
        competitor_idx=competitor_idx,
        gap_before=gap_before,
        untargeted_gap_before=untargeted_gap_before,
        num_new_pixels=len(pixel_changes),
        epsilon=epsilon,
        min_delta=min_delta,
    )
    accept, reason = should_accept_region_growth_batch(
        verification,
        gap_before=gap_before,
        seed_phase=seed_phase,
        epsilon=epsilon,
    )
    if not accept:
        return False, verification, None, reason

    commit_verified_delta(state=state, delta_try=delta_try, logits=verification.logits)
    accepted_group = AcceptedGroup(
        competitor_idx=competitor_idx,
        feature_cell=region.feature_cell,
        image_box=region.image_box,
        pixels=list(pixel_changes),
        gap_before=verification.gap_before,
        gap_after=verification.gap_after,
        gain=verification.real_gain,
        gain_per_pixel=verification.gain_per_pixel,
        round_id=round_id,
    )
    state.accepted_groups.append(accepted_group)
    return True, verification, accepted_group, reason


def grow_ranked_region(
    model: torch.nn.Module,
    state: AttackState,
    region: RankedRegion,
    pixel_grad_raw: torch.Tensor,
    *,
    true_idx: int,
    competitor_idx: int,
    epsilon: float,
    min_delta: float,
    max_delta: float,
    round_id: int = 0,
    deadline: float | None = None,
    max_pixels: int | None = None,
) -> RegionGrowResult:
    """
    Seed then adaptively grow one ranked region in verified pixel batches.

    Stops on saturation (gain <= 0), two consecutive failures, pixel cap, or budget.
    """
    pixel_cap = int(max_pixels) if max_pixels is not None else REGION_GROW_MAX_PIXELS_PER_REGION
    session = RegionGrowSession(region=region, batch_size=REGION_GROW_INITIAL_BATCH)
    gap_before = competitor_gap(state.logits.unsqueeze(0), true_idx, competitor_idx)
    untargeted_gap_before = untargeted_gap(state.logits.unsqueeze(0), true_idx)
    best_verification: TrialVerification | None = None
    stopped_reason = "not_started"

    while not session.stopped and session.pixels_applied < pixel_cap:
        if deadline is not None and time.monotonic() >= deadline:
            session.stopped = True
            stopped_reason = "timeout"
            break

        batch_size = _clamp_grow_batch_size(session.batch_size)
        candidates = select_unused_pixels_in_region(
            state=state,
            pixel_grad_raw=pixel_grad_raw,
            image_box=region.image_box,
            top_n=batch_size,
        )
        if not candidates:
            session.stopped = True
            stopped_reason = "no_candidates"
            break

        pixel_changes = [candidate.as_change() for candidate in candidates[:batch_size]]
        if not _region_growth_budget_ok(state, len(pixel_changes), max_delta):
            session.stopped = True
            stopped_reason = "linf_budget"
            break

        seed_phase = not session.seed_accepted
        accepted, verification, _, reason = try_region_growth_batch(
            model=model,
            state=state,
            pixel_changes=pixel_changes,
            region=region,
            true_idx=true_idx,
            competitor_idx=competitor_idx,
            gap_before=gap_before,
            untargeted_gap_before=untargeted_gap_before,
            epsilon=epsilon,
            min_delta=min_delta,
            seed_phase=seed_phase,
            round_id=round_id,
        )

        if verification is not None:
            if verification.flip_candidate and (
                best_verification is None or verification.norm < best_verification.norm
            ):
                best_verification = verification

        if not accepted:
            session.failure_count += 1
            session.batch_size = _decrease_grow_batch_size(session.batch_size)
            if not session.seed_accepted:
                session.stopped = True
                stopped_reason = f"seed_rejected:{reason}"
                break
            if session.failure_count >= REGION_GROW_MAX_FAILURES:
                session.stopped = True
                stopped_reason = f"failures:{reason}"
            continue

        session.pixels_applied += len(pixel_changes)
        session.failure_count = 0
        gap_before = verification.gap_after if verification else gap_before
        untargeted_gap_before = (
            verification.untargeted_gap_after if verification else untargeted_gap_before
        )

        if not session.seed_accepted:
            session.seed_accepted = True
            stopped_reason = "seed_accepted"

        strength = classify_region_gain_strength(
            verification.real_gain if verification else 0.0,
            verification.gain_per_pixel if verification else 0.0,
        )
        if strength == "strong":
            session.batch_size = _increase_grow_batch_size(session.batch_size)
        elif strength == "weak":
            session.batch_size = _decrease_grow_batch_size(session.batch_size)

        if verification is not None and verification.flip_candidate:
            session.stopped = True
            stopped_reason = "flip_found"
            break

    if session.stopped and stopped_reason == "not_started":
        stopped_reason = "max_pixels" if session.pixels_applied >= pixel_cap else "done"

    return RegionGrowResult(
        region=region,
        pixels_applied=session.pixels_applied,
        seed_accepted=session.seed_accepted,
        stopped_reason=stopped_reason,
        best_verification=best_verification,
    )


def grow_ranked_regions(
    model: torch.nn.Module,
    state: AttackState,
    ranked_regions: list[RankedRegion],
    pixel_grad_raw: torch.Tensor,
    *,
    true_idx: int,
    competitor_idx: int,
    epsilon: float,
    min_delta: float,
    max_delta: float,
    round_id: int = 0,
    deadline: float | None = None,
    max_regions: int = DEFAULT_TOP_REGIONS_PER_COMPETITOR,
    max_pixels_per_region: int | None = None,
) -> tuple[int, TrialVerification | None]:
    """
    Run adaptive region growing across ranked regions for one attack step.

    Each region seeds with an initial batch, then grows while marginal gain remains.
    """
    total_applied = 0
    best_flip: TrialVerification | None = None

    for region in ranked_regions[:max_regions]:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if state.linf >= float(max_delta) - 1e-9:
            break

        result = grow_ranked_region(
            model=model,
            state=state,
            region=region,
            pixel_grad_raw=pixel_grad_raw,
            true_idx=true_idx,
            competitor_idx=competitor_idx,
            epsilon=epsilon,
            min_delta=min_delta,
            max_delta=max_delta,
            round_id=round_id,
            deadline=deadline,
            max_pixels=max_pixels_per_region,
        )
        total_applied += result.pixels_applied
        if result.best_verification is not None and result.best_verification.flip_candidate:
            if best_flip is None or result.best_verification.norm < best_flip.norm:
                best_flip = result.best_verification
        if result.flip_found:
            break

    return total_applied, best_flip


def apply_pixel_candidates(
    state: AttackState,
    candidates: list[PixelCandidate],
    *,
    max_delta: float,
    max_pixels: int = DEFAULT_MAX_PIXELS_PER_STEP,
) -> int:
    """
    Apply discrete ±1/255 updates without forward verification.

    Prefer apply_verified_pixel_candidates during attack search.
    """
    applied = 0
    for candidate in candidates:
        if applied >= max_pixels:
            break
        if state.linf + PIXEL_STEP_RAW > float(max_delta) + 1e-9:
            break
        if apply_pixel_change(state, candidate.as_change()):
            applied += 1
    return applied


def compute_competitor_gap_gradients(
    model: torch.nn.Module,
    adv_raw: torch.Tensor,
    true_idx: int,
    competitor_idx: int,
) -> CompetitorGapGradients:
    """
    Backprop gap_k through forward_logits_features on decoded raw [3, H, W].

    adv_raw = clean + delta; gradient flows through Torchvision PREPROCESS so
    pixel_grad_raw aligns with validator-scored pixels.
    """
    if adv_raw.ndim != 3:
        raise ValueError(f"Expected adv_raw CHW tensor, got shape {tuple(adv_raw.shape)}")

    adv_leaf = adv_raw.detach().requires_grad_(True)
    logits, features = forward_logits_features(model=model, image_bchw=adv_leaf.unsqueeze(0))
    features.retain_grad()
    gap_k = logits[0, true_idx] - logits[0, competitor_idx]
    model.zero_grad(set_to_none=True)
    gap_k.backward(retain_graph=False)

    pixel_grad_raw = adv_leaf.grad
    if pixel_grad_raw is None:
        pixel_grad_raw = torch.zeros_like(adv_leaf)

    return CompetitorGapGradients(
        competitor_idx=competitor_idx,
        gap_k=float(gap_k.item()),
        logits=logits.detach(),
        features=features.detach(),
        pixel_grad_raw=pixel_grad_raw.detach(),
        feature_grad=None if features.grad is None else features.grad.detach(),
    )


def classifier_weight_matrix(model: torch.nn.Module) -> torch.Tensor:
    """EfficientNetV2-L final linear weights [num_classes, 1280]."""
    return model.classifier[-1].weight.detach()


def compute_gap_weight_vector(
    model: torch.nn.Module,
    true_idx: int,
    competitor_idx: int,
) -> torch.Tensor:
    """W_gap = W_true - W_competitor with shape [1280]."""
    weights = classifier_weight_matrix(model=model)
    return weights[true_idx] - weights[competitor_idx]


def compute_gap_cam(
    model: torch.nn.Module,
    features: torch.Tensor,
    true_idx: int,
    competitor_idx: int,
) -> CompetitorGapCam:
    """
    Gap CAM for one competitor on feature map A.

    gap_cam[yf, xf] = sum_c W_gap[c] * A[c, yf, xf]

    Positive values: feature cell supports true over competitor.
    Negative values: feature cell supports competitor over true.
    """
    if features.ndim == 4:
        feature_map = features[0]
    elif features.ndim == 3:
        feature_map = features
    else:
        raise ValueError(f"Expected feature map [1,C,H,W] or [C,H,W], got {tuple(features.shape)}")

    w_gap = compute_gap_weight_vector(
        model=model,
        true_idx=true_idx,
        competitor_idx=competitor_idx,
    ).to(device=feature_map.device, dtype=feature_map.dtype)

    gap_cam = torch.einsum("c,cyx->yx", w_gap, feature_map)
    return CompetitorGapCam(
        competitor_idx=competitor_idx,
        w_gap=w_gap.detach(),
        gap_cam=gap_cam.detach(),
    )


def compute_gap_cams_for_competitors(
    model: torch.nn.Module,
    features: torch.Tensor,
    true_idx: int,
    competitor_indices: list[int] | tuple[int, ...],
) -> list[CompetitorGapCam]:
    """Gap CAM for each competitor in the current top-K race."""
    return [
        compute_gap_cam(
            model=model,
            features=features,
            true_idx=true_idx,
            competitor_idx=int(comp_idx),
        )
        for comp_idx in competitor_indices
    ]


def gap_cam_to_pixel_weights(
    gap_cam: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    include_competitor_support: bool = True,
    competitor_support_scale: float = 0.5,
) -> torch.Tensor:
    """
    Upsample gap CAM to decoded image size for pixel-gradient weighting.

    Mainly weight positive (true-supporting) cells; optionally include negative
    competitor-supporting cells at reduced scale.
    """
    true_support = F.relu(gap_cam)
    if include_competitor_support:
        competitor_support = F.relu(-gap_cam) * float(competitor_support_scale)
        weights = true_support + competitor_support
    else:
        weights = true_support

    weights = F.interpolate(
        weights.unsqueeze(0).unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    peak = float(weights.max().item())
    if peak > 1e-12:
        weights = weights / peak
    else:
        weights = torch.ones_like(weights)
    return weights.detach()


def _as_feature_map(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features[0]
    if features.ndim == 3:
        return features
    raise ValueError(f"Expected feature map [1,C,H,W] or [C,H,W], got {tuple(features.shape)}")


def _upsample_feature_map(feature_map: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        feature_map.unsqueeze(0).unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def _normalize_positive_map(values: torch.Tensor) -> torch.Tensor:
    peak = float(values.max().item())
    if peak > 1e-12:
        return values / peak
    return torch.ones_like(values)


def compute_activation_gradient_maps(
    features: torch.Tensor,
    feature_grad: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Activation-gradient maps for the current competitor gap.

    abs_map[yf, xf] = sum_c abs(A.grad[c, yf, xf] * A[c, yf, xf])
    dir_map[yf, xf] = sum_c A.grad[c, yf, xf] * A[c, yf, xf]
    """
    feature_map = _as_feature_map(features)
    height, width = feature_map.shape[-2:]
    if feature_grad is None:
        zeros = torch.zeros((height, width), device=feature_map.device, dtype=feature_map.dtype)
        return zeros, zeros

    grad_map = _as_feature_map(feature_grad)
    product = grad_map * feature_map
    dir_map = product.sum(dim=0)
    abs_map = product.abs().sum(dim=0)
    return abs_map.detach(), dir_map.detach()


def compute_competitor_activation_maps(
    grad_pack: CompetitorGapGradients,
) -> CompetitorActivationMaps:
    abs_map, dir_map = compute_activation_gradient_maps(
        features=grad_pack.features,
        feature_grad=grad_pack.feature_grad,
    )
    return CompetitorActivationMaps(
        competitor_idx=grad_pack.competitor_idx,
        abs_map=abs_map,
        dir_map=dir_map,
    )


def activation_guided_pixel_weights(
    abs_map: torch.Tensor,
    dir_map: torch.Tensor,
    gap_cam: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    include_competitor_support: bool = True,
    competitor_support_scale: float = 0.5,
) -> torch.Tensor:
    """
    Combine activation-gradient importance with gap CAM direction.

    abs_map drives region ranking (which 15x15 cells control the gap).
    dir_map + gap_cam provide directional interpretation.
    """
    abs_up = _upsample_feature_map(abs_map, target_hw)
    dir_up = _upsample_feature_map(dir_map, target_hw)
    gap_up = _upsample_feature_map(gap_cam, target_hw)

    importance = _normalize_positive_map(abs_up)

    true_support = F.relu(gap_up)
    if include_competitor_support:
        competitor_support = F.relu(-gap_up) * float(competitor_support_scale)
        structural = true_support + competitor_support
    else:
        structural = true_support

    # Positive dir_map: increasing activation raises gap -> prioritize reducing there.
    # Negative dir_map: increasing activation lowers gap -> secondary competitor support.
    directional = F.relu(dir_up) + float(competitor_support_scale) * F.relu(-dir_up)

    weights = importance * (structural + 1e-6) * (directional + 1e-6)
    peak = float(weights.max().item())
    if peak > 1e-12:
        weights = weights / peak
    else:
        weights = importance
    return weights.detach()


def compute_preprocess_geometry(raw_h: int, raw_w: int) -> PreprocessGeometry:
    """
    Approximate resize/center-crop geometry for decoded [H, W] before PREPROCESS.

    Matches torchvision ImageClassification: shorter side → 480, then center crop 480.
    """
    if raw_h <= 0 or raw_w <= 0:
        raise ValueError(f"Invalid decoded image size: ({raw_h}, {raw_w})")

    if raw_h <= raw_w:
        resized_h = MODEL_CROP_SIZE
        resized_w = int(round(raw_w * MODEL_CROP_SIZE / raw_h))
        crop_top = 0.0
        crop_left = (resized_w - MODEL_CROP_SIZE) / 2.0
    else:
        resized_w = MODEL_CROP_SIZE
        resized_h = int(round(raw_h * MODEL_CROP_SIZE / raw_w))
        crop_left = 0.0
        crop_top = (resized_h - MODEL_CROP_SIZE) / 2.0

    return PreprocessGeometry(
        raw_h=raw_h,
        raw_w=raw_w,
        resized_h=resized_h,
        resized_w=resized_w,
        crop_top=crop_top,
        crop_left=crop_left,
    )


def _clip_raw_box(
    y1: float,
    y2: float,
    x1: float,
    x2: float,
    raw_h: int,
    raw_w: int,
) -> tuple[int, int, int, int]:
    iy1 = max(0, int(math.floor(y1)))
    iy2 = min(raw_h, int(math.ceil(y2)))
    ix1 = max(0, int(math.floor(x1)))
    ix2 = min(raw_w, int(math.ceil(x2)))

    if iy2 <= iy1:
        iy2 = min(raw_h, iy1 + 1)
    if ix2 <= ix1:
        ix2 = min(raw_w, ix1 + 1)
    return iy1, iy2, ix1, ix2


def feature_cell_to_raw_image_box(
    yf: int,
    xf: int,
    raw_h: int,
    raw_w: int,
    *,
    expand: float = FEATURE_CELL_EXPAND_FIRST,
    geometry: PreprocessGeometry | None = None,
) -> tuple[int, int, int, int]:
    """
    Map 15×15 feature cell to decoded raw image box [y1, y2) × [x1, x2).

    Feature cells live after PREPROCESS; pixel perturbations live on decoded [H, W].
    """
    if not (0 <= yf < FEATURE_MAP_SIZE and 0 <= xf < FEATURE_MAP_SIZE):
        raise ValueError(f"Feature cell ({yf}, {xf}) outside [{FEATURE_MAP_SIZE}, {FEATURE_MAP_SIZE})")

    geom = geometry or compute_preprocess_geometry(raw_h=raw_h, raw_w=raw_w)
    cell_span = MODEL_CROP_SIZE / FEATURE_MAP_SIZE
    box_size = float(expand) * cell_span

    crop_center_y = (yf + 0.5) * cell_span
    crop_center_x = (xf + 0.5) * cell_span
    crop_y1 = crop_center_y - box_size / 2.0
    crop_y2 = crop_center_y + box_size / 2.0
    crop_x1 = crop_center_x - box_size / 2.0
    crop_x2 = crop_center_x + box_size / 2.0

    resized_y1 = crop_y1 + geom.crop_top
    resized_y2 = crop_y2 + geom.crop_top
    resized_x1 = crop_x1 + geom.crop_left
    resized_x2 = crop_x2 + geom.crop_left

    raw_y1 = resized_y1 * raw_h / geom.resized_h
    raw_y2 = resized_y2 * raw_h / geom.resized_h
    raw_x1 = resized_x1 * raw_w / geom.resized_w
    raw_x2 = resized_x2 * raw_w / geom.resized_w

    return _clip_raw_box(raw_y1, raw_y2, raw_x1, raw_x2, raw_h=raw_h, raw_w=raw_w)


def feature_cells_to_raw_image_boxes(
    feature_cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    raw_h: int,
    raw_w: int,
    *,
    expand: float = FEATURE_CELL_EXPAND_FIRST,
) -> list[tuple[int, int, int, int]]:
    geometry = compute_preprocess_geometry(raw_h=raw_h, raw_w=raw_w)
    return [
        feature_cell_to_raw_image_box(
            yf=yf,
            xf=xf,
            raw_h=raw_h,
            raw_w=raw_w,
            expand=expand,
            geometry=geometry,
        )
        for yf, xf in feature_cells
    ]


def ranked_feature_cell_boxes(
    abs_map: torch.Tensor,
    raw_h: int,
    raw_w: int,
    *,
    top_n: int,
    expand: float = FEATURE_CELL_EXPAND_FIRST,
) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    """Rank feature cells by abs_map and map each to a decoded-image box."""
    cells = rank_feature_cells(abs_map=abs_map, top_n=top_n)
    boxes = feature_cells_to_raw_image_boxes(
        feature_cells=cells,
        raw_h=raw_h,
        raw_w=raw_w,
        expand=expand,
    )
    return list(zip(cells, boxes))


def _normalize_cell_grid(values: torch.Tensor) -> torch.Tensor:
    """Normalize absolute feature-grid values to [0, 1] by global max."""
    peak = float(values.abs().max().item())
    if peak <= 1e-12:
        return torch.zeros_like(values)
    return values.abs() / peak


def pixel_gradient_density_in_box(
    pixel_grad_raw: torch.Tensor,
    image_box: tuple[int, int, int, int],
    *,
    top_n: int = REGION_PIXEL_GRAD_TOP_N,
) -> float:
    """
    Average of top-N |pixel_grad_raw| values inside decoded-image box.

    Measures whether mapped raw pixels have usable gradient signal.
    """
    y1, y2, x1, x2 = image_box
    patch = pixel_grad_raw[:, y1:y2, x1:x2].abs().reshape(-1)
    if patch.numel() == 0:
        return 0.0

    count = min(max(int(top_n), 1), int(patch.numel()))
    top_values = torch.topk(patch, k=count, largest=True).values
    return float(top_values.mean().item())


def score_feature_cell_region(
    gap_cam: torch.Tensor,
    abs_map: torch.Tensor,
    pixel_grad_raw: torch.Tensor,
    yf: int,
    xf: int,
    raw_h: int,
    raw_w: int,
    *,
    gap_cam_norm: torch.Tensor | None = None,
    abs_map_norm: torch.Tensor | None = None,
    density_scale: float = 1.0,
    expand: float = FEATURE_CELL_EXPAND_FIRST,
    geometry: PreprocessGeometry | None = None,
    pixel_grad_top_n: int = REGION_PIXEL_GRAD_TOP_N,
) -> RankedRegion:
    gap_grid = gap_cam_norm if gap_cam_norm is not None else _normalize_cell_grid(gap_cam)
    abs_grid = abs_map_norm if abs_map_norm is not None else _normalize_cell_grid(abs_map)

    geom = geometry or compute_preprocess_geometry(raw_h=raw_h, raw_w=raw_w)
    image_box = feature_cell_to_raw_image_box(
        yf=yf,
        xf=xf,
        raw_h=raw_h,
        raw_w=raw_w,
        expand=expand,
        geometry=geom,
    )
    density = pixel_gradient_density_in_box(
        pixel_grad_raw=pixel_grad_raw,
        image_box=image_box,
        top_n=pixel_grad_top_n,
    )
    density_term = density / density_scale if density_scale > 1e-12 else 0.0

    gap_term = float(gap_grid[yf, xf].item())
    abs_term = float(abs_grid[yf, xf].item())
    region_score = gap_term + abs_term + density_term

    return RankedRegion(
        competitor_idx=-1,
        feature_cell=(yf, xf),
        image_box=image_box,
        region_score=region_score,
        gap_cam_term=gap_term,
        abs_map_term=abs_term,
        pixel_grad_density_term=density_term,
        pixel_grad_density=density,
    )


def rank_competitor_regions(
    gap_cam: torch.Tensor,
    abs_map: torch.Tensor,
    pixel_grad_raw: torch.Tensor,
    competitor_idx: int,
    raw_h: int,
    raw_w: int,
    *,
    top_regions: int = DEFAULT_TOP_REGIONS_PER_COMPETITOR,
    expand: float = FEATURE_CELL_EXPAND_FIRST,
    pixel_grad_top_n: int = REGION_PIXEL_GRAD_TOP_N,
) -> list[RankedRegion]:
    """
    Rank feature cells for one competitor by combined region score.

    region_score =
        normalized(|gap_cam|) + normalized(abs_map) + normalized(pixel grad density)
    """
    gap_norm = _normalize_cell_grid(gap_cam)
    abs_norm = _normalize_cell_grid(abs_map)
    geometry = compute_preprocess_geometry(raw_h=raw_h, raw_w=raw_w)

    densities: list[float] = []
    candidates: list[RankedRegion] = []
    for yf in range(FEATURE_MAP_SIZE):
        for xf in range(FEATURE_MAP_SIZE):
            scored = score_feature_cell_region(
                gap_cam=gap_cam,
                abs_map=abs_map,
                pixel_grad_raw=pixel_grad_raw,
                yf=yf,
                xf=xf,
                raw_h=raw_h,
                raw_w=raw_w,
                gap_cam_norm=gap_norm,
                abs_map_norm=abs_norm,
                density_scale=1.0,
                expand=expand,
                geometry=geometry,
                pixel_grad_top_n=pixel_grad_top_n,
            )
            densities.append(scored.pixel_grad_density)
            candidates.append(
                RankedRegion(
                    competitor_idx=competitor_idx,
                    feature_cell=scored.feature_cell,
                    image_box=scored.image_box,
                    region_score=scored.region_score,
                    gap_cam_term=scored.gap_cam_term,
                    abs_map_term=scored.abs_map_term,
                    pixel_grad_density_term=0.0,
                    pixel_grad_density=scored.pixel_grad_density,
                )
            )

    density_scale = max(densities) if densities else 0.0
    rescored: list[RankedRegion] = []
    for candidate in candidates:
        density_term = (
            candidate.pixel_grad_density / density_scale if density_scale > 1e-12 else 0.0
        )
        rescored.append(
            RankedRegion(
                competitor_idx=competitor_idx,
                feature_cell=candidate.feature_cell,
                image_box=candidate.image_box,
                region_score=candidate.gap_cam_term + candidate.abs_map_term + density_term,
                gap_cam_term=candidate.gap_cam_term,
                abs_map_term=candidate.abs_map_term,
                pixel_grad_density_term=density_term,
                pixel_grad_density=candidate.pixel_grad_density,
            )
        )

    rescored.sort(key=lambda region: region.region_score, reverse=True)
    if top_regions <= 0:
        return rescored
    return rescored[: min(top_regions, len(rescored))]


def rank_all_competitor_regions(
    model: torch.nn.Module,
    state: AttackState,
    race: TopKCompetitorRace,
    *,
    top_regions_per_competitor: int = DEFAULT_TOP_REGIONS_PER_COMPETITOR,
    expand: float = FEATURE_CELL_EXPAND_FIRST,
    pixel_grad_top_n: int = REGION_PIXEL_GRAD_TOP_N,
) -> list[RankedRegion]:
    """Rank top regions for each competitor in the current top-K race."""
    raw_h, raw_w = state.clean.shape[-2:]
    ranked_all: list[RankedRegion] = []

    for entry in race.competitors:
        grad_pack = compute_competitor_gap_gradients(
            model=model,
            adv_raw=state.adv_raw,
            true_idx=state.true_idx,
            competitor_idx=entry.idx,
        )
        gap_cam_pack = compute_gap_cam(
            model=model,
            features=grad_pack.features,
            true_idx=state.true_idx,
            competitor_idx=entry.idx,
        )
        activation_maps = compute_competitor_activation_maps(grad_pack)
        ranked_all.extend(
            rank_competitor_regions(
                gap_cam=gap_cam_pack.gap_cam,
                abs_map=activation_maps.abs_map,
                pixel_grad_raw=grad_pack.pixel_grad_raw,
                competitor_idx=entry.idx,
                raw_h=raw_h,
                raw_w=raw_w,
                top_regions=top_regions_per_competitor,
                expand=expand,
                pixel_grad_top_n=pixel_grad_top_n,
            )
        )

    ranked_all.sort(key=lambda region: region.region_score, reverse=True)
    return ranked_all


def region_mask_from_ranked_regions(
    ranked_regions: list[RankedRegion],
    raw_h: int,
    raw_w: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean [H, W] mask covering all ranked region boxes."""
    mask = torch.zeros((raw_h, raw_w), dtype=torch.bool, device=device)
    for region in ranked_regions:
        y1, y2, x1, x2 = region.image_box
        mask[y1:y2, x1:x2] = True
    return mask


def rank_feature_cells(abs_map: torch.Tensor, top_n: int) -> list[tuple[int, int]]:
    """Rank feature cells by activation-gradient importance (pair with feature_cell_to_raw_image_box)."""
    flat = abs_map.reshape(-1)
    if flat.numel() == 0 or top_n <= 0:
        return []

    count = min(top_n, flat.numel())
    _, indices = torch.topk(flat, k=count, largest=True)
    height, width = abs_map.shape
    ranked: list[tuple[int, int]] = []
    for index in indices.tolist():
        yf = int(index // width)
        xf = int(index % width)
        ranked.append((yf, xf))
    return ranked


def gap_cam_from_feature_grad(
    features: torch.Tensor,
    feature_grad: torch.Tensor | None,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """Legacy activation-gradient map; prefer compute_gap_cam for competitor weighting."""
    if feature_grad is None:
        return torch.ones(target_hw, device=features.device, dtype=features.dtype)

    weights = feature_grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True).relu()
    cam = F.interpolate(
        cam,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    cam = cam.squeeze(0).squeeze(0)
    peak = float(cam.max().item())
    if peak > 1e-12:
        cam = cam / peak
    return cam.detach()


def compute_raw_space_gradients(
    model: torch.nn.Module,
    state: AttackState,
    competitor_idx: int,
) -> CompetitorGapGradients:
    """Compute gap gradients for current attack state and one competitor."""
    return compute_competitor_gap_gradients(
        model=model,
        adv_raw=state.adv_raw,
        true_idx=state.true_idx,
        competitor_idx=competitor_idx,
    )


def _sparse_update_mask(
    guided_grad: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    flat = guided_grad.abs().reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(guided_grad, dtype=torch.bool)

    count = max(1, int(flat.numel() * fraction))
    threshold = torch.topk(flat, count, largest=True).values[-1]
    return guided_grad.abs() >= threshold


def clone_attack_state(state: AttackState) -> AttackState:
    return AttackState(
        clean=state.clean,
        delta=state.delta.clone(),
        changed_mask=state.changed_mask.clone(),
        true_idx=state.true_idx,
        logits=state.logits.clone(),
        current_competitor_idx=state.current_competitor_idx,
        top_k_competitors=list(state.top_k_competitors),
        accepted_groups=list(state.accepted_groups),
    )


def _count_changed_pixels(state: AttackState) -> int:
    return int(state.changed_mask.sum().item())


def _compute_path_rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())


def build_beam_node(
    state: AttackState,
    *,
    recent_gain_per_pixel: float = 0.0,
    round_id: int = 0,
    expansion_idx: int = 0,
) -> BeamNode:
    logits_row = state.logits
    pred_idx = int(logits_row.argmax().item())
    untargeted = untargeted_gap(logits_row.unsqueeze(0), state.true_idx)
    adv = state.adv
    return BeamNode(
        state=state,
        untargeted_gap=float(untargeted),
        changed_pixels=_count_changed_pixels(state),
        recent_gain_per_pixel=float(recent_gain_per_pixel),
        rmse=_compute_path_rmse(state.clean, adv),
        norm=state.linf,
        flipped=pred_idx != state.true_idx,
        pred_idx=pred_idx,
        round_id=round_id,
        expansion_idx=expansion_idx,
    )


def beam_rank_key(
    node: BeamNode,
    *,
    min_delta: float,
    max_delta: float,
) -> tuple[float, ...]:
    """
    Rank beam paths (lower is better):

    1. flipped with valid norm
    2. lower untargeted gap
    3. fewer changed pixel-channels
    4. higher recent gain_per_pixel
    5. lower RMSE
    """
    valid_flip = (
        node.flipped
        and node.norm >= float(min_delta) - 1e-9
        and node.norm <= float(max_delta) + 1e-9
    )
    if valid_flip:
        return (
            0.0,
            node.norm,
            float(node.changed_pixels),
            -node.recent_gain_per_pixel,
            node.rmse,
        )
    return (
        1.0,
        node.untargeted_gap,
        float(node.changed_pixels),
        -node.recent_gain_per_pixel,
        node.rmse,
    )


def _is_valid_flip_node(node: BeamNode, *, min_delta: float, max_delta: float) -> bool:
    return (
        node.flipped
        and node.norm >= float(min_delta) - 1e-9
        and node.norm <= float(max_delta) + 1e-9
    )


def prune_beam_to_width(
    nodes: list[BeamNode],
    *,
    beam_width: int,
    min_delta: float,
    max_delta: float,
) -> list[BeamNode]:
    if not nodes:
        return []
    ranked = sorted(
        nodes,
        key=lambda node: beam_rank_key(node, min_delta=min_delta, max_delta=max_delta),
    )
    return ranked[: max(1, int(beam_width))]


def refresh_beam_node(
    model: torch.nn.Module,
    node: BeamNode,
    *,
    top_k: int,
) -> BeamNode:
    with torch.no_grad():
        refresh_state_logits(model=model, state=node.state, top_k=top_k)
    return build_beam_node(
        node.state,
        recent_gain_per_pixel=node.recent_gain_per_pixel,
        round_id=node.round_id,
        expansion_idx=node.expansion_idx,
    )


def _expansion_target(
    expansion_idx: int,
    competitors: list[CompetitorEntry],
    *,
    top_regions: int,
) -> tuple[int, int] | None:
    if not competitors or top_regions <= 0:
        return None
    comp_slot = int(expansion_idx) // int(top_regions)
    region_slot = int(expansion_idx) % int(top_regions)
    if comp_slot >= len(competitors):
        return None
    return comp_slot, region_slot


def expand_beam_node(
    model: torch.nn.Module,
    node: BeamNode,
    *,
    true_idx: int,
    epsilon: float,
    min_delta: float,
    max_delta: float,
    top_k: int,
    top_regions: int,
    round_id: int,
    deadline: float | None,
) -> BeamNode | None:
    """Expand one beam path by growing a single competitor region branch."""
    if deadline is not None and time.monotonic() >= deadline:
        return None

    child_state = clone_attack_state(node.state)
    with torch.no_grad():
        race = compute_top_k_competitors(
            logits=child_state.logits.unsqueeze(0),
            true_idx=true_idx,
            k=top_k,
        )
    if not race.competitors:
        return None

    max_attempts = min(len(race.competitors) * int(top_regions), 12)
    for offset in range(max_attempts):
        if deadline is not None and time.monotonic() >= deadline:
            return None

        exp_idx = node.expansion_idx + offset
        target = _expansion_target(
            exp_idx,
            list(race.competitors),
            top_regions=top_regions,
        )
        if target is None:
            break
        comp_slot, region_slot = target
        competitor = race.competitors[comp_slot]

        trial_state = clone_attack_state(node.state) if offset > 0 else child_state
        grad_pack = compute_competitor_gap_gradients(
            model=model,
            adv_raw=trial_state.adv_raw,
            true_idx=true_idx,
            competitor_idx=competitor.idx,
        )
        pixel_grad = grad_pack.pixel_grad_raw
        if pixel_grad is None or float(pixel_grad.abs().max()) <= 0.0:
            continue

        gap_cam_pack = compute_gap_cam(
            model=model,
            features=grad_pack.features,
            true_idx=true_idx,
            competitor_idx=competitor.idx,
        )
        activation_maps = compute_competitor_activation_maps(grad_pack)
        raw_h, raw_w = trial_state.clean.shape[-2:]
        ranked_regions = rank_competitor_regions(
            gap_cam=gap_cam_pack.gap_cam,
            abs_map=activation_maps.abs_map,
            pixel_grad_raw=pixel_grad,
            competitor_idx=competitor.idx,
            raw_h=raw_h,
            raw_w=raw_w,
            top_regions=top_regions,
        )
        if region_slot >= len(ranked_regions):
            continue

        grow_result = grow_ranked_region(
            model=model,
            state=trial_state,
            region=ranked_regions[region_slot],
            pixel_grad_raw=pixel_grad,
            true_idx=true_idx,
            competitor_idx=competitor.idx,
            epsilon=epsilon,
            min_delta=min_delta,
            max_delta=max_delta,
            round_id=round_id,
            deadline=deadline,
            max_pixels=REGION_GROW_INITIAL_BATCH,
        )
        if grow_result.pixels_applied <= 0 and not grow_result.seed_accepted:
            continue

        recent_gain = 0.0
        if grow_result.best_verification is not None:
            recent_gain = grow_result.best_verification.gain_per_pixel
        elif trial_state.accepted_groups:
            recent_gain = trial_state.accepted_groups[-1].gain_per_pixel

        with torch.no_grad():
            refresh_state_logits(model=model, state=trial_state, top_k=top_k)

        return build_beam_node(
            trial_state,
            recent_gain_per_pixel=recent_gain,
            round_id=round_id,
            expansion_idx=exp_idx + 1,
        )

    return None


def run_beam_search_phase(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
    *,
    epsilon: float,
    min_delta: float,
    budget: AttackTimeBudget,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_k: int = DEFAULT_BEAM_TOP_K,
    top_regions: int = DEFAULT_BEAM_TOP_REGIONS,
    max_rounds: int = DEFAULT_STEPS,
) -> tuple[list[BeamNode], BeamNode | None]:
    max_delta = _effective_max_delta(epsilon)
    min_delta = float(min_delta)
    initial_state = init_attack_state(model=model, clean=clean, true_idx=true_idx)
    beam = [build_beam_node(initial_state)]
    best_flip: BeamNode | None = None

    round_idx = 0
    while round_idx < max_rounds:
        now = time.monotonic()
        if now >= budget.search_end or now >= budget.search_round_end:
            break

        children: list[BeamNode] = []
        for node in beam:
            child = expand_beam_node(
                model=model,
                node=node,
                true_idx=true_idx,
                epsilon=epsilon,
                min_delta=min_delta,
                max_delta=max_delta,
                top_k=top_k,
                top_regions=top_regions,
                round_id=round_idx,
                deadline=budget.search_end,
            )
            if child is not None:
                children.append(child)
                if _is_valid_flip_node(child, min_delta=min_delta, max_delta=max_delta):
                    if best_flip is None or child.norm < best_flip.norm:
                        best_flip = child

        if not children:
            fallback_state = clone_attack_state(
                min(
                    beam,
                    key=lambda node: beam_rank_key(node, min_delta=min_delta, max_delta=max_delta),
                ).state
            )
            with torch.no_grad():
                race = compute_top_k_competitors(
                    logits=fallback_state.logits.unsqueeze(0),
                    true_idx=true_idx,
                    k=top_k,
                )
            if race.competitors:
                active = race.easiest.idx
                grad_pack = compute_competitor_gap_gradients(
                    model=model,
                    adv_raw=fallback_state.adv_raw,
                    true_idx=true_idx,
                    competitor_idx=active,
                )
                pixel_grad = grad_pack.pixel_grad_raw
                if pixel_grad is not None and float(pixel_grad.abs().max()) > 0.0:
                    gap_cam_pack = compute_gap_cam(
                        model=model,
                        features=grad_pack.features,
                        true_idx=true_idx,
                        competitor_idx=active,
                    )
                    activation_maps = compute_competitor_activation_maps(grad_pack)
                    raw_h, raw_w = fallback_state.clean.shape[-2:]
                    ranked_regions = rank_competitor_regions(
                        gap_cam=gap_cam_pack.gap_cam,
                        abs_map=activation_maps.abs_map,
                        pixel_grad_raw=pixel_grad,
                        competitor_idx=active,
                        raw_h=raw_h,
                        raw_w=raw_w,
                        top_regions=top_regions,
                    )
                    applied, last_verify = grow_ranked_regions(
                        model=model,
                        state=fallback_state,
                        ranked_regions=ranked_regions,
                        pixel_grad_raw=pixel_grad,
                        true_idx=true_idx,
                        competitor_idx=active,
                        epsilon=epsilon,
                        min_delta=min_delta,
                        max_delta=max_delta,
                        round_id=round_idx,
                        deadline=budget.search_end,
                        max_pixels_per_region=REGION_GROW_MAX_PIXELS_PER_REGION,
                    )
                    if applied > 0:
                        recent_gain = 0.0
                        if fallback_state.accepted_groups:
                            recent_gain = fallback_state.accepted_groups[-1].gain_per_pixel
                        child = build_beam_node(
                            fallback_state,
                            recent_gain_per_pixel=recent_gain,
                            round_id=round_idx,
                            expansion_idx=beam[0].expansion_idx + 1,
                        )
                        children.append(child)
                        if last_verify is not None and last_verify.flip_candidate:
                            if best_flip is None or last_verify.norm < best_flip.norm:
                                best_flip = build_beam_node(
                                    fallback_state,
                                    recent_gain_per_pixel=last_verify.gain_per_pixel,
                                    round_id=round_idx,
                                )
            if not children:
                break

        beam = prune_beam_to_width(
            beam + children,
            beam_width=beam_width,
            min_delta=min_delta,
            max_delta=max_delta,
        )
        round_idx += 1

    return beam, best_flip


def deepen_beam_node(
    model: torch.nn.Module,
    node: BeamNode,
    *,
    true_idx: int,
    epsilon: float,
    min_delta: float,
    max_delta: float,
    top_k: int,
    top_regions: int,
    deadline: float | None,
) -> BeamNode:
    """Run full region growth on the best beam path during the prune phase."""
    state = clone_attack_state(node.state)
    with torch.no_grad():
        race = compute_top_k_competitors(
            logits=state.logits.unsqueeze(0),
            true_idx=true_idx,
            k=top_k,
        )
    if not race.competitors:
        return node

    active = race.easiest.idx
    grad_pack = compute_competitor_gap_gradients(
        model=model,
        adv_raw=state.adv_raw,
        true_idx=true_idx,
        competitor_idx=active,
    )
    pixel_grad = grad_pack.pixel_grad_raw
    if pixel_grad is None or float(pixel_grad.abs().max()) <= 0.0:
        return node

    gap_cam_pack = compute_gap_cam(
        model=model,
        features=grad_pack.features,
        true_idx=true_idx,
        competitor_idx=active,
    )
    activation_maps = compute_competitor_activation_maps(grad_pack)
    raw_h, raw_w = state.clean.shape[-2:]
    ranked_regions = rank_competitor_regions(
        gap_cam=gap_cam_pack.gap_cam,
        abs_map=activation_maps.abs_map,
        pixel_grad_raw=pixel_grad,
        competitor_idx=active,
        raw_h=raw_h,
        raw_w=raw_w,
        top_regions=top_regions,
    )
    grow_ranked_regions(
        model=model,
        state=state,
        ranked_regions=ranked_regions,
        pixel_grad_raw=pixel_grad,
        true_idx=true_idx,
        competitor_idx=active,
        epsilon=epsilon,
        min_delta=min_delta,
        max_delta=max_delta,
        round_id=node.round_id,
        deadline=deadline,
        max_pixels_per_region=REGION_GROW_MAX_PIXELS_PER_REGION,
    )
    recent_gain = node.recent_gain_per_pixel
    if state.accepted_groups:
        recent_gain = state.accepted_groups[-1].gain_per_pixel
    with torch.no_grad():
        refresh_state_logits(model=model, state=state, top_k=top_k)
    return build_beam_node(
        state,
        recent_gain_per_pixel=recent_gain,
        round_id=node.round_id,
        expansion_idx=node.expansion_idx,
    )


def prune_beam_candidates(
    model: torch.nn.Module,
    beam: list[BeamNode],
    *,
    true_idx: int,
    top_k: int,
    beam_width: int,
    min_delta: float,
    max_delta: float,
    budget: AttackTimeBudget,
) -> list[BeamNode]:
    """
    Re-score surviving paths and drop dominated candidates during the prune phase.
    """
    if not beam:
        return beam

    refreshed: list[BeamNode] = []
    for node in beam:
        if time.monotonic() >= budget.prune_end:
            break
        refreshed.append(refresh_beam_node(model=model, node=node, top_k=top_k))

    if not refreshed:
        return beam

    ranked_for_deepen = sorted(
        refreshed,
        key=lambda node: beam_rank_key(node, min_delta=min_delta, max_delta=max_delta),
    )
    deepened: list[BeamNode] = []
    for candidate in ranked_for_deepen[: max(1, beam_width)]:
        if time.monotonic() >= budget.prune_end:
            deepened.append(candidate)
            continue
        deepened.append(
            deepen_beam_node(
                model=model,
                node=candidate,
                true_idx=true_idx,
                epsilon=epsilon,
                min_delta=min_delta,
                max_delta=max_delta,
                top_k=top_k,
                top_regions=DEFAULT_BEAM_TOP_REGIONS,
                deadline=budget.prune_end,
            )
        )
    refreshed = deepened

    refreshed.sort(key=lambda node: beam_rank_key(node, min_delta=min_delta, max_delta=max_delta))

    pruned: list[BeamNode] = []
    for candidate in refreshed:
        dominated = False
        for keeper in pruned:
            keeper_flip = _is_valid_flip_node(keeper, min_delta=min_delta, max_delta=max_delta)
            cand_flip = _is_valid_flip_node(candidate, min_delta=min_delta, max_delta=max_delta)
            if keeper_flip and not cand_flip:
                dominated = True
                break
            if keeper_flip and cand_flip and keeper.norm <= candidate.norm + 1e-9:
                if keeper.changed_pixels <= candidate.changed_pixels:
                    dominated = True
                    break
            if not keeper_flip and not cand_flip:
                if (
                    keeper.untargeted_gap <= candidate.untargeted_gap + 1e-9
                    and keeper.changed_pixels <= candidate.changed_pixels
                    and keeper.rmse <= candidate.rmse + 1e-9
                ):
                    dominated = True
                    break
        if not dominated:
            pruned.append(candidate)

    return prune_beam_to_width(
        pruned,
        beam_width=beam_width,
        min_delta=min_delta,
        max_delta=max_delta,
    )


def validator_passes_adv(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_idx: int,
    *,
    min_delta: float,
    max_delta: float,
) -> bool:
    if adv.shape != clean.shape:
        return False
    pred_idx = inference_predict_index(model=model, image_chw=adv)
    norm = float((adv - clean).abs().max().item())
    if pred_idx == true_idx:
        return False
    if norm < float(min_delta) - 1e-9 or norm > float(max_delta) + 1e-9:
        return False
    ssim = compute_ssim(clean, adv)
    psnr_db = compute_psnr_db(clean, adv)
    return ssim >= float(MIN_SSIM) and psnr_db >= float(MIN_PSNR_DB)


def final_validate_roundtrip(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_idx: int,
    *,
    min_delta: float,
    max_delta: float,
) -> torch.Tensor:
    """PNG encode/decode roundtrip plus validator-style forward checks."""
    candidate = adv.detach().clamp(0.0, 1.0)
    try:
        roundtrip = decode_image_b64(encode_image_b64(candidate)).to(device=clean.device)
    except Exception:
        roundtrip = candidate

    if roundtrip.shape != clean.shape:
        return candidate

    if validator_passes_adv(
        model=model,
        clean=clean,
        adv=roundtrip,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        return roundtrip.clamp(0.0, 1.0)

    if validator_passes_adv(
        model=model,
        clean=clean,
        adv=candidate,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        return candidate

    return candidate


def select_best_beam_node(
    beam: list[BeamNode],
    *,
    min_delta: float,
    max_delta: float,
    best_flip: BeamNode | None = None,
) -> BeamNode | None:
    if best_flip is not None and _is_valid_flip_node(
        best_flip,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        return best_flip
    if not beam:
        return None
    return min(
        beam,
        key=lambda node: beam_rank_key(node, min_delta=min_delta, max_delta=max_delta),
    )


def _pgd_fallback_attack(
    model: torch.nn.Module,
    state: AttackState,
    true_idx: int,
    *,
    max_delta: float,
    min_delta: float,
    deadline: float,
    steps: int = 10,
) -> torch.Tensor:
    adv = state.adv.clone()
    true_tensor = torch.tensor([true_idx], device=state.clean.device)
    fallback_step = max(max_delta / 4.0, PIXEL_STEP_RAW)
    for _ in range(steps):
        if time.monotonic() >= deadline:
            break
        adv_raw = adv.detach().clone().requires_grad_(True)
        logits, _ = forward_logits_features(model=model, image_bchw=adv_raw.unsqueeze(0))
        loss = F.cross_entropy(logits, true_tensor)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = adv_raw.grad
        if grad is None:
            break
        adv = adv.detach() + fallback_step * grad.sign()
        adv = torch.max(
            torch.min(adv, state.clean + max_delta),
            state.clean - max_delta,
        ).clamp(0.0, 1.0)
        state.delta = (adv - state.clean).detach()
        refresh_state_logits(model=model, state=state, top_k=DEFAULT_TOP_K)
        race = compute_top_k_competitors(logits=state.logits.unsqueeze(0), true_idx=true_idx, k=1)
        if race.is_success(state.logits.unsqueeze(0)) and state.linf >= min_delta:
            return state.adv
    return adv


def feature_guided_sparse_untargeted_attack(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
    epsilon: float,
    min_delta: float,
    timeout_seconds: float | int,
    *,
    steps: int = DEFAULT_STEPS,
    sparse_fraction: float = SPARSE_PIXEL_FRACTION,
    top_k: int = DEFAULT_BEAM_TOP_K,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    top_regions: int = DEFAULT_BEAM_TOP_REGIONS,
) -> torch.Tensor:
    """
    Untargeted attack in decoded raw image space [3, H, W] in [0, 1].

    Timeout-aware beam search with phased budgets:
    60% search, 25% pruning, 10% PNG roundtrip validation, 5% buffer.
    """
    del sparse_fraction  # kept for miner API compatibility

    max_delta = _effective_max_delta(epsilon)
    min_delta = float(min_delta)
    budget = AttackTimeBudget.from_timeout(timeout_seconds)

    effective_beam_width = int(beam_width)
    if float(timeout_seconds) <= float(TIMEOUT_SECONDS):
        effective_beam_width = min(effective_beam_width, BEAM_WIDTH_FAST)

    beam, best_flip = run_beam_search_phase(
        model=model,
        clean=clean,
        true_idx=true_idx,
        epsilon=epsilon,
        min_delta=min_delta,
        budget=budget,
        beam_width=effective_beam_width,
        top_k=int(top_k),
        top_regions=int(top_regions),
        max_rounds=int(steps),
    )

    if time.monotonic() < budget.prune_end:
        beam = prune_beam_candidates(
            model=model,
            beam=beam,
            true_idx=true_idx,
            top_k=int(top_k),
            beam_width=effective_beam_width,
            min_delta=min_delta,
            max_delta=max_delta,
            budget=budget,
        )

    best_node = select_best_beam_node(
        beam,
        min_delta=min_delta,
        max_delta=max_delta,
        best_flip=best_flip,
    )

    if best_node is None:
        fallback_state = init_attack_state(model=model, clean=clean, true_idx=true_idx)
        adv = fallback_state.adv
    else:
        adv = best_node.adv
        fallback_state = clone_attack_state(best_node.state)

    if time.monotonic() < budget.validate_end:
        adv = final_validate_roundtrip(
            model=model,
            clean=clean,
            adv=adv,
            true_idx=true_idx,
            min_delta=min_delta,
            max_delta=max_delta,
        )

    if not validator_passes_adv(
        model=model,
        clean=clean,
        adv=adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        adv = _pgd_fallback_attack(
            model=model,
            state=fallback_state,
            true_idx=true_idx,
            max_delta=max_delta,
            min_delta=min_delta,
            deadline=budget.hard_end,
            steps=min(10, int(steps)),
        )

    return adv.clamp(0.0, 1.0)
