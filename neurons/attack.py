from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from perturbnet.constants import MIN_PSNR_DB, MIN_SSIM, TIMEOUT_SECONDS
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import forward_logits_features, logits_for_images, predict_index, predict_label

from neurons.attack_log import AttackLogSession, build_validator_snapshot, idx_label

logger = logging.getLogger(__name__)

FEATURE_MAP_SIZE = 15
MODEL_CROP_SIZE = 480
DEFAULT_STEPS = 40

# Feature-cell box expansion in 480×480 crop space (one cell ≈ 32×32 px).
FEATURE_CELL_EXPAND_FIRST = 2.0
FEATURE_CELL_EXPAND_REFINE = 1.5

# Validator-visible pixel step in decoded [0, 1] image space.
PIXEL_STEP_RAW = 1.0 / 255.0

# IMPORTANT:
# Winner-quality subnet attacks should stay at exactly one uint8 step.
# Validator winners are commonly norm=0.003922, i.e. 1/255.
# Do not allow PGD/fallback to use 2/255, 3/255, etc.

# Sparse fallback tries only one-step ±1/255 candidates, never dense multi-step PGD.
SPARSE_FALLBACK_PREFIX_SIZES = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SPARSE_FALLBACK_MAX_CANDIDATES = int(os.getenv("PERTURB_SPARSE_FALLBACK_MAX_CANDIDATES", "20000"))

# Untargeted top-K competitor race.
TOP_K_FAST = 5
TOP_K_STARTING = 8
TOP_K_STRONG = 20
DEFAULT_TOP_K = TOP_K_STRONG

# Region ranking.
TOP_REGIONS_FAST = 4
TOP_REGIONS_STARTING = 6
TOP_REGIONS_QUALITY = 12
DEFAULT_TOP_REGIONS_PER_COMPETITOR = TOP_REGIONS_QUALITY
REGION_PIXEL_GRAD_TOP_N = 32

# Discrete ±1/255 pixel selection inside ranked regions.
DEFAULT_TOP_PIXELS_PER_REGION = 8
DEFAULT_MAX_PIXELS_PER_STEP = 24
DEFAULT_MIN_GAIN_PER_PIXEL = 1e-6

# Region growing after a verified seed batch (step 13): batch 8 -> 64.
REGION_GROW_INITIAL_BATCH = 8
REGION_GROW_MAX_BATCH = 64
REGION_GROW_MIN_BATCH = 4
REGION_GROW_MAX_PIXELS_PER_REGION = 64
REGION_GROW_STRONG_GAIN_PER_PIXEL = 1e-4
REGION_GROW_CLOSE_TO_FLIP_GAP = 0.5
REGION_GROW_CLOSE_TO_FLIP_RATIO = 0.15
REGION_GROW_MAX_FAILURES = 2

# Beam search (step 14): strong offline default K=20, beam=8, regions=12.
BEAM_WIDTH_FAST = 2
BEAM_WIDTH_STARTING = 4
BEAM_WIDTH_STRONG = 8
DEFAULT_BEAM_WIDTH = BEAM_WIDTH_STRONG
DEFAULT_BEAM_TOP_K = TOP_K_STRONG
DEFAULT_BEAM_TOP_REGIONS = TOP_REGIONS_QUALITY
ATTACK_TIME_SEARCH_FRACTION = 0.60
ATTACK_TIME_PRUNE_FRACTION = 0.25
ATTACK_TIME_VALIDATE_FRACTION = 0.10
ATTACK_TIME_BUFFER_FRACTION = 0.05
ATTACK_SEARCH_ROUND_FRACTION = 0.75

# Safety flip margin before aggressive final pruning (step 16).
DEFAULT_FLIP_MARGIN_BEFORE_PRUNE = 0.03

# Validator-score aware pruning chunk sizes (step 15).
PRUNE_CHUNK_SIZES = (16, 8, 4, 1)


@dataclass(frozen=True)
class AttackHyperparams:
    """Tunable attack-engine profile for subnet timeout vs offline testing."""

    top_k: int
    beam_width: int
    top_regions_per_competitor: int
    feature_box_expand: float = FEATURE_CELL_EXPAND_FIRST
    region_grow_initial_batch: int = REGION_GROW_INITIAL_BATCH
    region_grow_max_batch: int = REGION_GROW_MAX_BATCH
    region_grow_min_batch: int = REGION_GROW_MIN_BATCH
    region_grow_max_pixels_per_region: int = REGION_GROW_MAX_PIXELS_PER_REGION
    search_time_fraction: float = ATTACK_TIME_SEARCH_FRACTION
    prune_time_fraction: float = ATTACK_TIME_PRUNE_FRACTION
    validate_time_fraction: float = ATTACK_TIME_VALIDATE_FRACTION
    buffer_time_fraction: float = ATTACK_TIME_BUFFER_FRACTION
    flip_margin_before_prune: float = DEFAULT_FLIP_MARGIN_BEFORE_PRUNE
    shrink_beam_on_short_timeout: bool = False


ATTACK_PRESET_FAST = AttackHyperparams(
    top_k=TOP_K_FAST,
    beam_width=BEAM_WIDTH_FAST,
    top_regions_per_competitor=TOP_REGIONS_FAST,
    shrink_beam_on_short_timeout=True,
)

ATTACK_PRESET_DEFAULT = AttackHyperparams(
    top_k=TOP_K_STARTING,
    beam_width=BEAM_WIDTH_STARTING,
    top_regions_per_competitor=TOP_REGIONS_STARTING,
    shrink_beam_on_short_timeout=True,
)

ATTACK_PRESET_STRONG = AttackHyperparams(
    top_k=TOP_K_STRONG,
    beam_width=BEAM_WIDTH_STRONG,
    top_regions_per_competitor=TOP_REGIONS_QUALITY,
    region_grow_initial_batch=8,
    region_grow_max_batch=64,
    region_grow_max_pixels_per_region=64,
    shrink_beam_on_short_timeout=False,
)

DEFAULT_ATTACK_HYPERPARAMS = ATTACK_PRESET_DEFAULT


def resolve_attack_hyperparams(preset: str | None = None) -> AttackHyperparams:
    """
    Resolve attack profile from explicit name or PERTURB_ATTACK_PRESET env var.

    Presets: fast | default | strong (offline)
    """
    name = (preset or os.getenv("PERTURB_ATTACK_PRESET") or "default").strip().lower()
    if name in {"fast", "quick", "subnet"}:
        return ATTACK_PRESET_FAST
    if name in {"default", "balanced", "starting", "start"}:
        return ATTACK_PRESET_DEFAULT
    if name in {"strong", "offline", "quality"}:
        return ATTACK_PRESET_STRONG
    logger.warning("Unknown attack preset %r; using default live-miner profile", name)
    return ATTACK_PRESET_DEFAULT


@dataclass(frozen=True)
class CandidateStats:
    norm: float
    rmse: float


@dataclass(frozen=True)
class FlipGapStats:
    logits: torch.Tensor
    true_idx: int
    pred_idx: int
    best_other_idx: int
    untargeted_gap: float
    norm: float
    rmse: float

    @property
    def flipped(self) -> bool:
        return self.pred_idx != self.true_idx

@dataclass(frozen=True)
class PngRoundtripResult:
    final_adv: torch.Tensor
    encoded_b64: str
    decoded_adv: torch.Tensor
    pred_idx: int
    norm: float
    rmse: float
    passed: bool
    restored_from_backup: bool
    reason: str


@dataclass
class FeatureGuidedAttackOutput:
    adv: torch.Tensor
    pre_prune_adv: torch.Tensor
    accepted_groups: list[AcceptedGroup]
    fallback_state: AttackState
    flip_stats: FlipGapStats | None = None
    log_session: AttackLogSession | None = None


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
    weak_count: int = 0
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


def inference_predict_index(model: torch.nn.Module, image_chw: torch.Tensor) -> int:
    """Canonical class index via perturbnet.model (includes PREPROCESS)."""
    return predict_index(model=model, image_chw=image_chw)


def inference_predict_label(model: torch.nn.Module, image_chw: torch.Tensor) -> str:
    """Canonical label string via perturbnet.model (includes PREPROCESS)."""
    return predict_label(model=model, image_chw=image_chw)


def _effective_max_delta(epsilon: float) -> float:
    """Linf cap passed from the miner (already clamped to subnet policy)."""
    return float(epsilon)


def _uses_one_step_linf_cap(max_delta: float) -> bool:
    """True when the miner capped Linf to a single ±1/255 step."""
    return float(max_delta) <= PIXEL_STEP_RAW + 1e-9


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


def competitor_gap(logits: torch.Tensor, true_idx: int, competitor_idx: int) -> float:
    row = _logits_row(logits)
    return float((row[true_idx] - row[competitor_idx]).item())


def untargeted_gap(logits: torch.Tensor, true_idx: int) -> float:
    race = compute_top_k_competitors(logits=logits, true_idx=true_idx, k=1)
    if not race.competitors:
        return 0.0
    return race.competitors[0].gap_k


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
    competitor_gain_ok = (
        real_gain > 0.0 and gain_per_pixel >= float(min_gain_per_pixel)
    )
    strong_competitor_gain = (
        competitor_gain_ok
        and classify_region_gain_strength(real_gain, gain_per_pixel) == "strong"
    )
    competitor_close_to_flip = competitor_gain_ok and is_gap_close_to_flip(gap_after, gap_before)

    progress_accept = (
        competitor_gain_ok
        and norm_ok
        and value_ok
        and (
            untargeted_improved
            or strong_competitor_gain
            or competitor_close_to_flip
        )
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
    elif gain_per_pixel < float(min_gain_per_pixel):
        reason = "weak_gain_per_pixel"
    elif not untargeted_improved and not strong_competitor_gain and not competitor_close_to_flip:
        reason = "competitor_gain_not_strong_enough"
    elif not untargeted_improved:
        reason = "competitor_specific_gain"
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


def _clamp_grow_batch_size(batch_size: int) -> int:
    return max(REGION_GROW_MIN_BATCH, min(REGION_GROW_MAX_BATCH, int(batch_size)))


def _increase_grow_batch_size(batch_size: int) -> int:
    return _clamp_grow_batch_size(max(batch_size + 4, batch_size * 2))


def _decrease_grow_batch_size(batch_size: int) -> int:
    return _clamp_grow_batch_size(max(REGION_GROW_MIN_BATCH, batch_size // 2))


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
    log_session: AttackLogSession | None = None,
    beam_id: int | None = None,
    batch_size: int | None = None,
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
        if log_session is not None:
            log_session.log_attack_batch(
                verification=verification,
                region=region,
                competitor_idx=competitor_idx,
                batch_size=int(batch_size or len(pixel_changes)),
                accepted=False,
                reason=reason,
                beam_id=beam_id,
                round_id=round_id,
                gap_before=gap_before,
            )
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
    if log_session is not None:
        log_session.log_attack_batch(
            verification=verification,
            region=region,
            competitor_idx=competitor_idx,
            batch_size=int(batch_size or len(pixel_changes)),
            accepted=True,
            reason=reason,
            beam_id=beam_id,
            round_id=round_id,
            gap_before=gap_before,
        )
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
    initial_batch: int | None = None,
    log_session: AttackLogSession | None = None,
    beam_id: int | None = None,
) -> RegionGrowResult:
    """
    Seed then adaptively grow one ranked region in verified pixel batches.

    Stops on flip, saturation (gain <= 0), two consecutive failures, pixel cap, or budget.
    """
    pixel_cap = int(max_pixels) if max_pixels is not None else REGION_GROW_MAX_PIXELS_PER_REGION
    session = RegionGrowSession(
        region=region,
        batch_size=int(initial_batch or REGION_GROW_INITIAL_BATCH),
    )
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
            log_session=log_session,
            beam_id=beam_id,
            batch_size=batch_size,
        )

        if verification is not None:
            if verification.flip_candidate and (
                best_verification is None or verification.norm < best_verification.norm
            ):
                best_verification = verification

        if not accepted:
            session.failure_count += 1
            if reason in {"weak_gain_not_close_to_flip", "weak_gain_per_pixel"}:
                session.weak_count += 1
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
            stopped_reason = "flip_found"
            session.stopped = True
            break

    if session.stopped and stopped_reason == "not_started":
        stopped_reason = "max_pixels" if session.pixels_applied >= pixel_cap else "done"

    if log_session is not None:
        log_session.log_region_saturation(
            region=region,
            competitor_idx=competitor_idx,
            fail_count=session.failure_count,
            weak_count=session.weak_count,
            pixels_applied=session.pixels_applied,
            stopped_reason=stopped_reason,
            round_id=round_id,
        )

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
            break
        if result.stopped_reason == "flip_found":
            break

    return total_applied, best_flip


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


def _as_feature_map(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features[0]
    if features.ndim == 3:
        return features
    raise ValueError(f"Expected feature map [1,C,H,W] or [C,H,W], got {tuple(features.shape)}")


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
    Lower is better.

    In strict one-step Linf mode, many successful candidates have the same norm=1/255.
    Therefore shortest RMSE is mostly controlled by changed pixel-channel count.
    """
    valid_flip = (
        node.flipped
        and node.norm >= float(min_delta) - 1e-9
        and node.norm <= float(max_delta) + 1e-9
    )

    if valid_flip:
        margin = -float(node.untargeted_gap)  # bigger positive margin is safer
        return (
            0.0,
            float(node.changed_pixels),
            float(node.rmse),
            float(node.norm),
            -margin,
            -float(node.recent_gain_per_pixel),
        )

    return (
        1.0,
        float(node.untargeted_gap),
        float(node.changed_pixels),
        float(node.rmse),
        -float(node.recent_gain_per_pixel),
    )


def _better_flip_node(
    candidate: BeamNode,
    current: BeamNode | None,
    *,
    min_delta: float,
    max_delta: float,
) -> bool:
    if not _is_valid_flip_node(candidate, min_delta=min_delta, max_delta=max_delta):
        return False
    if current is None:
        return True
    return beam_rank_key(candidate, min_delta=min_delta, max_delta=max_delta) < beam_rank_key(
        current,
        min_delta=min_delta,
        max_delta=max_delta,
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
    region_grow_initial_batch: int = REGION_GROW_INITIAL_BATCH,
    region_grow_max_pixels_per_region: int = REGION_GROW_MAX_PIXELS_PER_REGION,
    log_session: AttackLogSession | None = None,
    beam_id: int | None = None,
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

    max_attempts = min(len(race.competitors) * int(top_regions), 4)
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
            max_pixels=region_grow_max_pixels_per_region,
            initial_batch=region_grow_initial_batch,
            log_session=log_session,
            beam_id=beam_id,
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
    region_grow_initial_batch: int = REGION_GROW_INITIAL_BATCH,
    region_grow_max_pixels_per_region: int = REGION_GROW_MAX_PIXELS_PER_REGION,
    log_session: AttackLogSession | None = None,
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

        if log_session is not None:
            logits_source = beam[0].state.logits if beam else initial_state.logits
            with torch.no_grad():
                race = compute_top_k_competitors(
                    logits=logits_source.unsqueeze(0),
                    true_idx=true_idx,
                    k=top_k,
                )
            log_session.log_topk_competitors(race, round_id=round_idx)

        children: list[BeamNode] = []
        for beam_idx, node in enumerate(beam):
            if time.monotonic() >= budget.search_end:
                break
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
                region_grow_initial_batch=region_grow_initial_batch,
                region_grow_max_pixels_per_region=region_grow_max_pixels_per_region,
                log_session=log_session,
                beam_id=beam_idx,
            )
            if child is not None:
                children.append(child)
                if _better_flip_node(child, best_flip, min_delta=min_delta, max_delta=max_delta):
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
                        max_pixels_per_region=region_grow_max_pixels_per_region,
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
                            candidate_flip = build_beam_node(
                                fallback_state,
                                recent_gain_per_pixel=last_verify.gain_per_pixel,
                                round_id=round_idx,
                            )
                            if _better_flip_node(candidate_flip, best_flip, min_delta=min_delta, max_delta=max_delta):
                                best_flip = candidate_flip
            if not children:
                break

        beam = prune_beam_to_width(
            beam + children,
            beam_width=beam_width,
            min_delta=min_delta,
            max_delta=max_delta,
        )
        if log_session is not None:
            log_session.log_beam_round(beam, round_id=round_idx)
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
    region_grow_max_pixels_per_region: int = REGION_GROW_MAX_PIXELS_PER_REGION,
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
        max_pixels_per_region=region_grow_max_pixels_per_region,
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


def _region_group_key(group: AcceptedGroup) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    return (group.competitor_idx, group.feature_cell, group.image_box)


def _group_accepted_region_batches(
    accepted_groups: list[AcceptedGroup],
) -> list[tuple[tuple[int, tuple[int, int], tuple[int, int, int, int]], list[AcceptedGroup]]]:
    grouped: dict[tuple[int, tuple[int, int], tuple[int, int, int, int]], list[AcceptedGroup]] = {}
    for group in accepted_groups:
        grouped.setdefault(_region_group_key(group), []).append(group)

    region_groups: list[tuple[tuple[int, tuple[int, int], tuple[int, int, int, int]], list[AcceptedGroup]]] = []
    for key, batches in grouped.items():
        batches.sort(key=lambda batch: (batch.round_id, -batch.gain_per_pixel))
        region_groups.append((key, batches))

    region_groups.sort(
        key=lambda item: (
            sum(batch.gain for batch in item[1]),
            sum(batch.gain_per_pixel for batch in item[1]),
        )
    )
    return region_groups


def delta_from_pixel_changes(
    clean: torch.Tensor,
    pixel_changes: list[PixelChange],
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = torch.zeros_like(clean)
    changed_mask = torch.zeros_like(clean, dtype=torch.bool)
    for change in pixel_changes:
        step = float(change.direction) * PIXEL_STEP_RAW
        delta[change.channel, change.y, change.x] += step
        changed_mask[change.channel, change.y, change.x] = True
    return delta, changed_mask


def reverse_pixel_changes(
    delta: torch.Tensor,
    changed_mask: torch.Tensor,
    pixel_changes: list[PixelChange],
) -> tuple[torch.Tensor, torch.Tensor]:
    new_delta = delta.clone()
    new_mask = changed_mask.clone()
    for change in pixel_changes:
        step = float(change.direction) * PIXEL_STEP_RAW
        c, y, x = change.channel, change.y, change.x
        new_delta[c, y, x] = new_delta[c, y, x] - step
        if abs(float(new_delta[c, y, x].item())) <= 1e-12:
            new_delta[c, y, x] = 0.0
            new_mask[c, y, x] = False
    return new_delta, new_mask


def _removal_breaks_min_delta(
    delta: torch.Tensor,
    pixel_changes: list[PixelChange],
    *,
    min_delta: float,
) -> bool:
    """
    Reject removals that drop Linf below validator min_delta floor.

    IMPORTANT:
    Existing prune call sites pass only:
        _removal_breaks_min_delta(state.delta, pixels, min_delta=min_delta)

    So this function must rebuild changed_mask internally.
    """
    if not pixel_changes:
        return False

    changed_mask = delta.abs() > 1e-12
    trial_delta, _ = reverse_pixel_changes(delta, changed_mask, pixel_changes)
    new_linf = float(trial_delta.abs().max().item())
    return new_linf < float(min_delta) - 1e-9


def _collect_pixels_newest_first(accepted_groups: list[AcceptedGroup]) -> list[PixelChange]:
    pixels: list[PixelChange] = []
    for batch in reversed(accepted_groups):
        pixels.extend(batch.pixels)
    return pixels


def _prune_preserves_validator_flip(
    model: torch.nn.Module,
    clean: torch.Tensor,
    trial_delta: torch.Tensor,
    true_idx: int,
    *,
    min_delta: float,
    max_delta: float,
) -> bool:
    adv = (clean + trial_delta).clamp(0.0, 1.0)
    return validator_passes_adv(
        model=model,
        clean=clean,
        adv=adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    )


def _commit_pruned_state(
    model: torch.nn.Module,
    state: AttackState,
    trial_delta: torch.Tensor,
    trial_mask: torch.Tensor,
    accepted_groups: list[AcceptedGroup],
    *,
    top_k: int,
) -> None:
    state.delta = trial_delta.detach()
    state.changed_mask = trial_mask.detach()
    state.accepted_groups = list(accepted_groups)
    refresh_state_logits(model=model, state=state, top_k=top_k)


def prune_validator_aware_state(
    model: torch.nn.Module,
    state: AttackState,
    *,
    true_idx: int,
    min_delta: float,
    max_delta: float,
    top_k: int = DEFAULT_TOP_K,
    deadline: float | None = None,
    log_session: AttackLogSession | None = None,
) -> AttackState:
    """
    Validator-score aware pruning (step 15).

    A. Remove weakest accepted region groups whole
    B. Remove later-added batches inside each region group
    C. Remove pixel chunks (16 -> 8 -> 4 -> 1), newest first
    D. Never drop Linf below min_delta
    """
    if not state.accepted_groups:
        return state

    adv = state.adv
    if not validator_passes_adv(
        model=model,
        clean=state.clean,
        adv=adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        return state

    before_snap = build_validator_snapshot(
        logits=state.logits,
        true_idx=true_idx,
        clean=state.clean,
        adv=adv,
        changed_mask=state.changed_mask,
    )
    if log_session is not None:
        log_session.log_prune_start(before_snap)

    remaining_groups = list(state.accepted_groups)

    # A. Remove whole accepted region groups, weakest first.
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        region_groups = _group_accepted_region_batches(remaining_groups)
        if not region_groups:
            break

        removed_any = False
        for region_key, batches in region_groups:
            pixels_to_remove = [pixel for batch in batches for pixel in batch.pixels]
            if not pixels_to_remove:
                remaining_groups = [
                    group for group in remaining_groups if _region_group_key(group) != region_key
                ]
                removed_any = True
                break
            if _removal_breaks_min_delta(state.delta, pixels_to_remove, min_delta=min_delta):
                continue

            trial_delta, trial_mask = reverse_pixel_changes(
                state.delta,
                state.changed_mask,
                pixels_to_remove,
            )
            if not _prune_preserves_validator_flip(
                model=model,
                clean=state.clean,
                trial_delta=trial_delta,
                true_idx=true_idx,
                min_delta=min_delta,
                max_delta=max_delta,
            ):
                continue

            remaining_groups = [
                group for group in remaining_groups if _region_group_key(group) != region_key
            ]
            _commit_pruned_state(
                model=model,
                state=state,
                trial_delta=trial_delta,
                trial_mask=trial_mask,
                accepted_groups=remaining_groups,
                top_k=top_k,
            )
            removed_any = True
            break

        if not removed_any:
            break

    # B. Remove later-added batches inside each surviving region group.
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            break
        region_groups = _group_accepted_region_batches(remaining_groups)
        removed_any = False
        for _region_key, batches in region_groups:
            if len(batches) <= 1:
                continue
            batch = batches[-1]
            if batch not in remaining_groups:
                continue
            if _removal_breaks_min_delta(state.delta, batch.pixels, min_delta=min_delta):
                continue

            trial_delta, trial_mask = reverse_pixel_changes(
                state.delta,
                state.changed_mask,
                batch.pixels,
            )
            if not _prune_preserves_validator_flip(
                model=model,
                clean=state.clean,
                trial_delta=trial_delta,
                true_idx=true_idx,
                min_delta=min_delta,
                max_delta=max_delta,
            ):
                continue

            remaining_groups = [group for group in remaining_groups if group is not batch]
            _commit_pruned_state(
                model=model,
                state=state,
                trial_delta=trial_delta,
                trial_mask=trial_mask,
                accepted_groups=remaining_groups,
                top_k=top_k,
            )
            removed_any = True
            break
        if not removed_any:
            break

    # C. Remove pixel chunks newest-first at sizes 16 -> 8 -> 4 -> 1.
    for chunk_size in PRUNE_CHUNK_SIZES:
        if deadline is not None and time.monotonic() >= deadline:
            break

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break

            ordered_pixels = _collect_pixels_newest_first(remaining_groups)
            active_pixels = [
                pixel
                for pixel in ordered_pixels
                if state.changed_mask[pixel.channel, pixel.y, pixel.x]
            ]
            if len(active_pixels) < chunk_size:
                break

            removed_any = False
            max_start = max(0, len(active_pixels) - chunk_size)
            for start_idx in range(0, max_start + 1, chunk_size):
                chunk = active_pixels[start_idx : start_idx + chunk_size]
                if _removal_breaks_min_delta(state.delta, chunk, min_delta=min_delta):
                    continue

                trial_delta, trial_mask = reverse_pixel_changes(
                    state.delta,
                    state.changed_mask,
                    chunk,
                )
                if not _prune_preserves_validator_flip(
                    model=model,
                    clean=state.clean,
                    trial_delta=trial_delta,
                    true_idx=true_idx,
                    min_delta=min_delta,
                    max_delta=max_delta,
                ):
                    continue

                chunk_set = {(p.channel, p.y, p.x, p.direction) for p in chunk}
                updated_groups: list[AcceptedGroup] = []
                for group in remaining_groups:
                    kept_pixels = [
                        pixel
                        for pixel in group.pixels
                        if (pixel.channel, pixel.y, pixel.x, pixel.direction) not in chunk_set
                    ]
                    if kept_pixels:
                        updated_groups.append(
                            AcceptedGroup(
                                competitor_idx=group.competitor_idx,
                                feature_cell=group.feature_cell,
                                image_box=group.image_box,
                                pixels=kept_pixels,
                                gap_before=group.gap_before,
                                gap_after=group.gap_after,
                                gain=group.gain,
                                gain_per_pixel=group.gain_per_pixel,
                                round_id=group.round_id,
                            )
                        )
                remaining_groups = updated_groups
                _commit_pruned_state(
                    model=model,
                    state=state,
                    trial_delta=trial_delta,
                    trial_mask=trial_mask,
                    accepted_groups=remaining_groups,
                    top_k=top_k,
                )
                removed_any = True
                break

            if not removed_any:
                break

    after_snap = build_validator_snapshot(
        logits=state.logits,
        true_idx=true_idx,
        clean=state.clean,
        adv=state.adv,
        changed_mask=state.changed_mask,
    )
    if log_session is not None:
        log_session.log_prune_result(before_snap, after_snap)
        log_session.log_accepted_groups(state.accepted_groups)

    return state


def prune_beam_node_validator_aware(
    model: torch.nn.Module,
    node: BeamNode,
    *,
    true_idx: int,
    min_delta: float,
    max_delta: float,
    top_k: int,
    deadline: float | None,
    log_session: AttackLogSession | None = None,
) -> BeamNode:
    pruned_state = prune_validator_aware_state(
        model=model,
        state=clone_attack_state(node.state),
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
        top_k=top_k,
        deadline=deadline,
        log_session=log_session,
    )
    return build_beam_node(
        pruned_state,
        recent_gain_per_pixel=node.recent_gain_per_pixel,
        round_id=node.round_id,
        expansion_idx=node.expansion_idx,
    )


def prune_beam_candidates(
    model: torch.nn.Module,
    beam: list[BeamNode],
    *,
    true_idx: int,
    epsilon: float,
    top_k: int,
    beam_width: int,
    min_delta: float,
    max_delta: float,
    budget: AttackTimeBudget,
    top_regions: int = DEFAULT_BEAM_TOP_REGIONS,
    region_grow_max_pixels_per_region: int = REGION_GROW_MAX_PIXELS_PER_REGION,
    log_session: AttackLogSession | None = None,
) -> list[BeamNode]:
    """
    Re-score surviving paths, deepen, validator-aware prune, and drop dominated candidates.
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
                top_regions=top_regions,
                deadline=budget.prune_end,
                region_grow_max_pixels_per_region=region_grow_max_pixels_per_region,
            )
        )
    refreshed = deepened

    validator_pruned: list[BeamNode] = []
    for candidate in refreshed:
        if time.monotonic() >= budget.prune_end:
            validator_pruned.append(candidate)
            continue
        if _is_valid_flip_node(candidate, min_delta=min_delta, max_delta=max_delta):
            validator_pruned.append(
                prune_beam_node_validator_aware(
                    model=model,
                    node=candidate,
                    true_idx=true_idx,
                    min_delta=min_delta,
                    max_delta=max_delta,
                    top_k=top_k,
                    deadline=budget.prune_end,
                    log_session=log_session,
                )
            )
        else:
            validator_pruned.append(candidate)
    refreshed = validator_pruned

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


def candidate_stats(clean: torch.Tensor, adv: torch.Tensor) -> CandidateStats:
    norm = float((adv - clean).abs().max().item())
    rmse = _compute_path_rmse(clean, adv)
    return CandidateStats(norm=norm, rmse=rmse)

def enforce_one_step_delta(clean: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """
    Convert any candidate adv into strict {-1/255, 0, +1/255} delta.

    This is a safety guard before PNG roundtrip / final return.
    It prevents accumulated floating-point or PGD-style deltas from becoming 2/255+.
    """
    diff = (adv.detach() - clean.detach()).to(dtype=clean.dtype)
    delta = torch.zeros_like(diff)

    half_step = PIXEL_STEP_RAW * 0.5
    delta[diff > half_step] = PIXEL_STEP_RAW
    delta[diff < -half_step] = -PIXEL_STEP_RAW

    return (clean + delta).clamp(0.0, 1.0)


def one_step_linf_ok(clean: torch.Tensor, adv: torch.Tensor) -> bool:
    norm = float((adv - clean).abs().max().item())
    return norm <= PIXEL_STEP_RAW + 1e-9

def check_flip_and_gap(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_idx: int,
) -> FlipGapStats:
    logits = _logits_row(inference_logits(model=model, image_chw=adv)).detach().to(dtype=torch.float32)
    pred_idx = int(logits.argmax().item())
    masked = logits.clone()
    masked[true_idx] = float("-inf")
    best_other_idx = int(masked.argmax().item())
    stats = candidate_stats(clean, adv)
    return FlipGapStats(
        logits=logits,
        true_idx=true_idx,
        pred_idx=pred_idx,
        best_other_idx=best_other_idx,
        untargeted_gap=untargeted_gap(logits.unsqueeze(0), true_idx),
        norm=stats.norm,
        rmse=stats.rmse,
    )


def _has_flip_margin(stats: FlipGapStats, *, margin: float) -> bool:
    if not stats.flipped:
        return False
    true_logit = float(stats.logits[stats.true_idx].item())
    best_other_logit = float(stats.logits[stats.best_other_idx].item())
    return best_other_logit >= true_logit + float(margin)


def _adv_from_accepted_group(clean: torch.Tensor, group: AcceptedGroup) -> torch.Tensor:
    delta, _ = delta_from_pixel_changes(clean, group.pixels)
    return (clean + delta).clamp(0.0, 1.0)


def _roundtrip_restore_candidates(
    clean: torch.Tensor,
    adv: torch.Tensor,
    *,
    accepted_groups: list[AcceptedGroup] | None,
    restore_adv_candidates: list[torch.Tensor] | None,
) -> list[torch.Tensor]:
    """
    Candidate order should prefer the final/pruned adv first.

    Previous order tried restore backups before final adv, which can return
    a higher-RMSE pre-prune result even when final adv passes.
    """
    candidates: list[torch.Tensor] = []
    seen: set[bytes] = set()

    def _key(tensor: torch.Tensor) -> bytes:
        q = (tensor.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
        return q.numpy().tobytes()

    def _add(tensor: torch.Tensor) -> None:
        key = _key(tensor)
        if key in seen:
            return
        seen.add(key)
        candidates.append(tensor.detach().clamp(0.0, 1.0))

    # 1. Final pruned candidate first.
    _add(adv)

    # 2. Try individual high-quality groups. Sometimes one group alone flips with lower RMSE.
    if accepted_groups:
        for group in sorted(
            accepted_groups,
            key=lambda batch: (len(batch.pixels), -batch.gain_per_pixel, -batch.gain),
        ):
            _add(_adv_from_accepted_group(clean, group))

    # 3. Backups last only.
    if restore_adv_candidates:
        for item in restore_adv_candidates:
            _add(item)

    return candidates
    

def png_roundtrip_verify(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_idx: int,
    *,
    min_delta: float,
    max_delta: float,
    accepted_groups: list[AcceptedGroup] | None = None,
    restore_adv_candidates: list[torch.Tensor] | None = None,
) -> PngRoundtripResult:
    """
    PNG encode/decode roundtrip using perturbnet.image_io helpers.

    Instead of returning the first passing candidate, test all candidate sources
    and return the lowest-RMSE validator-passing decoded tensor.
    """
    strict_max_delta = float(max_delta)
    one_step_cap = _uses_one_step_linf_cap(max_delta)

    trial_adv_list = _roundtrip_restore_candidates(
        clean=clean,
        adv=adv,
        accepted_groups=accepted_groups,
        restore_adv_candidates=restore_adv_candidates,
    )

    best_result: PngRoundtripResult | None = None

    last_encoded = ""
    last_decoded: torch.Tensor | None = None
    last_pred_idx = true_idx
    last_stats = CandidateStats(norm=0.0, rmse=0.0)
    last_reason = "no_candidate"

    for index, trial_adv in enumerate(trial_adv_list):
        if one_step_cap:
            trial_adv = enforce_one_step_delta(clean, trial_adv)

        encoded = encode_image_b64(trial_adv)
        decoded = decode_image_b64(encoded).to(device=clean.device)

        stats = candidate_stats(clean, decoded)
        pred_idx = inference_predict_index(model=model, image_chw=decoded)

        last_encoded = encoded
        last_decoded = decoded
        last_pred_idx = pred_idx
        last_stats = stats

        if decoded.shape != clean.shape:
            last_reason = "shape_mismatch"
            continue

        if pred_idx == true_idx:
            last_reason = "roundtrip_not_flipped"
            continue

        if stats.norm < float(min_delta) - 1e-9:
            last_reason = "below_min_delta"
            continue

        if stats.norm > strict_max_delta + 1e-9:
            last_reason = "above_one_step_linf" if one_step_cap else "above_max_delta"
            continue

        ssim = compute_ssim(clean, decoded)
        psnr_db = compute_psnr_db(clean, decoded)
        if ssim < float(MIN_SSIM):
            last_reason = "below_min_ssim"
            continue
        if psnr_db < float(MIN_PSNR_DB):
            last_reason = "below_min_psnr"
            continue

        candidate_result = PngRoundtripResult(
            final_adv=decoded.detach().clamp(0.0, 1.0),
            encoded_b64=encoded,
            decoded_adv=decoded.detach().clamp(0.0, 1.0),
            pred_idx=pred_idx,
            norm=stats.norm,
            rmse=stats.rmse,
            passed=True,
            restored_from_backup=index > 0,
            reason="roundtrip_ok" if index == 0 else "restored_backup",
        )

        if best_result is None or candidate_result.rmse < best_result.rmse - 1e-12:
            best_result = candidate_result

    if best_result is not None:
        return best_result

    if last_decoded is None:
        fallback = enforce_one_step_delta(clean, adv) if one_step_cap else adv.detach().clamp(0.0, 1.0)
        last_encoded = encode_image_b64(fallback)
        last_decoded = decode_image_b64(last_encoded).to(device=clean.device)
        last_stats = candidate_stats(clean, last_decoded)
        last_pred_idx = inference_predict_index(model=model, image_chw=last_decoded)
        last_reason = "roundtrip_failed"

    return PngRoundtripResult(
        final_adv=last_decoded.detach().clamp(0.0, 1.0),
        encoded_b64=last_encoded,
        decoded_adv=last_decoded.detach().clamp(0.0, 1.0),
        pred_idx=last_pred_idx,
        norm=last_stats.norm,
        rmse=last_stats.rmse,
        passed=False,
        restored_from_backup=False,
        reason=last_reason,
    )


def roundtrip_pixel_prune(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_idx: int,
    *,
    min_delta: float,
    max_delta: float,
    deadline: float | None = None,
    max_checks: int = 512,
) -> torch.Tensor:
    """
    Final decoded-space pruning.

    Removes changed pixel-channels after PNG decode, then re-encodes/re-decodes
    and keeps the removal only if validator checks still pass.
    """
    best = adv.detach().clone().clamp(0.0, 1.0)

    if not validator_passes_adv(
        model=model,
        clean=clean,
        adv=best,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        return best

    changed = (best - clean).abs() > 1e-12
    coords = changed.nonzero(as_tuple=False)

    if coords.numel() == 0:
        return best

    # Try larger chunks first, then single pixels.
    checks = 0
    for chunk_size in (16, 8, 4, 1):
        improved = True
        while improved:
            improved = False
            coords = ((best - clean).abs() > 1e-12).nonzero(as_tuple=False)

            if coords.shape[0] < chunk_size:
                break

            # Newest info is unavailable after decode, so scan from the end.
            for start in range(max(0, coords.shape[0] - chunk_size), -1, -chunk_size):
                if deadline is not None and time.monotonic() >= deadline:
                    return best
                if checks >= max_checks:
                    return best

                chunk = coords[start : start + chunk_size]
                trial = best.clone()

                for c, y, x in chunk.tolist():
                    trial[int(c), int(y), int(x)] = clean[int(c), int(y), int(x)]

                if _uses_one_step_linf_cap(max_delta):
                    trial = enforce_one_step_delta(clean, trial)

                encoded = encode_image_b64(trial)
                decoded = decode_image_b64(encoded).to(device=clean.device)

                checks += 1

                if not validator_passes_adv(
                    model=model,
                    clean=clean,
                    adv=decoded,
                    true_idx=true_idx,
                    min_delta=min_delta,
                    max_delta=max_delta,
                ):
                    continue

                if candidate_stats(clean, decoded).rmse < candidate_stats(clean, best).rmse - 1e-12:
                    best = decoded.detach().clamp(0.0, 1.0)
                    improved = True
                    break

    return best


def select_best_beam_node(
    beam: list[BeamNode],
    *,
    min_delta: float,
    max_delta: float,
    best_flip: BeamNode | None = None,
) -> BeamNode | None:
    candidates = list(beam)

    if best_flip is not None and _is_valid_flip_node(
        best_flip,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        candidates.append(best_flip)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda node: beam_rank_key(node, min_delta=min_delta, max_delta=max_delta),
    )


def pgd_fallback_attack(
    model: torch.nn.Module,
    state: AttackState,
    true_idx: int,
    *,
    max_delta: float,
    min_delta: float,
    deadline: float,
    steps: int = 10,
) -> torch.Tensor:
    return _pgd_fallback_attack(
        model=model,
        state=state,
        true_idx=true_idx,
        max_delta=max_delta,
        min_delta=min_delta,
        deadline=deadline,
        steps=steps,
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
    """
    Sparse one-step fallback.

    This replaces dense multi-step PGD.
    It never applies more than ±1/255 to any decoded raw pixel-channel.

    Goal:
        Try to find a flip with norm ≈ 1/255 and as few extra pixels as possible.

    If it cannot flip, return the current state.adv instead of sending a dense 3/255 result.
    """
    del steps

    one_step_cap = _uses_one_step_linf_cap(max_delta)
    max_delta = float(max_delta)

    base_adv = enforce_one_step_delta(state.clean, state.adv) if one_step_cap else state.adv.detach().clone()
    base_delta = (base_adv - state.clean).detach()

    # If current sparse state already flips under one-step rule, keep it.
    if validator_passes_adv(
        model=model,
        clean=state.clean,
        adv=base_adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        state.delta = base_delta
        state.changed_mask = state.delta.abs() > 1e-12
        refresh_state_logits(model=model, state=state, top_k=DEFAULT_TOP_K)
        return state.adv

    if time.monotonic() >= deadline:
        return base_adv

    # Compute untargeted CE gradient from current one-step sparse state.
    true_tensor = torch.tensor([true_idx], device=state.clean.device)
    adv_raw = base_adv.detach().clone().requires_grad_(True)

    logits, _ = forward_logits_features(model=model, image_bchw=adv_raw.unsqueeze(0))
    loss = F.cross_entropy(logits, true_tensor)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad = adv_raw.grad
    if grad is None or float(grad.abs().max().item()) <= 0.0:
        return base_adv

    # Candidate direction for untargeted CE fallback:
    # increase CE loss, so move with sign(grad), but only one raw step.
    clean = state.clean
    current_changed = base_delta.abs() > 1e-12

    flat_grad = grad.detach().abs().reshape(-1)
    total = int(flat_grad.numel())
    top_n = min(int(SPARSE_FALLBACK_MAX_CANDIDATES), total)

    values, indices = torch.topk(flat_grad, k=top_n, largest=True)

    candidate_changes: list[PixelChange] = []
    _, height, width = clean.shape

    for flat_idx in indices.tolist():
        if time.monotonic() >= deadline:
            break

        c = int(flat_idx // (height * width))
        rem = int(flat_idx % (height * width))
        y = int(rem // width)
        x = int(rem % width)

        if current_changed[c, y, x]:
            continue

        g = float(grad[c, y, x].item())
        if g == 0.0:
            continue

        direction = 1 if g > 0.0 else -1
        next_raw = float(clean[c, y, x].item()) + float(direction) * PIXEL_STEP_RAW

        # Reject clipping-invalid one-step changes.
        if next_raw < -1e-9 or next_raw > 1.0 + 1e-9:
            continue

        candidate_changes.append(PixelChange(channel=c, y=y, x=x, direction=direction))

    if not candidate_changes:
        return base_adv

    # Try increasing sparse prefixes, then binary search the first flipping prefix.
    best_adv = base_adv
    best_rmse = float("inf")

    def _build_adv(prefix_len: int) -> torch.Tensor:
        delta_try = base_delta.clone()
        for change in candidate_changes[:prefix_len]:
            delta_try[change.channel, change.y, change.x] = float(change.direction) * PIXEL_STEP_RAW
        return (clean + delta_try).clamp(0.0, 1.0)

    for prefix_size in SPARSE_FALLBACK_PREFIX_SIZES:
        if time.monotonic() >= deadline:
            break

        prefix_size = min(int(prefix_size), len(candidate_changes))
        if prefix_size <= 0:
            continue

        trial_adv = _build_adv(prefix_size)

        if not one_step_linf_ok(clean, trial_adv):
            continue

        if validator_passes_adv(
            model=model,
            clean=clean,
            adv=trial_adv,
            true_idx=true_idx,
            min_delta=min_delta,
            max_delta=max_delta,
        ):
            # Binary search minimal prefix that still flips.
            lo, hi = 1, prefix_size
            while lo < hi and time.monotonic() < deadline:
                mid = (lo + hi) // 2
                mid_adv = _build_adv(mid)

                if validator_passes_adv(
                    model=model,
                    clean=clean,
                    adv=mid_adv,
                    true_idx=true_idx,
                    min_delta=min_delta,
                    max_delta=max_delta,
                ):
                    hi = mid
                else:
                    lo = mid + 1

            final_adv = _build_adv(hi)
            stats = candidate_stats(clean, final_adv)
            if stats.rmse < best_rmse:
                best_adv = final_adv
                best_rmse = stats.rmse
            break

    # Commit only if fallback found a valid one-step flip.
    if validator_passes_adv(
        model=model,
        clean=clean,
        adv=best_adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        state.delta = (best_adv - clean).detach()
        state.changed_mask = state.delta.abs() > 1e-12
        refresh_state_logits(model=model, state=state, top_k=DEFAULT_TOP_K)
        return state.adv

    # Do NOT return dense/multi-step fallback.
    # Return previous sparse state, even if it did not flip.
    return base_adv


def run_feature_guided_attack(
    model: torch.nn.Module,
    clean: torch.Tensor,
    true_idx: int,
    epsilon: float,
    min_delta: float,
    timeout_seconds: float | int,
    *,
    hyperparams: AttackHyperparams | None = None,
    task_id: str = "unknown",
    true_label: str = "",
) -> FeatureGuidedAttackOutput:
    """
    Full feature-guided attack with metadata for miner finalization (steps 4-7).
    """
    hp = hyperparams or DEFAULT_ATTACK_HYPERPARAMS
    top_k = hp.top_k
    beam_width = hp.beam_width
    top_regions = hp.top_regions_per_competitor
    flip_margin_before_prune = hp.flip_margin_before_prune

    max_delta = _effective_max_delta(epsilon)
    min_delta = float(min_delta)
    budget = AttackTimeBudget.from_timeout(
        timeout_seconds,
        search_fraction=hp.search_time_fraction,
        prune_fraction=hp.prune_time_fraction,
        validate_fraction=hp.validate_time_fraction,
        buffer_fraction=hp.buffer_time_fraction,
    )

    effective_beam_width = beam_width
    if hp.shrink_beam_on_short_timeout and float(timeout_seconds) <= float(TIMEOUT_SECONDS):
        effective_beam_width = min(beam_width, ATTACK_PRESET_FAST.beam_width)

    session = AttackLogSession.create(
        task_id=task_id,
        true_idx=true_idx,
        true_label=true_label or idx_label(true_idx),
        min_delta=min_delta,
        epsilon=float(epsilon),
        effective_max_delta=max_delta,
        timeout_seconds=float(timeout_seconds),
        budget=budget,
    )

    initial_state = init_attack_state(model=model, clean=clean, true_idx=true_idx)
    start_snap = build_validator_snapshot(
        logits=initial_state.logits,
        true_idx=true_idx,
        clean=clean,
        adv=initial_state.adv,
        changed_mask=initial_state.changed_mask,
    )
    session.log_validator_snapshot(start_snap, prefix="[START]")

    pgd_used = False

    beam, best_flip = run_beam_search_phase(
        model=model,
        clean=clean,
        true_idx=true_idx,
        epsilon=epsilon,
        min_delta=min_delta,
        budget=budget,
        beam_width=effective_beam_width,
        top_k=top_k,
        top_regions=top_regions,
        max_rounds=DEFAULT_STEPS,
        region_grow_initial_batch=hp.region_grow_initial_batch,
        region_grow_max_pixels_per_region=hp.region_grow_max_pixels_per_region,
        log_session=session,
    )

    if time.monotonic() < budget.prune_end:
        beam = prune_beam_candidates(
            model=model,
            beam=beam,
            true_idx=true_idx,
            epsilon=epsilon,
            top_k=top_k,
            beam_width=effective_beam_width,
            min_delta=min_delta,
            max_delta=max_delta,
            budget=budget,
            top_regions=top_regions,
            region_grow_max_pixels_per_region=hp.region_grow_max_pixels_per_region,
            log_session=session,
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
        pre_prune_adv = adv.clone()
        accepted_groups: list[AcceptedGroup] = []
    else:
        pre_prune_adv = best_node.adv.clone()
        accepted_groups = list(best_node.state.accepted_groups)
        flip_stats = check_flip_and_gap(model=model, clean=clean, adv=pre_prune_adv, true_idx=true_idx)
        if _is_valid_flip_node(best_node, min_delta=min_delta, max_delta=max_delta) and _has_flip_margin(
            flip_stats,
            margin=flip_margin_before_prune,
        ):
            pruned_state = prune_validator_aware_state(
                model=model,
                state=clone_attack_state(best_node.state),
                true_idx=true_idx,
                min_delta=min_delta,
                max_delta=max_delta,
                top_k=top_k,
                deadline=budget.validate_end,
                log_session=session,
            )
            best_node = build_beam_node(pruned_state)
        adv = best_node.adv
        fallback_state = clone_attack_state(best_node.state)
        accepted_groups = list(best_node.state.accepted_groups)

    flip_stats = check_flip_and_gap(model=model, clean=clean, adv=adv, true_idx=true_idx)

    if not validator_passes_adv(
        model=model,
        clean=clean,
        adv=adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
    ):
        pgd_used = True
        adv = pgd_fallback_attack(
            model=model,
            state=fallback_state,
            true_idx=true_idx,
            max_delta=max_delta,
            min_delta=min_delta,
            deadline=budget.hard_end,
            steps=min(10, DEFAULT_STEPS),
        )
        flip_stats = check_flip_and_gap(model=model, clean=clean, adv=adv, true_idx=true_idx)

    post_ssim: float | None = None
    post_psnr: float | None = None
    if flip_stats.flipped:
        post_ssim = compute_ssim(clean, adv)
        post_psnr = compute_psnr_db(clean, adv)

    post_snap = build_validator_snapshot(
        logits=flip_stats.logits,
        true_idx=true_idx,
        clean=clean,
        adv=adv,
        changed_mask=(adv - clean).abs() > 1e-12,
        ssim=post_ssim,
        psnr_db=post_psnr,
    )
    session.log_validator_snapshot(post_snap, prefix="[POST_ATTACK]")
    if post_snap.flipped and not session.flip_logged:
        session.log_flip(
            pred_idx=post_snap.pred_idx,
            norm_linf=post_snap.norm_linf,
            rmse=post_snap.rmse,
            margin=post_snap.margin,
            changed_pixel_channels=post_snap.changed_pixel_channels,
            ssim=post_ssim,
            psnr_db=post_psnr,
        )
    session.log_accepted_groups(accepted_groups)
    session.pgd_used = pgd_used

    return FeatureGuidedAttackOutput(
        adv=adv.clamp(0.0, 1.0),
        pre_prune_adv=pre_prune_adv.clamp(0.0, 1.0),
        accepted_groups=accepted_groups,
        fallback_state=fallback_state,
        flip_stats=flip_stats,
        log_session=session,
    )


def finalize_miner_adversarial(
    model: torch.nn.Module,
    clean: torch.Tensor,
    attack_output: FeatureGuidedAttackOutput,
    true_idx: int,
    *,
    min_delta: float,
    max_delta: float,
    deadline: float | None = None,
    pgd_steps: int = 10,
    log_session: AttackLogSession | None = None,
) -> tuple[torch.Tensor, PngRoundtripResult, FlipGapStats]:
    """
    Miner finalization: PGD fallback if needed, then PNG roundtrip verification.
    """
    session = log_session or attack_output.log_session
    pgd_used = bool(session.pgd_used) if session is not None else False

    adv = attack_output.adv
    stats = attack_output.flip_stats or check_flip_and_gap(
        model=model,
        clean=clean,
        adv=adv,
        true_idx=true_idx,
    )

    if not stats.flipped and deadline is not None:
        pgd_used = True
        adv = pgd_fallback_attack(
            model=model,
            state=attack_output.fallback_state,
            true_idx=true_idx,
            max_delta=max_delta,
            min_delta=min_delta,
            deadline=deadline,
            steps=pgd_steps,
        )
        stats = check_flip_and_gap(model=model, clean=clean, adv=adv, true_idx=true_idx)
        if session is not None:
            session.pgd_used = True

    roundtrip = png_roundtrip_verify(
        model=model,
        clean=clean,
        adv=adv,
        true_idx=true_idx,
        min_delta=min_delta,
        max_delta=max_delta,
        accepted_groups=attack_output.accepted_groups,
        restore_adv_candidates=[attack_output.pre_prune_adv],
    )

    if roundtrip.passed:
        prune_deadline = deadline if deadline is not None else None
        pruned_decoded = roundtrip_pixel_prune(
            model=model,
            clean=clean,
            adv=roundtrip.final_adv,
            true_idx=true_idx,
            min_delta=min_delta,
            max_delta=max_delta,
            deadline=prune_deadline,
            max_checks=int(os.getenv("PERTURB_FINAL_PRUNE_MAX_CHECKS", "512")),
        )

        pruned_roundtrip = png_roundtrip_verify(
            model=model,
            clean=clean,
            adv=pruned_decoded,
            true_idx=true_idx,
            min_delta=min_delta,
            max_delta=max_delta,
            accepted_groups=None,
            restore_adv_candidates=[roundtrip.final_adv],
        )

        if pruned_roundtrip.passed and pruned_roundtrip.rmse <= roundtrip.rmse + 1e-12:
            roundtrip = pruned_roundtrip

    final_stats = check_flip_and_gap(
        model=model,
        clean=clean,
        adv=roundtrip.final_adv,
        true_idx=true_idx,
    )

    if session is not None:
        rt_ssim: float | None = None
        rt_psnr: float | None = None
        if final_stats.flipped:
            rt_ssim = compute_ssim(clean, roundtrip.final_adv)
            rt_psnr = compute_psnr_db(clean, roundtrip.final_adv)
        rt_margin = -float(final_stats.untargeted_gap)
        session.log_roundtrip(
            pred_idx=roundtrip.pred_idx,
            flipped=roundtrip.pred_idx != true_idx,
            norm_linf=roundtrip.norm,
            rmse=roundtrip.rmse,
            margin=rt_margin,
            passed=roundtrip.passed,
            restored_from_backup=roundtrip.restored_from_backup,
            reason=roundtrip.reason,
            ssim=rt_ssim,
            psnr_db=rt_psnr,
        )
        final_snap = build_validator_snapshot(
            logits=final_stats.logits,
            true_idx=true_idx,
            clean=clean,
            adv=roundtrip.final_adv,
            changed_mask=(roundtrip.final_adv - clean).abs() > 1e-12,
            ssim=rt_ssim,
            psnr_db=rt_psnr,
        )
        session.log_final_validation(final_snap, pgd_used=pgd_used)

    return roundtrip.final_adv, roundtrip, final_stats
