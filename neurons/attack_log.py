"""Structured attack metrics logging for miner status inspection."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from neurons.attack_excel import append_summary_row, sanitize_filename, write_attack_workbook
from perturbnet.model import LABELS

if TYPE_CHECKING:
    from neurons.attack import (
        AcceptedGroup,
        AttackTimeBudget,
        CompetitorEntry,
        RankedRegion,
        TopKCompetitorRace,
        TrialVerification,
    )

logger = logging.getLogger(__name__)


def idx_label(idx: int) -> str:
    if 0 <= int(idx) < len(LABELS):
        return LABELS[int(idx)]
    return str(idx)


def count_pixel_changes(
    delta: torch.Tensor,
    changed_mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Return (changed_pixel_channels, changed_rgb_pixels)."""
    if changed_mask is not None:
        channel_count = int(changed_mask.sum().item())
        rgb_pixels = int(changed_mask.any(dim=0).sum().item())
        return channel_count, rgb_pixels
    nonzero = delta.abs() > 1e-12
    return int(nonzero.sum().item()), int(nonzero.any(dim=0).sum().item())


@dataclass
class ValidatorPassSnapshot:
    true_idx: int
    pred_idx: int
    best_other_idx: int
    true_logit: float
    best_other_logit: float
    untargeted_gap: float
    margin: float
    flipped: bool
    norm_linf: float
    rmse: float
    changed_pixel_channels: int
    changed_rgb_pixels: int
    ssim: float | None = None
    psnr_db: float | None = None

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "true_idx": self.true_idx,
            "pred_idx": self.pred_idx,
            "true_label": idx_label(self.true_idx),
            "pred_label": idx_label(self.pred_idx),
            "best_other_idx": self.best_other_idx,
            "best_other_label": idx_label(self.best_other_idx),
            "true_logit": round(self.true_logit, 4),
            "best_other_logit": round(self.best_other_logit, 4),
            "untargeted_gap": round(self.untargeted_gap, 4),
            "margin": round(self.margin, 4),
            "flipped": self.flipped,
            "norm_linf": round(self.norm_linf, 6),
            "rmse": round(self.rmse, 6),
            "changed_pixel_channels": self.changed_pixel_channels,
            "changed_rgb_pixels": self.changed_rgb_pixels,
            "ssim": None if self.ssim is None else round(self.ssim, 4),
            "psnr_db": None if self.psnr_db is None else round(self.psnr_db, 2),
        }


@dataclass
class AttackLogSession:
    """Per-task attack logging context."""

    task_id: str = "unknown"
    true_idx: int = -1
    true_label: str = ""
    min_delta: float = 0.003
    epsilon: float = 0.03
    effective_max_delta: float = 0.03
    start_time: float = field(default_factory=time.monotonic)
    deadline: float = 0.0
    budget_search_end: float = 0.0
    budget_prune_end: float = 0.0
    budget_validate_end: float = 0.0
    beam_id: int = 0
    round_id: int = 0
    last_competitor_idx: int | None = None
    accepted_group_count: int = 0
    flip_logged: bool = False
    pgd_used: bool = False
    prune_before_pixels: int = 0
    prune_before_rmse: float = 0.0
    prune_before_margin: float = 0.0
    excel_enabled: bool = True
    excel_rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    excel_export_path: Path | None = None
    excel_summary_path: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        true_idx: int,
        true_label: str,
        min_delta: float,
        epsilon: float,
        effective_max_delta: float,
        timeout_seconds: float,
        budget: AttackTimeBudget | None = None,
        excel_enabled: bool | None = None,
    ) -> AttackLogSession:
        if excel_enabled is None:
            excel_enabled = os.getenv("ATTACK_LOG_EXCEL", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        start = time.monotonic()
        session = cls(
            task_id=task_id,
            true_idx=true_idx,
            true_label=true_label,
            min_delta=float(min_delta),
            epsilon=float(epsilon),
            effective_max_delta=float(effective_max_delta),
            start_time=start,
            deadline=start + float(timeout_seconds),
            excel_enabled=bool(excel_enabled),
        )
        if budget is not None:
            session.budget_search_end = budget.search_end
            session.budget_prune_end = budget.prune_end
            session.budget_validate_end = budget.validate_end
        return session

    def _record(self, sheet: str, row: dict[str, Any], *, event_type: str | None = None) -> None:
        if not self.excel_enabled:
            return
        payload = dict(row)
        payload.setdefault("task_id", self.task_id)
        payload.setdefault("recorded_at_ms", self.elapsed_ms())
        self.excel_rows.setdefault(sheet, []).append(payload)
        if event_type is not None:
            event_row = {
                "event_type": event_type,
                "task_id": self.task_id,
                "recorded_at_ms": payload["recorded_at_ms"],
                "round_id": payload.get("round_id", self.round_id),
                "beam_id": payload.get("beam_id", self.beam_id),
                "competitor_idx": payload.get("competitor_idx"),
                "gap_before": payload.get("gap_before"),
                "gap_after": payload.get("gap_after"),
                "real_gain": payload.get("real_gain"),
                "gain_per_pixel": payload.get("gain_per_pixel"),
                "accepted": payload.get("accepted"),
                "flipped": payload.get("flipped"),
                "norm_linf": payload.get("norm_linf"),
                "rmse": payload.get("rmse"),
                "margin": payload.get("margin"),
                "pred_idx": payload.get("pred_idx"),
                "elapsed_ms": payload.get("elapsed_ms", self.elapsed_ms()),
            }
            self.excel_rows.setdefault("all_events", []).append(event_row)

    def _timing_payload(self) -> dict[str, int]:
        return self.timing_fields()

    def export_excel(
        self,
        output_dir: str | Path,
        *,
        extra_meta: dict[str, Any] | None = None,
        append_master_summary: bool = True,
    ) -> Path | None:
        if not self.excel_enabled or not self.excel_rows:
            return None

        output_root = Path(output_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_task_id = sanitize_filename(self.task_id)
        output_path = output_root / f"{safe_task_id}_{timestamp}.xlsx"

        meta_row = {
            "task_id": self.task_id,
            "true_idx": self.true_idx,
            "true_label": self.true_label or idx_label(self.true_idx),
            "min_delta": round(self.min_delta, 6),
            "epsilon": round(self.epsilon, 6),
            "effective_max_delta": round(self.effective_max_delta, 6),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            meta_row.update(extra_meta)

        sheets = dict(self.excel_rows)
        sheets["task_meta"] = [meta_row]

        self.excel_export_path = write_attack_workbook(output_path=output_path, sheets=sheets)

        if append_master_summary:
            summary_candidates = sheets.get("validator_snapshots", [])
            final_rows = [row for row in summary_candidates if row.get("phase") == "FINAL"]
            summary_row = final_rows[-1] if final_rows else None
            if summary_row is None:
                post_rows = [row for row in summary_candidates if row.get("phase") == "POST_ATTACK"]
                summary_row = post_rows[-1] if post_rows else None
            if summary_row is not None:
                summary_path = output_root / "all_tasks_summary.xlsx"
                self.excel_summary_path = append_summary_row(summary_path, summary_row)

        logger.info(
            "[EXCEL] task=%s export_path=%s summary_path=%s sheets=%s rows=%s",
            self.task_id,
            self.excel_export_path,
            self.excel_summary_path,
            len(sheets),
            sum(len(rows) for rows in sheets.values()),
        )
        return self.excel_export_path

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.start_time) * 1000)

    def remaining_ms(self) -> int:
        return max(0, int((self.deadline - time.monotonic()) * 1000))

    def search_budget_ms(self) -> int:
        return max(0, int((self.budget_search_end - self.start_time) * 1000))

    def prune_budget_ms(self) -> int:
        if self.budget_search_end <= 0.0:
            return 0
        return max(0, int((self.budget_prune_end - self.budget_search_end) * 1000))

    def timing_fields(self) -> dict[str, int]:
        return {
            "elapsed_ms": self.elapsed_ms(),
            "remaining_ms": self.remaining_ms(),
            "search_budget_ms": self.search_budget_ms(),
            "prune_budget_ms": self.prune_budget_ms(),
        }

    def log_validator_snapshot(self, snap: ValidatorPassSnapshot, *, prefix: str = "[STATUS]") -> None:
        fields = snap.as_log_fields()
        fields.update(self._timing_payload())
        fields["task"] = self.task_id
        phase = prefix.strip("[]")
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("%s %s", prefix, rendered)
        row = dict(fields)
        row["phase"] = phase
        row["response_time_ms"] = self.elapsed_ms()
        self._record("validator_snapshots", row, event_type=phase)

    def log_topk_competitors(
        self,
        race: TopKCompetitorRace,
        *,
        round_id: int | None = None,
    ) -> None:
        if round_id is not None:
            self.round_id = int(round_id)
        entries: list[str] = []
        for rank, entry in enumerate(race.competitors):
            entries.append(
                f"(idx={entry.idx},label={idx_label(entry.idx)},gap={entry.gap_k:.4f},rank={rank})"
            )
        competitor_switched = (
            self.last_competitor_idx is not None
            and race.competitors
            and race.competitors[0].idx != self.last_competitor_idx
        )
        old_comp = self.last_competitor_idx
        if race.competitors:
            self.last_competitor_idx = race.competitors[0].idx
        timing = self._timing_payload()
        logger.info(
            "[TOPK] task=%s round=%s true_idx=%s true_logit=%.4f topK=[%s] "
            "competitor_switched=%s old=%s new=%s %s",
            self.task_id,
            self.round_id,
            race.true_idx,
            race.true_logit,
            ", ".join(entries),
            competitor_switched,
            old_comp,
            self.last_competitor_idx,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )
        self._record(
            "topk_rounds",
            {
                "round_id": self.round_id,
                "true_idx": race.true_idx,
                "true_logit": round(race.true_logit, 4),
                "competitor_switched": competitor_switched,
                "old_competitor_idx": old_comp,
                "new_competitor_idx": self.last_competitor_idx,
                "topk_count": len(race.competitors),
                **timing,
            },
            event_type="TOPK",
        )
        for rank, entry in enumerate(race.competitors):
            self._record(
                "topk_competitors",
                {
                    "round_id": self.round_id,
                    "rank": rank,
                    "competitor_idx": entry.idx,
                    "competitor_label": idx_label(entry.idx),
                    "gap_k": round(entry.gap_k, 4),
                    "logit": round(entry.logit, 4),
                    **timing,
                },
                event_type="TOPK_COMPETITOR",
            )

    def log_beam_round(self, beam_nodes: list[Any], *, round_id: int) -> None:
        self.round_id = int(round_id)
        if not beam_nodes:
            return
        timing = self._timing_payload()
        lines: list[str] = []
        for beam_rank, node in enumerate(beam_nodes):
            lines.append(
                f"beam={beam_rank} gap={node.untargeted_gap:.4f} "
                f"pixels={node.changed_pixels} rmse={node.rmse:.6f} "
                f"gpp={node.recent_gain_per_pixel:.6f} flip={node.flipped} linf={node.norm:.6f}"
            )
            self._record(
                "beam_paths",
                {
                    "round_id": self.round_id,
                    "beam_rank": beam_rank,
                    "beam_id": beam_rank,
                    "untargeted_gap": round(node.untargeted_gap, 4),
                    "changed_pixels": node.changed_pixels,
                    "rmse": round(node.rmse, 6),
                    "gain_per_pixel": round(node.recent_gain_per_pixel, 6),
                    "flipped": node.flipped,
                    "norm_linf": round(node.norm, 6),
                    **timing,
                },
                event_type="BEAM",
            )
        logger.info(
            "[BEAM] task=%s round=%s paths=%s %s",
            self.task_id,
            self.round_id,
            len(beam_nodes),
            " | ".join(lines),
        )

    def log_attack_batch(
        self,
        *,
        verification: TrialVerification | None,
        region: RankedRegion,
        competitor_idx: int,
        batch_size: int,
        accepted: bool,
        reason: str,
        beam_id: int | None = None,
        round_id: int | None = None,
        gap_before: float | None = None,
    ) -> None:
        if round_id is not None:
            self.round_id = int(round_id)
        if beam_id is not None:
            self.beam_id = int(beam_id)

        y1, y2, x1, x2 = region.image_box
        yf, xf = region.feature_cell
        gap_after = verification.gap_after if verification else gap_before
        gap_before_val = verification.gap_before if verification is not None else (gap_before or 0.0)
        real_gain = verification.real_gain if verification else 0.0
        gpp = verification.gain_per_pixel if verification else 0.0
        pred_idx = verification.pred_idx if verification else self.true_idx
        norm = verification.norm if verification else 0.0
        rmse = verification.rmse if verification else 0.0
        num_pixels = verification.num_new_pixels if verification else batch_size
        best_other = pred_idx if pred_idx != self.true_idx else competitor_idx
        margin = -float(gap_after) if gap_after is not None else 0.0
        flipped = pred_idx != self.true_idx
        timing = self._timing_payload()

        logger.info(
            "[ATTACK] task=%s round=%s beam=%s comp=%s label=%s "
            "gap=%.4f->%.4f gain=%.4f gpp=%.6f region=(%s,%s) box=%s:%s,%s:%s "
            "cam=%.4f act=%.4f pgd=%.4f region_score=%.4f batch=%s accepted=%s "
            "reason=%s pixels=%s linf=%.6f rmse=%.6f pred=%s best_other=%s margin=%.4f flip=%s %s",
            self.task_id,
            self.round_id,
            self.beam_id,
            competitor_idx,
            idx_label(competitor_idx),
            gap_before_val,
            float(gap_after or 0.0),
            real_gain,
            gpp,
            yf,
            xf,
            y1,
            y2,
            x1,
            x2,
            region.gap_cam_term,
            region.abs_map_term,
            region.pixel_grad_density,
            region.region_score,
            batch_size,
            accepted,
            reason,
            num_pixels,
            norm,
            rmse,
            pred_idx,
            best_other,
            margin,
            flipped,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )
        self._record(
            "attack_batches",
            {
                "round_id": self.round_id,
                "beam_id": self.beam_id,
                "competitor_idx": competitor_idx,
                "competitor_label": idx_label(competitor_idx),
                "gap_before": round(gap_before_val, 4),
                "gap_after": round(float(gap_after or 0.0), 4),
                "real_gain": round(real_gain, 4),
                "gain_per_pixel": round(gpp, 6),
                "region_yf": yf,
                "region_xf": xf,
                "box_y1": y1,
                "box_y2": y2,
                "box_x1": x1,
                "box_x2": x2,
                "cam_score": round(region.gap_cam_term, 4),
                "activation_grad_score": round(region.abs_map_term, 4),
                "pixel_grad_density": round(region.pixel_grad_density, 4),
                "region_score": round(region.region_score, 4),
                "batch_size": batch_size,
                "accepted": accepted,
                "reason": reason,
                "pixels": num_pixels,
                "norm_linf": round(norm, 6),
                "rmse": round(rmse, 6),
                "pred_idx": pred_idx,
                "best_other_idx": best_other,
                "margin": round(margin, 4),
                "flipped": flipped,
                **timing,
            },
            event_type="ATTACK",
        )

        if accepted:
            self.accepted_group_count += 1
            if flipped and not self.flip_logged:
                self.log_flip(
                    pred_idx=pred_idx,
                    norm_linf=norm,
                    rmse=rmse,
                    margin=margin,
                    changed_pixel_channels=num_pixels,
                    ssim=verification.ssim if verification else None,
                    psnr_db=verification.psnr_db if verification else None,
                )

    def log_region_saturation(
        self,
        *,
        region: RankedRegion,
        competitor_idx: int,
        fail_count: int,
        weak_count: int,
        pixels_applied: int,
        stopped_reason: str,
        round_id: int | None = None,
    ) -> None:
        if round_id is not None:
            self.round_id = int(round_id)
        yf, xf = region.feature_cell
        saturated = fail_count >= 2 or stopped_reason.startswith("failures")
        timing = self._timing_payload()
        logger.info(
            "[REGION] task=%s round=%s comp=%s region=(%s,%s) pixels_applied=%s "
            "region_fail_count=%s region_weak_count=%s saturated=%s reason=%s %s",
            self.task_id,
            self.round_id,
            competitor_idx,
            yf,
            xf,
            pixels_applied,
            fail_count,
            weak_count,
            saturated,
            stopped_reason,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )
        self._record(
            "regions",
            {
                "round_id": self.round_id,
                "competitor_idx": competitor_idx,
                "region_yf": yf,
                "region_xf": xf,
                "pixels_applied": pixels_applied,
                "region_fail_count": fail_count,
                "region_weak_count": weak_count,
                "saturated": saturated,
                "stopped_reason": stopped_reason,
                **timing,
            },
            event_type="REGION",
        )

    def log_accepted_groups(self, groups: list[AcceptedGroup], *, replace: bool = True) -> None:
        if replace:
            self.excel_rows["accepted_groups"] = []
        for group_id, group in enumerate(groups):
            yf, xf = group.feature_cell
            logger.info(
                "[GROUP] task=%s group_id=%s comp=%s region=(%s,%s) pixels=%s "
                "gain=%.4f gpp=%.6f gap=%.4f->%.4f round=%s",
                self.task_id,
                group_id,
                group.competitor_idx,
                yf,
                xf,
                len(group.pixels),
                group.gain,
                group.gain_per_pixel,
                group.gap_before,
                group.gap_after,
                group.round_id,
            )
            self._record(
                "accepted_groups",
                {
                    "group_id": group_id,
                    "competitor_idx": group.competitor_idx,
                    "region_yf": yf,
                    "region_xf": xf,
                    "pixels": len(group.pixels),
                    "gain": round(group.gain, 4),
                    "gain_per_pixel": round(group.gain_per_pixel, 6),
                    "gap_before": round(group.gap_before, 4),
                    "gap_after": round(group.gap_after, 4),
                    "round_id": group.round_id,
                },
                event_type="GROUP",
            )

    def log_prune_start(self, snap: ValidatorPassSnapshot) -> None:
        self.prune_before_pixels = snap.changed_pixel_channels
        self.prune_before_rmse = snap.rmse
        self.prune_before_margin = snap.margin

    def log_prune_result(
        self,
        before: ValidatorPassSnapshot,
        after: ValidatorPassSnapshot,
        *,
        roundtrip_ok: bool | None = None,
    ) -> None:
        removed = before.changed_pixel_channels - after.changed_pixel_channels
        timing = self._timing_payload()
        logger.info(
            "[PRUNE] task=%s before_pixels=%s after_pixels=%s removed=%s "
            "before_rmse=%.6f after_rmse=%.6f margin=%.4f pred=%s flipped=%s "
            "roundtrip=%s accepted_groups=%s %s",
            self.task_id,
            before.changed_pixel_channels,
            after.changed_pixel_channels,
            removed,
            before.rmse,
            after.rmse,
            after.margin,
            after.pred_idx,
            after.flipped,
            roundtrip_ok,
            self.accepted_group_count,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )
        self._record(
            "prune_events",
            {
                "before_pixels": before.changed_pixel_channels,
                "after_pixels": after.changed_pixel_channels,
                "removed_pixels": removed,
                "before_rmse": round(before.rmse, 6),
                "after_rmse": round(after.rmse, 6),
                "margin": round(after.margin, 4),
                "pred_idx": after.pred_idx,
                "flipped": after.flipped,
                "roundtrip_ok": roundtrip_ok,
                "accepted_groups": self.accepted_group_count,
                **timing,
            },
            event_type="PRUNE",
        )

    def log_flip(
        self,
        *,
        pred_idx: int,
        norm_linf: float,
        rmse: float,
        margin: float,
        changed_pixel_channels: int,
        ssim: float | None = None,
        psnr_db: float | None = None,
    ) -> None:
        self.flip_logged = True
        timing = self._timing_payload()
        logger.info(
            "[FLIP] task=%s pred=%s label=%s true=%s true_label=%s margin=%.4f "
            "linf=%.6f rmse=%.6f ssim=%s psnr=%s pixels=%s %s",
            self.task_id,
            pred_idx,
            idx_label(pred_idx),
            self.true_idx,
            self.true_label or idx_label(self.true_idx),
            margin,
            norm_linf,
            rmse,
            None if ssim is None else round(ssim, 4),
            None if psnr_db is None else round(psnr_db, 1),
            changed_pixel_channels,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )
        self._record(
            "flip_events",
            {
                "pred_idx": pred_idx,
                "pred_label": idx_label(pred_idx),
                "true_idx": self.true_idx,
                "true_label": self.true_label or idx_label(self.true_idx),
                "margin": round(margin, 4),
                "norm_linf": round(norm_linf, 6),
                "rmse": round(rmse, 6),
                "ssim": None if ssim is None else round(ssim, 4),
                "psnr_db": None if psnr_db is None else round(psnr_db, 1),
                "changed_pixel_channels": changed_pixel_channels,
                **timing,
            },
            event_type="FLIP",
        )

    def log_roundtrip(
        self,
        *,
        pred_idx: int,
        flipped: bool,
        norm_linf: float,
        rmse: float,
        margin: float,
        passed: bool,
        restored_from_backup: bool,
        reason: str,
        ssim: float | None = None,
        psnr_db: float | None = None,
    ) -> None:
        timing = self._timing_payload()
        logger.info(
            "[ROUNDTRIP] task=%s roundtrip_pred=%s roundtrip_label=%s "
            "roundtrip_flipped=%s roundtrip_norm=%.6f roundtrip_rmse=%.6f "
            "roundtrip_margin=%.4f roundtrip_ok=%s restored=%s reason=%s "
            "ssim=%s psnr=%s min_delta=%.6f effective_max=%.6f %s",
            self.task_id,
            pred_idx,
            idx_label(pred_idx),
            flipped,
            norm_linf,
            rmse,
            margin,
            passed,
            restored_from_backup,
            reason,
            None if ssim is None else round(ssim, 4),
            None if psnr_db is None else round(psnr_db, 1),
            self.min_delta,
            self.effective_max_delta,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )
        self._record(
            "roundtrip",
            {
                "roundtrip_pred": pred_idx,
                "roundtrip_label": idx_label(pred_idx),
                "roundtrip_flipped": flipped,
                "roundtrip_norm": round(norm_linf, 6),
                "roundtrip_rmse": round(rmse, 6),
                "roundtrip_margin": round(margin, 4),
                "roundtrip_ok": passed,
                "restored_from_backup": restored_from_backup,
                "reason": reason,
                "ssim": None if ssim is None else round(ssim, 4),
                "psnr_db": None if psnr_db is None else round(psnr_db, 1),
                "min_delta": round(self.min_delta, 6),
                "effective_max_delta": round(self.effective_max_delta, 6),
                **timing,
            },
            event_type="ROUNDTRIP",
        )

    def log_final_validation(self, snap: ValidatorPassSnapshot, *, pgd_used: bool = False) -> None:
        snap_dict = snap.as_log_fields()
        snap_dict.update(self._timing_payload())
        snap_dict["task"] = self.task_id
        snap_dict["response_time_ms"] = self.elapsed_ms()
        snap_dict["pgd_fallback"] = pgd_used
        snap_dict["accepted_groups"] = self.accepted_group_count
        snap_dict["min_delta"] = round(self.min_delta, 6)
        snap_dict["epsilon"] = round(self.epsilon, 6)
        snap_dict["effective_max_delta"] = round(self.effective_max_delta, 6)
        rendered = " ".join(f"{key}={value}" for key, value in snap_dict.items())
        logger.info("[FINAL] %s", rendered)
        row = dict(snap_dict)
        row["phase"] = "FINAL"
        self._record("validator_snapshots", row, event_type="FINAL")


def build_validator_snapshot(
    *,
    logits: torch.Tensor,
    true_idx: int,
    clean: torch.Tensor,
    adv: torch.Tensor,
    changed_mask: torch.Tensor | None = None,
    ssim: float | None = None,
    psnr_db: float | None = None,
) -> ValidatorPassSnapshot:
    if logits.ndim == 2:
        row = logits[0]
    else:
        row = logits
    pred_idx = int(row.argmax().item())
    true_logit = float(row[true_idx].item())
    masked = row.clone()
    masked[true_idx] = float("-inf")
    best_other_idx = int(masked.argmax().item())
    best_other_logit = float(masked.max().item())
    untargeted_gap = true_logit - best_other_logit
    margin = best_other_logit - true_logit
    norm_linf = float((adv - clean).abs().max().item())
    rmse = float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())
    channels, rgb = count_pixel_changes(adv - clean, changed_mask)
    return ValidatorPassSnapshot(
        true_idx=true_idx,
        pred_idx=pred_idx,
        best_other_idx=best_other_idx,
        true_logit=true_logit,
        best_other_logit=best_other_logit,
        untargeted_gap=untargeted_gap,
        margin=margin,
        flipped=pred_idx != true_idx,
        norm_linf=norm_linf,
        rmse=rmse,
        changed_pixel_channels=channels,
        changed_rgb_pixels=rgb,
        ssim=ssim,
        psnr_db=psnr_db,
    )
