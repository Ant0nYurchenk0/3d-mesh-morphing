"""
Results reporter — writes summary.csv, summary_per_model.csv, and summary.md
for a session.

summary.csv columns:
    shape, model, chamfer_distance, f_score, hausdorff_distance,
    volume_iou, normal_consistency, morphing_readiness_score,
    dino_similarity, cleanup_cost, error

summary_per_model.csv columns:
    model, n_shapes, mean_<metric> for each numeric metric

summary.md:
    Two tables: per-shape detail and per-model mean aggregation.

Usage:
    Reporter(cfg).write(session, results)

    results: dict mapping (shape_name, model_name) → metrics dict
             (as returned by MetricsCalculator.compute_all or null_metrics())
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .session import Session

log = logging.getLogger(__name__)

_METRIC_COLS = [
    "chamfer_distance",
    "f_score",
    "hausdorff_distance",
    "volume_iou",
    "normal_consistency",
    "morphing_readiness_score",
    "dino_similarity",
    "cleanup_cost",
]

_PER_MODEL_CSV_COLS = ["model", "n_shapes"] + [f"mean_{c}" for c in _METRIC_COLS]

_CSV_COLS = ["shape", "model", *_METRIC_COLS, "error"]


def null_metrics(error_msg: str) -> dict:
    """Return a metrics dict with all values null and an error message."""
    return {k: None for k in _METRIC_COLS} | {"error": error_msg}


class Reporter:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        # Load thresholds for annotation
        mc = _get(cfg, "metrics", {})
        self._thresholds = _get(mc, "thresholds", {})

    def write(self, session: Session, results: dict[tuple[str, str], dict]) -> None:
        """
        Write summary.csv, summary_per_model.csv, and summary.md for *session*.

        results keys: (shape_name, model_name)
        results values: metrics dict (from compute_all or null_metrics)
        """
        rows = self._build_rows(results)
        model_rows = self._build_model_summary_rows(rows)
        self._write_csv(session.summary_csv_path, rows)
        self._write_model_summary_csv(session.summary_per_model_csv_path, model_rows)
        self._write_md(session.summary_md_path, rows, model_rows)
        log.info("[reporter] wrote %s", session.summary_csv_path)
        log.info("[reporter] wrote %s", session.summary_per_model_csv_path)
        log.info("[reporter] wrote %s", session.summary_md_path)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _write_md(self, path: Path, rows: list[dict], model_rows: list[dict]) -> None:
        lines = [
            "# Image-to-3D Evaluation — Session Summary",
            "",
            "## Per-shape results",
            "",
            "| Shape | Model | CD ↓ | F-Score ↑ | Hausdorff ↓ | Vol IoU ↑ | Normal Cons ↑ | MRS ↑ | DINO ↑ | Cost ↓ | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]

        for r in rows:
            err = r.get("error", "") or ""
            if err:
                status = f"❌ {err[:60]}"
                cd = f_sc = hd = viou = nc = mrs = dino = cost = "N/A"
            else:
                cd = _fmt(r.get("chamfer_distance"))
                f_sc = _fmt(r.get("f_score"))
                hd = _fmt(r.get("hausdorff_distance"))
                viou = _fmt(r.get("volume_iou"), na_str="—")
                nc = _fmt(r.get("normal_consistency"))
                mrs = _fmt(r.get("morphing_readiness_score"))
                dino = _fmt(r.get("dino_similarity"), na_str="—")
                raw_cost = r.get("cleanup_cost")
                cost = str(raw_cost) if raw_cost is not None and raw_cost != "" else "—"
                status = self._quality_label(r)

            lines.append(
                f"| {r['shape']} | {r['model']} "
                f"| {cd} | {f_sc} | {hd} | {viou} | {nc} | {mrs} "
                f"| {dino} | {cost} | {status} |"
            )

        lines += [
            "",
            "**Thresholds (unit-sphere normalised):**",
            "- CD: excellent < 0.005, good < 0.010, poor > 0.030",
            "- F-Score: excellent > 0.90, good > 0.85, poor < 0.70",
            "- MRS: excellent > 0.80, good > 0.65, poor < 0.40",
            "",
            "*Volume IoU shown as `—` when either mesh is not watertight.*",
            "*DINO shown as `—` when torch/transformers not installed (`uv sync --extra dino`).*",
            "",
            "## Per-model mean (successful shapes only)",
            "",
            "| Model | N | CD ↓ | F-Score ↑ | Hausdorff ↓ | Vol IoU ↑ | Normal Cons ↑ | MRS ↑ | DINO ↑ | Cost ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for r in model_rows:
            lines.append(
                f"| {r['model']} | {r['n_shapes']} "
                f"| {_fmt(r.get('mean_chamfer_distance'))} "
                f"| {_fmt(r.get('mean_f_score'))} "
                f"| {_fmt(r.get('mean_hausdorff_distance'))} "
                f"| {_fmt(r.get('mean_volume_iou'), na_str='—')} "
                f"| {_fmt(r.get('mean_normal_consistency'))} "
                f"| {_fmt(r.get('mean_morphing_readiness_score'))} "
                f"| {_fmt(r.get('mean_dino_similarity'), na_str='—')} "
                f"| {_fmt(r.get('mean_cleanup_cost'), digits=1)} |"
            )

        lines += [
            "",
            "*Means computed only over shapes where the model succeeded (no error).*",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_rows(self, results: dict) -> list[dict]:
        rows = []
        for (shape, model), metrics in sorted(results.items()):
            row: dict = {"shape": shape, "model": model}
            for col in _METRIC_COLS:
                v = metrics.get(col)
                if col == "cleanup_cost":
                    row[col] = "" if v is None else str(int(v))
                elif isinstance(v, float):
                    row[col] = f"{v:.8f}"
                else:
                    row[col] = "" if v is None else str(v)
            row["error"] = metrics.get("error", "") or ""
            rows.append(row)
        return rows

    def _write_model_summary_csv(self, path: Path, model_rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_PER_MODEL_CSV_COLS)
            writer.writeheader()
            writer.writerows(model_rows)

    def _build_model_summary_rows(self, rows: list[dict]) -> list[dict]:
        """
        Aggregate per-shape rows into per-model means.

        Only rows without an error value contribute to the mean.
        Metrics that are None/empty for a given row (e.g. volume_iou when
        not watertight, dino_similarity when torch absent) are excluded from
        that metric's mean but do not disqualify the whole row.
        """
        # Accumulate numeric values per model
        buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in rows:
            if r.get("error"):
                continue
            model = r["model"]
            for col in _METRIC_COLS:
                raw = r.get(col)
                if raw is None or raw == "":
                    continue
                try:
                    buckets[model][col].append(float(raw))
                except (ValueError, TypeError):
                    pass

        model_rows = []
        for model in sorted(buckets.keys()):
            vals = buckets[model]
            # n_shapes = number of successful rows for this model
            # Use chamfer_distance as the canonical present-check
            n = len(vals.get("chamfer_distance", []))
            row: dict = {"model": model, "n_shapes": n}
            for col in _METRIC_COLS:
                col_vals = vals.get(col, [])
                mean_key = f"mean_{col}"
                if col_vals:
                    mean = sum(col_vals) / len(col_vals)
                    row[mean_key] = f"{mean:.8f}" if col != "cleanup_cost" else f"{mean:.1f}"
                else:
                    row[mean_key] = ""
            model_rows.append(row)

        return model_rows

    def _quality_label(self, row: dict) -> str:
        """Return a short quality annotation based on CD and F-Score thresholds."""
        try:
            cd = float(row.get("chamfer_distance") or "inf")
            fs = float(row.get("f_score") or "0")
        except (ValueError, TypeError):
            return "?"

        cd_thr = self._thresholds.get("chamfer_distance", {})
        fs_thr = self._thresholds.get("f_score", {})

        cd_exc = float(cd_thr.get("excellent", 0.005))
        cd_good = float(cd_thr.get("good", 0.010))
        cd_poor = float(cd_thr.get("poor", 0.030))

        fs_exc = float(fs_thr.get("excellent", 0.90))
        fs_good = float(fs_thr.get("good", 0.85))

        if cd < cd_exc and fs > fs_exc:
            return "excellent"
        if cd < cd_good and fs > fs_good:
            return "good"
        if cd > cd_poor or fs < float(fs_thr.get("poor", 0.70)):
            return "poor"
        return "ok"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _fmt(v: Any, digits: int = 5, na_str: str = "N/A") -> str:
    if v is None:
        return na_str
    try:
        return f"{float(v):.{digits}f}"
    except (ValueError, TypeError):
        return na_str


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
