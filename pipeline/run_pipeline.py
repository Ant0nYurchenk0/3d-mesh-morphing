#!/usr/bin/env python3
"""
Image-to-3D Evaluation Pipeline — entry point.

Usage:
    # Full run (all shapes × all models)
    uv run python run_pipeline.py

    # Subset run
    uv run python run_pipeline.py --shapes gear,sphere --models trellis,hunyuan3d

    # Dry run: download + render only (no model calls)
    uv run python run_pipeline.py --dry-run --shapes gear

    # Custom config / session dir
    uv run python run_pipeline.py --config my_config.yaml --session-dir /tmp/sessions

    # List available shapes and models (no run)
    uv run python run_pipeline.py --list

Session artifacts are written to:
    sessions/YYYY-MM-DD_HHMMSS_{uid}/
        {shape}_{model}/
            original.{ext}        ← GT mesh (normalised)
            render.png             ← 1024×1024 canonical render
            reconstructed.glb      ← model output
            metrics.json           ← 6 metrics + metadata
        summary.csv
        summary.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging setup — do this before any module-level log calls
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_PIPELINE_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _PIPELINE_DIR / "config.yaml"
_DEFAULT_SESSIONS = _PIPELINE_DIR / "sessions"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image-to-3D evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--shapes",
        default=None,
        help="Comma-separated list of shape names to test (default: all in config)",
    )
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names to test (default: all in config)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Download + render only; skip model calls and metrics",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {_DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help=f"Session output root (default: {_DEFAULT_SESSIONS})",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List available shapes and models then exit",
    )
    p.add_argument(
        "--recompute-metrics",
        metavar="SESSION_DIR",
        type=Path,
        default=None,
        help=(
            "Re-run metrics only on an existing session directory. "
            "Skips download, render, and model inference; reads "
            "original.* + reconstructed.* from each sub-folder and "
            "overwrites metrics.json + summary files in place."
        ),
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load .env (HF_TOKEN etc.)
    env_path = _PIPELINE_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        log.debug("Loaded .env from %s", env_path)
    else:
        load_dotenv()  # look in cwd / parent dirs

    # Load config
    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        return 1
    cfg = load_config(args.config)

    # Override API delay from environment
    if "API_DELAY_SECONDS" in os.environ:
        cfg.setdefault("pipeline", {})["api_delay_seconds"] = float(
            os.environ["API_DELAY_SECONDS"]
        )

    # --list
    if args.list:
        _print_available(cfg)
        return 0

    # --recompute-metrics
    if args.recompute_metrics:
        from src.metrics import MetricsCalculator
        from src.reporter import Reporter
        from src.session import Session
        return _recompute_from_session(
            args.recompute_metrics,
            cfg,
            MetricsCalculator(cfg),
            Reporter(cfg),
        )

    # Resolve shapes and models
    all_shapes: dict[str, dict] = cfg.get("shapes", {})
    all_models: dict[str, dict] = cfg.get("models", {})

    shapes = _filter_keys(all_shapes, args.shapes, "shapes")
    models = _filter_keys(all_models, args.models, "models") if not args.dry_run else {}

    if not shapes:
        log.error("No shapes to process. Check --shapes or config.yaml")
        return 1

    log.info("Shapes  : %s", list(shapes.keys()))
    log.info("Models  : %s", list(models.keys()) if models else "(dry run)")

    # Session
    session_root = args.session_dir or Path(
        _get(cfg, "pipeline", {}).get("session_dir", "sessions")
    )
    if not session_root.is_absolute():
        session_root = _PIPELINE_DIR / session_root

    from src.session import Session
    session = Session(base_dir=session_root)
    log.info("Session : %s", session.dir)

    from src.downloader import Downloader
    from src.renderer import render_mesh
    from src.metrics import MetricsCalculator
    from src.reporter import Reporter, null_metrics
    from src.models import get_model_client

    downloader = Downloader(cfg)
    metrics_calc = MetricsCalculator(cfg)
    reporter = Reporter(cfg)
    api_delay = float(_get(cfg, "pipeline", {}).get("api_delay_seconds", 2.0))

    results: dict[tuple[str, str], dict] = {}
    exit_code = 0

    # -----------------------------------------------------------------------
    # Main loop: shapes × models
    # -----------------------------------------------------------------------
    for shape_name, shape_cfg in shapes.items():
        log.info("=" * 60)
        log.info("SHAPE: %s", shape_name)

        # --- Step 1: Download ---
        try:
            mesh_path = downloader.download(shape_name, shape_cfg)
            log.info("[%s] mesh: %s", shape_name, mesh_path)
        except Exception as exc:
            log.error("[%s] download failed: %s", shape_name, exc)
            # Record failure for all models of this shape
            for model_name in models:
                results[(shape_name, model_name)] = null_metrics(f"download failed: {exc}")
            exit_code = 1
            continue

        # --- Step 2: Render ---
        import trimesh as _trimesh
        try:
            mesh = _trimesh.load(str(mesh_path), force="mesh", process=False)
            img = render_mesh(mesh, cfg)
        except Exception as exc:
            log.error("[%s] render failed: %s", shape_name, exc)
            for model_name in models:
                results[(shape_name, model_name)] = null_metrics(f"render failed: {exc}")
            exit_code = 1
            continue

        if args.dry_run:
            dry_dir = session.render_only_dir(shape_name)
            dry_dir.save_mesh(mesh_path, ext=mesh_path.suffix.lstrip("."))
            render_out = dry_dir.save_render(img)
            log.info("[%s] dry-run render saved: %s", shape_name, render_out)
            continue

        # --- Steps 3+4: Model calls + metrics ---
        for model_name, model_cfg in models.items():
            log.info("  MODEL: %s", model_name)
            art = session.artifact_dir(shape_name, model_name)

            # Save GT mesh and render into artifact dir
            gt_copy = art.save_mesh(mesh_path)
            render_path = art.save_render(img)

            # --- Step 3: Reconstruct ---
            try:
                client = get_model_client(model_name, model_cfg)
                recon_raw = client.reconstruct(render_path)
                recon_path = art.save_reconstructed(recon_raw)
                log.info("  [%s/%s] reconstructed: %s", shape_name, model_name, recon_path)
            except Exception as exc:
                log.error("  [%s/%s] reconstruction failed: %s", shape_name, model_name, exc)
                art.save_error(str(exc))
                results[(shape_name, model_name)] = null_metrics(str(exc))
                exit_code = 1
                time.sleep(api_delay)
                continue

            # --- Step 4: Metrics ---
            try:
                metrics = metrics_calc.compute_all(gt_copy, recon_path)
                art.save_metrics(metrics)
                results[(shape_name, model_name)] = metrics
                log.info(
                    "  [%s/%s] CD=%.5f F=%.3f MRS=%.3f",
                    shape_name, model_name,
                    metrics["chamfer_distance"],
                    metrics["f_score"],
                    metrics["morphing_readiness_score"],
                )
            except Exception as exc:
                log.error("  [%s/%s] metrics failed: %s", shape_name, model_name, exc)
                art.save_error(f"metrics failed: {exc}")
                results[(shape_name, model_name)] = null_metrics(f"metrics failed: {exc}")
                exit_code = 1

            time.sleep(api_delay)

    # -----------------------------------------------------------------------
    # Write summary tables
    # -----------------------------------------------------------------------
    if results and not args.dry_run:
        reporter.write(session, results)
        log.info("=" * 60)
        log.info("Session complete: %s", session.dir)
        log.info("Summary CSV : %s", session.summary_csv_path)
        log.info("Summary MD  : %s", session.summary_md_path)
        _print_summary(results)
    elif args.dry_run:
        log.info("=" * 60)
        log.info("Dry run complete. Renders in: %s", session.dir)

    return exit_code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recompute_from_session(
    session_path: Path,
    cfg: dict,
    metrics_calc: Any,
    reporter: Any,
) -> int:
    """
    Re-run metrics on all (shape, model) pairs found in *session_path*.

    Each subdirectory whose name contains an underscore and has both an
    original.* and a reconstructed.* file is treated as a {shape}_{model}
    pair.  Metrics are recomputed, metrics.json is overwritten in place,
    and summary.csv / summary.md are regenerated.
    """
    from src.session import Session, ArtifactDir
    from src.reporter import null_metrics

    try:
        session = Session.open_existing(session_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    log.info("Recomputing metrics in session: %s", session.dir)

    # Discover all sub-directories that look like shape_model pairs
    results: dict[tuple[str, str], dict] = {}
    exit_code = 0

    subdirs = sorted(d for d in session.dir.iterdir() if d.is_dir())
    if not subdirs:
        log.error("No sub-directories found in %s", session.dir)
        return 1

    for subdir in subdirs:
        art = ArtifactDir(subdir)
        gt_path = art.original_path
        recon_path = art.reconstructed_path

        # Parse shape / model from directory name (convention: {shape}_{model})
        name = subdir.name
        if "_" not in name:
            log.debug("Skipping %s — no underscore in name", name)
            continue

        # Try all known model names to split correctly
        all_models = list(cfg.get("models", {}).keys())
        shape_name = model_name = None
        for m in all_models:
            if name.endswith(f"_{m}"):
                model_name = m
                shape_name = name[: -(len(m) + 1)]
                break
        if shape_name is None or model_name is None:
            # Fallback: split on last underscore
            shape_name, model_name = name.rsplit("_", 1)

        if gt_path is None:
            log.warning("[%s] no original.* found — skipping", name)
            results[(shape_name, model_name)] = null_metrics("no GT mesh found")
            continue

        if recon_path is None:
            log.warning("[%s] no reconstructed.* found — skipping", name)
            results[(shape_name, model_name)] = null_metrics("no reconstructed mesh found")
            continue

        log.info("[%s/%s] recomputing metrics ...", shape_name, model_name)
        try:
            metrics = metrics_calc.compute_all(gt_path, recon_path)
            art.save_metrics(metrics)
            results[(shape_name, model_name)] = metrics
            log.info(
                "[%s/%s] CD=%.5f F=%.3f MRS=%.3f cost=%s",
                shape_name, model_name,
                metrics["chamfer_distance"],
                metrics["f_score"],
                metrics["morphing_readiness_score"],
                metrics.get("cleanup_cost", "?"),
            )
        except Exception as exc:
            log.error("[%s/%s] metrics failed: %s", shape_name, model_name, exc)
            art.save_metrics({"error": f"metrics failed: {exc}"})
            results[(shape_name, model_name)] = null_metrics(f"metrics failed: {exc}")
            exit_code = 1

    if results:
        reporter.write(session, results)
        log.info("Summary CSV : %s", session.summary_csv_path)
        log.info("Summary MD  : %s", session.summary_md_path)
        _print_summary(results)

    return exit_code


def _filter_keys(
    mapping: dict,
    selection: str | None,
    label: str,
) -> dict:
    if selection is None:
        return mapping
    requested = [s.strip() for s in selection.split(",") if s.strip()]
    unknown = [k for k in requested if k not in mapping]
    if unknown:
        log.warning("Unknown %s (ignored): %s. Available: %s",
                    label, unknown, list(mapping.keys()))
    return {k: mapping[k] for k in requested if k in mapping}


def _print_available(cfg: dict) -> None:
    print("\nAvailable shapes:")
    for name in cfg.get("shapes", {}):
        print(f"  {name}")
    print("\nAvailable models:")
    for name in cfg.get("models", {}):
        print(f"  {name}")


def _print_summary(results: dict) -> None:
    from src.reporter import _METRIC_COLS
    print("\n─── Quick summary ───")
    print(f"{'Shape':<10} {'Model':<12} {'CD':>9} {'F':>7} {'MRS':>7} {'Status'}")
    print("─" * 55)
    for (shape, model), m in sorted(results.items()):
        err = m.get("error")
        if err:
            print(f"{shape:<10} {model:<12} {'ERR':>9} {'—':>7} {'—':>7}  ❌ {err[:30]}")
        else:
            cd = m.get("chamfer_distance")
            fs = m.get("f_score")
            mrs = m.get("morphing_readiness_score")
            print(
                f"{shape:<10} {model:<12} "
                f"{cd:>9.5f} {fs:>7.3f} {mrs:>7.3f}"
            )


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
