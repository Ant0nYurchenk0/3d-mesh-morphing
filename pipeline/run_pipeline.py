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

    # List available shapes and models
    uv run python run_pipeline.py --list

    # Recompute metrics on an existing session
    uv run python run_pipeline.py --recompute-metrics sessions/2026-03-14_145044_7182f77e

    # Custom config or session directory
    uv run python run_pipeline.py --config my_config.yaml --session-dir /tmp/sessions

Session artifacts are written to:
    sessions/YYYY-MM-DD_HHMMSS_{uid}/
        {shape}_{model}/
            original.{ext}        ← GT mesh (normalised)
            render.png            ← 1024×1024 canonical render
            reconstructed.glb     ← model output
            metrics.json          ← 8 metrics + metadata
        summary.csv
        summary_per_model.csv
        summary.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

_PIPELINE_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _PIPELINE_DIR / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image-to-3D evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--shapes",
        default=None,
        help="Comma-separated shape names to test (default: all in config)",
    )
    p.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to test (default: all in config)",
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
        help="Session output root (default: pipeline/sessions/)",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print available shapes and models then exit",
    )
    p.add_argument(
        "--recompute-metrics",
        metavar="SESSION_DIR",
        type=Path,
        default=None,
        help=(
            "Re-run metrics only on an existing session directory. "
            "Reads original.* + reconstructed.* from each sub-folder and "
            "overwrites metrics.json + summary files in place."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    env_path = _PIPELINE_DIR / ".env"
    load_dotenv(env_path if env_path.exists() else None)

    from src.config import BenchmarkConfig

    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        return 1

    cfg = BenchmarkConfig.from_yaml(args.config)

    if os.environ.get("API_DELAY_SECONDS"):
        cfg.pipeline.api_delay_seconds = float(os.environ["API_DELAY_SECONDS"])

    if args.list:
        print("\nAvailable shapes:", list(cfg.shapes.keys()))
        print("Available models:", list(cfg.models.keys()))
        return 0

    shapes = _filter(cfg.shapes, args.shapes, "shapes")
    models = _filter(cfg.models, args.models, "models") if not args.dry_run else {}

    if not shapes:
        log.error("No shapes to process. Check --shapes or config.yaml.")
        return 1

    log.info("Shapes : %s", list(shapes.keys()))
    log.info("Models : %s", list(models.keys()) if models else "(dry run)")

    session_base = args.session_dir
    if session_base is None:
        session_base = _PIPELINE_DIR / cfg.pipeline.session_dir

    from src.pipeline.graph import build_graph

    initial_state = {
        "cfg": cfg,
        "shapes": shapes,
        "models": models,
        "dry_run": args.dry_run,
        "session_base": session_base,
        "recompute_from": args.recompute_metrics,
        "session": None,
        "mesh_paths": {},
        "render_images": {},
        "recon_paths": {},
        "results": {},
        "exit_code": 0,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    return int(final_state.get("exit_code", 0))


def _filter(mapping: dict, selection: str | None, label: str) -> dict:
    if selection is None:
        return mapping
    requested = [s.strip() for s in selection.split(",") if s.strip()]
    unknown = [k for k in requested if k not in mapping]
    if unknown:
        log.warning(
            "Unknown %s (ignored): %s. Available: %s",
            label, unknown, list(mapping.keys()),
        )
    return {k: mapping[k] for k in requested if k in mapping}


if __name__ == "__main__":
    sys.exit(main())
