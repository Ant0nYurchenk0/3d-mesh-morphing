#!/usr/bin/env python3
"""
3D Mesh Morphing Pipeline — entry point.

Usage:
    # Image input: enhance → base mesh → morph image → target mesh → (repair) → (transition)
    uv run python run_morphing.py --input-image path/to/photo.png --prompt-file prompt.txt

    # Mesh input: render → morph image → target mesh → (repair) → (transition)
    uv run python run_morphing.py --input-mesh path/to/object.glb --prompt-file prompt.txt

    # Skip GPT identity-copy step (node 1) and use raw input image directly
    uv run python run_morphing.py --input-image photo.png --prompt-file p.txt --skip-enhance

    # Skip base-mesh generation (node 2); only produce target mesh from morphed image
    uv run python run_morphing.py --input-image photo.png --prompt-file p.txt --skip-base-mesh

    # Choose which image-to-3D model to use
    uv run python run_morphing.py --input-image photo.png --prompt-file p.txt --model hunyuan3d

    # Custom config
    uv run python run_morphing.py --input-image photo.png --prompt-file p.txt \\
        --config my_config.yaml

    # List available models
    uv run python run_morphing.py --list

Session artifacts are written to:
    sessions/YYYY-MM-DD_HHMMSS_{uid}/
        enhanced_image.png    ← node 1: GPT-reproduced input image
        base_mesh.glb         ← node 2: image-to-3D base mesh (if generated)
        render.png            ← node 3: render of input mesh (if mesh input)
        morphed_image.png     ← node 4: GPT-morphed image
        target_mesh.glb       ← node 5: image-to-3D target mesh
        (repaired_mesh.glb)   ← node 6: repaired mesh (placeholder)
        (transition/)         ← node 7: morph transition frames (placeholder)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow importing from shared/ (sibling directory at the repo root)
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("morphing")

_PIPELINE_DIR = Path(__file__).parent
_DEFAULT_CONFIG = _PIPELINE_DIR / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D mesh morphing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input (mutually exclusive: image or mesh) ─────────────────────
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input-image",
        type=Path,
        metavar="IMAGE",
        help="Path to input image (PNG/JPEG). Triggers image flow.",
    )
    input_group.add_argument(
        "--input-mesh",
        type=Path,
        metavar="MESH",
        help="Path to input mesh (GLB/OBJ/STL/…). Triggers mesh flow.",
    )

    # ── Required: prompt ──────────────────────────────────────────────
    p.add_argument(
        "--prompt-file",
        type=Path,
        metavar="TXT",
        required=False,
        help="Path to a .txt file containing the morphing prompt for node 4.",
    )

    # ── Model selection ───────────────────────────────────────────────
    p.add_argument(
        "--model",
        default="trellis",
        help="Image-to-3D model name from config.yaml (default: trellis)",
    )

    # ── Skip flags ────────────────────────────────────────────────────
    p.add_argument(
        "--skip-enhance",
        action="store_true",
        help="Skip node 1 (GPT identity regeneration); use input image directly.",
    )
    p.add_argument(
        "--skip-base-mesh",
        action="store_true",
        help="Skip node 2 (base mesh generation); only target mesh is produced.",
    )

    # ── Paths / config ────────────────────────────────────────────────
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {_DEFAULT_CONFIG})",
    )

    # ── Misc ──────────────────────────────────────────────────────────
    p.add_argument(
        "--list",
        action="store_true",
        help="Print available models from config then exit.",
    )
    p.add_argument(
        "--remesh",
        action="store_true",
        help="Run Instant Meshes remesh as stage 3 of repair (node 6).",
    )
    p.add_argument(
        "--morph-method",
        choices=["sdf", "differential"],
        default="sdf",
        help=(
            "Morphing algorithm for node 7 (default: sdf). "
            "'sdf' = SDF interpolation + marching cubes; "
            "'differential' = placeholder (not yet implemented)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    env_path = _PIPELINE_DIR / ".env"
    from dotenv import load_dotenv
    load_dotenv(env_path if env_path.exists() else None)

    from src.config import MorphingConfig

    if not args.config.exists():
        log.error("Config file not found: %s", args.config)
        return 1

    cfg = MorphingConfig.from_yaml(args.config)

    if args.list:
        print("\nAvailable models:", list(cfg.models.keys()))
        return 0

    # ── Validate inputs ───────────────────────────────────────────────
    if args.input_image is None and args.input_mesh is None:
        log.error("Provide either --input-image or --input-mesh.")
        return 1

    if args.prompt_file is None:
        log.error("--prompt-file is required.")
        return 1

    if args.input_image and not args.input_image.exists():
        log.error("Input image not found: %s", args.input_image)
        return 1

    if args.input_mesh and not args.input_mesh.exists():
        log.error("Input mesh not found: %s", args.input_mesh)
        return 1

    if not args.prompt_file.exists():
        log.error("Prompt file not found: %s", args.prompt_file)
        return 1

    model_name = args.model
    if model_name not in cfg.models:
        log.error(
            "Model '%s' not in config. Available: %s",
            model_name, list(cfg.models.keys()),
        )
        return 1

    # Sessions are always stored under morphing_pipeline/sessions/
    session_base = _PIPELINE_DIR / cfg.pipeline.session_dir

    log.info("Input      : %s", args.input_image or args.input_mesh)
    log.info("Prompt file: %s", args.prompt_file)
    log.info("Model      : %s", model_name)
    log.info("skip_enhance   : %s", args.skip_enhance)
    log.info("skip_base_mesh : %s", args.skip_base_mesh)
    log.info("remesh         : %s", args.remesh)
    log.info("morph_method   : %s", args.morph_method)

    from src.pipeline.graph import build_graph

    initial_state = {
        "cfg": cfg,
        "input_image": str(args.input_image) if args.input_image else None,
        "input_mesh": str(args.input_mesh) if args.input_mesh else None,
        "prompt_file": str(args.prompt_file),
        "session_base": session_base,
        "skip_enhance": args.skip_enhance,
        "skip_base_mesh": args.skip_base_mesh,
        "model_name": model_name,
        "remesh": args.remesh,
        "morph_method": args.morph_method,
        # Set by nodes:
        "session": None,
        "prompt": "",
        "base_image_path": None,
        "base_mesh_path": None,
        "render_image_path": None,
        "morphed_image_path": None,
        "target_mesh_path": None,
        "repaired_mesh_path": None,
        "repair_report": None,
        "transition_path": None,
        "exit_code": 0,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    session = final_state.get("session")
    if session:
        log.info("Session complete: %s", session.dir)
        _print_summary(final_state)

    return int(final_state.get("exit_code", 0))


def _print_summary(state: dict) -> None:
    print("\n─── Morphing pipeline summary ───")
    fields = [
        ("base_image_path",    "Base image      "),
        ("base_mesh_path",     "Base mesh       "),
        ("render_image_path",  "Mesh render     "),
        ("morphed_image_path", "Morphed image   "),
        ("target_mesh_path",   "Target mesh     "),
        ("repaired_mesh_path", "Repaired mesh   "),
        ("transition_path",    "Transition      "),
    ]
    for key, label in fields:
        val = state.get(key)
        print(f"  {label}: {val or '—'}")


if __name__ == "__main__":
    sys.exit(main())
