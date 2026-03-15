"""
LangGraph pipeline nodes for the 3D mesh morphing pipeline.

Each node receives the full MorphingState, performs its stage of work,
and returns a partial state update dict.

Node execution order (see graph.py):

  Input: image
    setup → enhance_image (1) → image_to_base_mesh (2) → morph_image (4)
                                                              ↓
  Input: mesh                                          target_mesh (5)
    setup → render_mesh (3) ──────────────────────────────►  ↓
                                                        repair_mesh (6)
                                                              ↓
                                                       morph_meshes (7) → END

Node skip behaviour:
  - enhance_image_node:      if state.skip_enhance, copies input_image as base_image_path
  - image_to_base_mesh_node: if state.skip_base_mesh, sets base_mesh_path = None
  - render_mesh_node:        always runs when input_mesh is provided
  - repair_mesh_node:        multi-stage repair (PyMeshLab → ManifoldPlus → Instant Meshes)
  - morph_meshes_node:       placeholder (logs; no output yet)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .state import MorphingState

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Node 0: setup
# ------------------------------------------------------------------

def setup_node(state: MorphingState) -> dict:
    """
    Create a timestamped Session directory and load the morphing prompt.

    Sets: session, prompt
    """
    from shared.session import Session

    # session_base is always set by run_morphing.py to morphing_pipeline/sessions/
    session_base: Path = state["session_base"]
    session = Session(base_dir=session_base)
    log.info("[setup] session: %s", session.dir)

    prompt_file = Path(state["prompt_file"])
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    log.info("[setup] prompt loaded (%d chars)", len(prompt))

    return {"session": session, "prompt": prompt}


# ------------------------------------------------------------------
# Node 1: enhance_image  (skippable)
# ------------------------------------------------------------------

def enhance_image_node(state: MorphingState) -> dict:
    """
    Node 1 — identity regeneration via GPT (skippable).

    When skip_enhance=True, passes the raw input_image through without
    calling the OpenAI API.  When skip_enhance=False, sends the image to
    GPT with a prompt asking it to reproduce it faithfully.

    Sets: base_image_path
    """
    import os
    from src.image_generator import GPTImageGenerator

    input_image = state["input_image"]
    session = state["session"]

    if state.get("skip_enhance", False):
        log.info("[enhance_image] skipped — using raw input image")
        return {"base_image_path": input_image}

    cfg = state["cfg"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example → .env and add your OpenAI API key."
        )

    generator = GPTImageGenerator(
        api_key=api_key,
        model=cfg.image_generation.model,
        size=cfg.image_generation.size,
    )
    output_path = session.dir / "enhanced_image.png"
    try:
        generator.enhance(Path(input_image), output_path)
        log.info("[enhance_image] saved -> %s", output_path)
        return {"base_image_path": str(output_path)}
    except Exception as exc:
        log.error("[enhance_image] GPT call failed: %s — falling back to raw input", exc)
        return {"base_image_path": input_image}


# ------------------------------------------------------------------
# Node 2: image_to_base_mesh  (skippable)
# ------------------------------------------------------------------

def image_to_base_mesh_node(state: MorphingState) -> dict:
    """
    Node 2 — generate a base 3D mesh from the (possibly enhanced) image (skippable).

    When skip_base_mesh=True, skips the API call.  The base mesh is optional
    for the final morphing step — if absent, only the target mesh is generated.

    Sets: base_mesh_path
    """
    from shared.models import get_model_client

    if state.get("skip_base_mesh", False):
        log.info("[image_to_base_mesh] skipped")
        return {"base_mesh_path": None}

    cfg = state["cfg"]
    model_name = state["model_name"]
    model_cfg = cfg.models.get(model_name)
    if model_cfg is None:
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available: {list(cfg.models.keys())}"
        )

    session = state["session"]
    base_image = state.get("base_image_path")
    if not base_image:
        log.warning("[image_to_base_mesh] no base image — skipping")
        return {"base_mesh_path": None}

    log.info("[image_to_base_mesh] model=%s image=%s", model_name, base_image)
    try:
        client = get_model_client(model_name, model_cfg)
        raw_path = client.reconstruct(Path(base_image))
        dest = session.dir / f"base_mesh{raw_path.suffix}"
        import shutil
        shutil.copy2(raw_path, dest)
        log.info("[image_to_base_mesh] saved -> %s", dest)
        return {"base_mesh_path": str(dest)}
    except Exception as exc:
        log.error("[image_to_base_mesh] failed: %s", exc)
        return {"base_mesh_path": None}


# ------------------------------------------------------------------
# Node 3: render_mesh  (runs only when input is a mesh)
# ------------------------------------------------------------------

def render_mesh_node(state: MorphingState) -> dict:
    """
    Node 3 — render input mesh to a 2D image (runs when input_mesh is provided).

    Produces the base_image_path that feeds into morph_image_node (node 4).
    Also sets base_mesh_path to the input_mesh so it can be used in node 7.

    Sets: base_image_path, base_mesh_path, render_image_path
    """
    import trimesh
    from shared.renderer import Renderer
    from PIL import Image

    cfg = state["cfg"]
    input_mesh = state["input_mesh"]
    session = state["session"]

    log.info("[render_mesh] rendering: %s", input_mesh)
    try:
        mesh = trimesh.load(str(input_mesh), force="mesh", process=False)
        renderer = Renderer(cfg.render)
        img = renderer.render(mesh)

        render_path = session.dir / "render.png"
        Image.fromarray(img).save(render_path)
        log.info("[render_mesh] saved render -> %s", render_path)

        return {
            "render_image_path": str(render_path),
            "base_image_path": str(render_path),
            "base_mesh_path": input_mesh,
        }
    except Exception as exc:
        log.error("[render_mesh] failed: %s", exc)
        raise


# ------------------------------------------------------------------
# Node 4: morph_image
# ------------------------------------------------------------------

def morph_image_node(state: MorphingState) -> dict:
    """
    Node 4 — generate morphed image from base image using the user's prompt.

    Calls the OpenAI images.edit() API with the base_image_path and prompt.

    Sets: morphed_image_path
    """
    import os
    from src.image_generator import GPTImageGenerator

    cfg = state["cfg"]
    session = state["session"]
    base_image = state.get("base_image_path")
    prompt = state["prompt"]

    if not base_image:
        raise RuntimeError("[morph_image] no base_image_path in state")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example → .env and add your OpenAI API key."
        )

    generator = GPTImageGenerator(
        api_key=api_key,
        model=cfg.image_generation.model,
        size=cfg.image_generation.size,
    )
    output_path = session.dir / "morphed_image.png"
    log.info("[morph_image] generating morphed image from: %s", base_image)
    generator.morph(Path(base_image), prompt, output_path)
    log.info("[morph_image] saved -> %s", output_path)

    return {"morphed_image_path": str(output_path)}


# ------------------------------------------------------------------
# Node 5: target_mesh
# ------------------------------------------------------------------

def target_mesh_node(state: MorphingState) -> dict:
    """
    Node 5 — reconstruct 3D target mesh from the morphed image.

    Uses the same ImageTo3DClient infrastructure as benchmark_pipeline.

    Sets: target_mesh_path
    """
    from shared.models import get_model_client

    cfg = state["cfg"]
    model_name = state["model_name"]
    model_cfg = cfg.models.get(model_name)
    if model_cfg is None:
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available: {list(cfg.models.keys())}"
        )

    session = state["session"]
    morphed_image = state.get("morphed_image_path")
    if not morphed_image:
        raise RuntimeError("[target_mesh] no morphed_image_path in state")

    log.info("[target_mesh] model=%s image=%s", model_name, morphed_image)
    client = get_model_client(model_name, model_cfg)
    raw_path = client.reconstruct(Path(morphed_image))

    dest = session.dir / f"target_mesh{raw_path.suffix}"
    import shutil
    shutil.copy2(raw_path, dest)
    log.info("[target_mesh] saved -> %s", dest)

    return {"target_mesh_path": str(dest)}


# ------------------------------------------------------------------
# Node 6: repair_mesh
# ------------------------------------------------------------------

def repair_mesh_node(state: MorphingState) -> dict:
    """
    Node 6 — multi-stage mesh repair.

    Stage 1 (always): PyMeshLab — remove fragments, fix non-manifold,
        close holes, remove degenerate faces, decimate, re-orient normals.
    Stage 2 (if cleanup_cost > 0 after stage 1): ManifoldPlus CLI.
    Stage 3 (if remesh=True): Instant Meshes CLI.

    Sets: repaired_mesh_path, repair_report
    """
    import json
    from src.mesh_repairer import MeshRepairer

    target_mesh = state.get("target_mesh_path")
    if not target_mesh:
        raise RuntimeError("[repair_mesh] no target_mesh_path in state")

    cfg = state["cfg"]
    session = state["session"]
    output_path = session.dir / "repaired_mesh.glb"

    repairer = MeshRepairer(cfg.repair)
    report = repairer.repair(
        input_path=Path(target_mesh),
        output_path=output_path,
        session_dir=session.dir,
        remesh=state.get("remesh", False),
    )

    report_path = session.dir / "repair_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    log.info(
        "[repair_mesh] cost %d → %d  stages=%s  faces=%s",
        report["cost_before"], report["cost_final"],
        report["stages_run"], report["face_count_final"],
    )
    return {"repaired_mesh_path": str(output_path), "repair_report": report}


# ------------------------------------------------------------------
# Node 7: morph_meshes  (placeholder)
# ------------------------------------------------------------------

def morph_meshes_node(state: MorphingState) -> dict:
    """
    Node 7 — create transition between base and target meshes (PLACEHOLDER).

    Currently a no-op. Future implementation should compute an OT-based
    (Optimal Transport) or linear interpolation morphing sequence between
    base_mesh_path and repaired_mesh_path, producing a series of intermediate
    meshes or an animation file.

    Sets: transition_path (None until implemented)
    """
    base_mesh = state.get("base_mesh_path")
    repaired_mesh = state.get("repaired_mesh_path")

    log.info(
        "[morph_meshes] PLACEHOLDER — base=%s target=%s",
        base_mesh, repaired_mesh,
    )
    log.info("[morph_meshes] Morphing transition not yet implemented.")

    session = state["session"]
    log.info("[morph_meshes] session complete: %s", session.dir)

    return {"transition_path": None, "exit_code": 0}
