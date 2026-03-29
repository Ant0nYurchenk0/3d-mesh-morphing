"""
MorphingState — the shared state dict passed between LangGraph pipeline nodes.

Key design decisions:
  - All file paths are stored as str (not Path) for JSON-serialisability.
  - The Session object is not JSON-serialisable but lives only in-memory.
  - skip_enhance / skip_base_mesh flags control which nodes execute logic
    versus pass through immediately.
  - When input_mesh is provided (no image), the graph routes directly to
    render_mesh_node, bypassing enhance_image_node and image_to_base_mesh_node.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class MorphingState(TypedDict):
    # ---- Input (populated before graph starts) ----
    cfg: Any                        # MorphingConfig instance
    input_image: str | None         # path to input image, or None
    input_mesh: str | None          # path to input mesh, or None
    prompt_file: str                # path to .txt file with morphing prompt
    session_base: Any               # Path — always morphing_pipeline/sessions/ (set by run_morphing.py)
    skip_enhance: bool              # skip node 1 (enhance_image); uses raw input_image
    skip_base_mesh: bool            # skip node 2 (image_to_base_mesh); no base mesh gen
    model_name: str                 # image-to-3D model name (trellis, hunyuan3d, …)
    remesh: bool                    # run Instant Meshes remesh as stage 3 of repair (node 6)
    morph_method: str               # "sdf" | "differential" — which node 7 to run

    # ---- Set by setup_node ----
    session: Any                    # Session instance (in-memory only)
    prompt: str                     # morphing prompt loaded from prompt_file

    # ---- Accumulated during execution ----
    # Node 1 output (or input_image if skipped):
    base_image_path: str | None     # the image used for morphing (after enhance or from node 3)
    # Node 2 output (or input_mesh if mesh input):
    base_mesh_path: str | None      # base mesh for the final morph transition
    # Node 3 output (render of input mesh, only when input_mesh is used):
    render_image_path: str | None   # render of input_mesh used as base_image_path
    # Node 4 output:
    morphed_image_path: str | None  # GPT-generated morphed image
    # Node 5 output:
    target_mesh_path: str | None    # image-to-3D reconstruction of morphed image
    # Node 6 output:
    repaired_mesh_path: str | None  # repaired version of target mesh
    repair_report: dict | None      # repair stage summary (costs, stages_run, face_count)
    # Node 7 output:
    transition_path: str | None     # path to directory with transition frames

    # ---- Control ----
    exit_code: int
