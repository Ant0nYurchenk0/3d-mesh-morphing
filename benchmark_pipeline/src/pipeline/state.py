"""
BenchmarkState — the shared state dict passed between LangGraph pipeline nodes.

Key design decisions:
  - All file paths are stored as str (not Path) for JSON-serialisability.
  - Results are keyed by "{shape_name}__{model_name}" (double underscore) to
    avoid ambiguity with model names that contain single underscores.
  - render_images holds in-memory numpy arrays; no persistence between runs.
  - The Session object is not JSON-serialisable but is only used in-memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pathlib import Path


class BenchmarkState(TypedDict):
    # ---- Input (populated before graph starts) ----
    cfg: Any                        # BenchmarkConfig instance
    shapes: dict[str, dict]         # shape_name -> shape_cfg (CLI-filtered subset)
    models: dict[str, dict]         # model_name -> model_cfg (CLI-filtered subset)
    dry_run: bool
    session_base: Any               # Path | None — override for session root dir
    recompute_from: Any             # Path | None — existing session for --recompute-metrics

    # ---- Set by setup_node ----
    session: Any                    # Session instance (in-memory only)

    # ---- Accumulated during execution ----
    mesh_paths: dict[str, str]      # shape_name -> str(mesh_path)
    render_images: dict[str, Any]   # shape_name -> np.ndarray
    recon_paths: dict[str, str]     # "{shape}__{model}" -> str(recon_path)
    results: dict[str, dict]        # "{shape}__{model}" -> metrics dict (or error)

    # ---- Control ----
    exit_code: int
