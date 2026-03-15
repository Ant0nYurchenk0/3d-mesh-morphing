"""
Mesh quality assessment utilities shared between both pipelines.

  benchmark_pipeline — uses cleanup_cost as one of 8 evaluation metrics
  morphing_pipeline  — uses cleanup_cost to gate and report on repair stages

Public API:
    cleanup_cost(mesh: trimesh.Trimesh) -> int
        Count of morphing-blocking issues [0–7].
    check_mesh_file(path: Path) -> int
        Load mesh from file and return its cleanup_cost.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def cleanup_cost(mesh_pred: trimesh.Trimesh) -> int:
    """
    Count of morphing-blocking issues in *mesh_pred*. Range [0, 7].

    Issues checked (each contributes 1):
      1. Multiple connected components
      2. Not watertight
      3. Non-manifold edges (shared by >2 faces)
      4. Zero-area faces (area < 1e-10)
      5. Interior vertices (distance to centroid < 0.3 × mean distance)
      6. Euler number ≠ 2
      7. Too many vertices (>15 000)
    """
    cost = 0

    try:
        if len(mesh_pred.split(only_watertight=False)) > 1:
            cost += 1
    except Exception:
        pass

    if not mesh_pred.is_watertight:
        cost += 1

    try:
        edge_counts = Counter(map(tuple, np.sort(mesh_pred.edges, axis=1).tolist()))
        if any(c > 2 for c in edge_counts.values()):
            cost += 1
    except Exception:
        pass

    try:
        if np.any(mesh_pred.area_faces < 1e-10):
            cost += 1
    except Exception:
        pass

    try:
        dists = np.linalg.norm(mesh_pred.vertices - mesh_pred.centroid, axis=1)
        if np.any(dists < dists.mean() * 0.3):
            cost += 1
    except Exception:
        pass

    try:
        if int(mesh_pred.euler_number) != 2:
            cost += 1
    except Exception:
        pass

    if len(mesh_pred.vertices) > 15_000:
        cost += 1

    return cost


def check_mesh_file(path: Path) -> int:
    """
    Load mesh from *path* and return its cleanup_cost.

    Returns 7 (worst) if the file cannot be loaded.
    """
    try:
        loaded = trimesh.load(str(path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                g for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
            ]
            if not meshes:
                return 7
            loaded = trimesh.util.concatenate(meshes)
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            return 7
        return cleanup_cost(loaded)
    except Exception as exc:
        log.warning("[mesh_quality] check_mesh_file failed for %s: %s", path, exc)
        return 7
