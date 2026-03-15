"""
Shared mesh loading and normalisation utilities.

Used by Downloader (pre-processing) and MetricsCalculator (before metrics computation)
to avoid duplication between the two modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh file as a single Trimesh, merging all geometries from a Scene."""
    loaded = trimesh.load(str(path), force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not load a Trimesh from {path} (got {type(loaded).__name__})")

    if len(loaded.faces) == 0:
        raise ValueError(f"Mesh from {path} has no faces")

    return loaded


def normalise_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre at centroid and scale to unit bounding sphere."""
    v = mesh.vertices.copy().astype(np.float64)
    v -= v.mean(axis=0)
    max_norm = np.linalg.norm(v, axis=1).max()
    if max_norm > 1e-9:
        v /= max_norm
    return trimesh.Trimesh(vertices=v, faces=mesh.faces.copy(), process=False)
