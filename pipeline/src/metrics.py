"""
Mesh quality metrics for image-to-3D evaluation.

All metrics operate on a (ground-truth mesh, predicted mesh) pair.
Both meshes are normalised to unit bounding sphere before any computation
so that scale differences between the original and model output don't
produce meaningless numbers.

Metrics computed:
  1. Chamfer Distance (CD)         — bidirectional mean squared NN distance
  2. F-Score @ τ=0.02              — % surface within 2% accuracy (harmonic mean)
  3. Hausdorff Distance            — worst-case nearest-neighbour distance
  4. Volume IoU                    — voxel-grid volumetric overlap (watertight only)
  5. Normal Consistency (NC)       — mean |dot(normals)| at matched surface points
  6. Morphing Readiness Score (MRS)— weighted composite for OT-morphing suitability
                                     (50% structural health, 30% NC, 20% topology;
                                      geometric accuracy intentionally excluded)
  7. DINOv2 Similarity             — mean cosine similarity of DINOv2 CLS features
                                     across 8 matched viewpoints; primary perceptual
                                     ranking metric. Requires: uv sync --extra dino
                                     Returns None when torch/transformers not installed.
  8. Cleanup Cost                  — count of morphing-blocking issues (0–7) as
                                     diagnosed by diagnose_mesh.py; lower is better.

Reference thresholds (CD on unit sphere):
  CD < 0.005   excellent
  CD < 0.010   good
  CD > 0.030   bad

Public API:
    MetricsCalculator(cfg).compute_all(gt_path, pred_path) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import KDTree

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Top-level orchestrator
# ------------------------------------------------------------------

class MetricsCalculator:
    def __init__(self, cfg: Any) -> None:
        mc = _get(cfg, "metrics", {})
        self.n: int = _get(mc, "n_sample_points", 10_000)
        self.tau: float = _get(mc, "fscore_tau", 0.02)
        self.pitch_frac: float = _get(mc, "voxel_pitch_fraction", 1.0 / 64)

    def compute_all(self, gt_path: Path, pred_path: Path) -> dict:
        """
        Compute all 8 metrics for a (ground-truth, predicted) mesh pair.
        Returns a dict with keys: chamfer_distance, f_score, hausdorff_distance,
        volume_iou, normal_consistency, morphing_readiness_score,
        dino_similarity, cleanup_cost, metadata.
        volume_iou is None when either mesh is not watertight.
        dino_similarity is None when torch/transformers are not installed.
        """
        log.info("[metrics] loading GT: %s", gt_path)
        log.info("[metrics] loading pred: %s", pred_path)

        try:
            mesh_gt = _load_mesh(gt_path)
        except Exception as exc:
            raise ValueError(f"Failed to load GT mesh from {gt_path}: {exc}") from exc

        try:
            mesh_pred = _load_mesh(pred_path)
        except Exception as exc:
            raise ValueError(f"Failed to load predicted mesh from {pred_path}: {exc}") from exc

        # Normalise both to unit bounding sphere
        mesh_gt = _normalise_mesh(mesh_gt)
        mesh_pred = _normalise_mesh(mesh_pred)

        # Sample point clouds
        pts_gt = _sample_points(mesh_gt, self.n)
        pts_pred = _sample_points(mesh_pred, self.n)

        log.info("[metrics] computing CD, F-Score, Hausdorff...")
        cd = chamfer_distance(pts_gt, pts_pred)
        fs = fscore(pts_gt, pts_pred, tau=self.tau)
        hd = hausdorff_distance(pts_gt, pts_pred)

        log.info("[metrics] computing Volume IoU...")
        viou = volume_iou(mesh_gt, mesh_pred, pitch_fraction=self.pitch_frac)

        log.info("[metrics] computing Normal Consistency...")
        nc = normal_consistency(mesh_gt, mesh_pred, n=self.n)

        log.info("[metrics] computing MRS...")
        mrs = morphing_readiness_score(
            mesh_pred=mesh_pred,
            normal_consistency_score=nc,
        )

        log.info("[metrics] computing DINOv2 similarity...")
        dino_sim = dino_similarity(mesh_gt, mesh_pred)

        log.info("[metrics] computing cleanup cost...")
        cost = cleanup_cost(mesh_pred)

        result = {
            "chamfer_distance": round(cd, 8),
            "f_score": round(fs, 6),
            "hausdorff_distance": round(hd, 8),
            "volume_iou": round(viou, 6) if viou is not None else None,
            "normal_consistency": round(nc, 6),
            "morphing_readiness_score": round(mrs, 6),
            "dino_similarity": round(dino_sim, 6) if dino_sim is not None else None,
            "cleanup_cost": cost,
            "metadata": {
                "is_watertight_gt": bool(mesh_gt.is_watertight),
                "is_watertight_pred": bool(mesh_pred.is_watertight),
                "n_vertices_pred": len(mesh_pred.vertices),
                "n_faces_pred": len(mesh_pred.faces),
                "n_vertices_gt": len(mesh_gt.vertices),
                "n_faces_gt": len(mesh_gt.faces),
                "euler_number_pred": int(mesh_pred.euler_number),
                "euler_number_gt": int(mesh_gt.euler_number),
            },
        }
        dino_str = f"{dino_sim:.3f}" if dino_sim is not None else "N/A"
        log.info(
            "[metrics] CD=%.5f F=%.3f HD=%.5f IoU=%s NC=%.3f MRS=%.3f DINO=%s cost=%d",
            cd, fs, hd,
            f"{viou:.3f}" if viou is not None else "N/A",
            nc, mrs, dino_str, cost,
        )
        return result


# ------------------------------------------------------------------
# Individual metric functions
# ------------------------------------------------------------------

def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Bidirectional mean squared nearest-neighbour distance.
    Lower is better. On a unit sphere, <0.005 is excellent.
    """
    tree_b = KDTree(pts_b)
    tree_a = KDTree(pts_a)
    dist_ab, _ = tree_b.query(pts_a, k=1, workers=-1)
    dist_ba, _ = tree_a.query(pts_b, k=1, workers=-1)
    return float(np.mean(dist_ab ** 2) + np.mean(dist_ba ** 2))


def fscore(pts_gt: np.ndarray, pts_pred: np.ndarray, tau: float = 0.02) -> float:
    """
    F-Score at threshold τ.
    Precision = fraction of pred points within τ of any GT point.
    Recall    = fraction of GT points within τ of any pred point.
    Returns harmonic mean (F1). Range [0, 1]; >0.85 is good.
    """
    tree_gt = KDTree(pts_gt)
    tree_pred = KDTree(pts_pred)

    dist_pred_to_gt, _ = tree_gt.query(pts_pred, k=1, workers=-1)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt, k=1, workers=-1)

    precision = float(np.mean(dist_pred_to_gt < tau))
    recall = float(np.mean(dist_gt_to_pred < tau))

    if precision + recall < 1e-9:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def hausdorff_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance: max of the two one-sided Hausdorff distances.
    Catches floating debris and outliers that CD averages away.
    """
    tree_b = KDTree(pts_b)
    tree_a = KDTree(pts_a)
    dist_ab, _ = tree_b.query(pts_a, k=1, workers=-1)
    dist_ba, _ = tree_a.query(pts_b, k=1, workers=-1)
    return float(max(float(np.max(dist_ab)), float(np.max(dist_ba))))


def volume_iou(
    mesh_gt: trimesh.Trimesh,
    mesh_pred: trimesh.Trimesh,
    pitch_fraction: float = 1.0 / 64,
) -> float | None:
    """
    Volumetric IoU using voxel grids.
    Returns None if either mesh is not watertight (can't meaningfully fill volume).

    Approach: voxelise both at the same pitch, compute intersection/union
    of occupied voxel sets (rounded to a shared integer grid).
    """
    if not mesh_gt.is_watertight or not mesh_pred.is_watertight:
        log.debug("[metrics/viou] skipped — at least one mesh is not watertight")
        return None

    diag = float(np.linalg.norm(mesh_gt.bounding_box.extents))
    if diag < 1e-9:
        return None
    pitch = diag * pitch_fraction

    try:
        vox_gt = mesh_gt.voxelized(pitch=pitch).fill()
        vox_pred = mesh_pred.voxelized(pitch=pitch).fill()
    except Exception as exc:
        log.warning("[metrics/viou] voxelisation failed: %s", exc)
        return None

    pts_gt = vox_gt.points
    pts_pred = vox_pred.points

    if len(pts_gt) == 0 or len(pts_pred) == 0:
        return 0.0

    def to_set(pts: np.ndarray) -> set:
        rounded = np.round(pts / pitch).astype(np.int64)
        return set(map(tuple, rounded))

    set_gt = to_set(pts_gt)
    set_pred = to_set(pts_pred)

    intersection = len(set_gt & set_pred)
    union = len(set_gt | set_pred)
    return float(intersection / union) if union > 0 else 0.0


def normal_consistency(
    mesh_gt: trimesh.Trimesh,
    mesh_pred: trimesh.Trimesh,
    n: int = 10_000,
) -> float:
    """
    Mean |dot(normal_gt, normal_pred)| at nearest-neighbour surface samples.
    Uses absolute value to handle flipped orientations.
    Range [0, 1]; higher is better.
    """
    pts_gt, fi_gt = trimesh.sample.sample_surface(mesh_gt, n)
    normals_gt = mesh_gt.face_normals[fi_gt]

    pts_pred, fi_pred = trimesh.sample.sample_surface(mesh_pred, n)
    normals_pred = mesh_pred.face_normals[fi_pred]

    # For each GT sample, find nearest pred sample and compare normals
    tree_pred = KDTree(pts_pred)
    _, idx = tree_pred.query(pts_gt, k=1, workers=-1)

    matched = normals_pred[idx]
    dots = np.einsum("ij,ij->i", normals_gt, matched)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return float(np.mean(dots))


def morphing_readiness_score(
    mesh_pred: trimesh.Trimesh,
    normal_consistency_score: float,
) -> float:
    """
    Composite score predicting OT-morphing success. Range [0, 1].

    Weights tuned for the morphing use-case (geometric accuracy intentionally
    excluded — it is already captured by CD/F-Score and does not predict
    whether a mesh will survive MCF spherification and OT matching):

      Structural health (50%):
          struct = mean([is_watertight, is_manifold, is_single_component])
      Normal consistency (30%):
          NC (already in [0,1])
      Topology (20%):
          topo = 1.0 if genus==0 else max(0, 1 − genus×0.1)
    """

    # --- Structural health ---
    is_watertight = float(mesh_pred.is_watertight)

    # Manifold check: every edge shared by exactly 2 faces
    try:
        edges_sorted = mesh_pred.edges_sorted
        # Group edges; non-manifold edges appear ≠2 times
        unique, counts = np.unique(
            edges_sorted.view(np.dtype((np.void, edges_sorted.dtype.itemsize * 2))),
            return_counts=True,
        )
        # Edge-manifold condition: no edge shared by 3+ faces.
        # counts == 2 is too strict (fails for open meshes with boundary edges);
        # counts <= 2 correctly identifies only true non-manifold (fan) edges.
        is_manifold = float(np.all(counts <= 2))
    except Exception:
        # Fallback: approximation via trimesh topology
        is_manifold = float(mesh_pred.is_watertight)

    # Single connected component
    try:
        components = mesh_pred.split(only_watertight=False)
        is_single = float(len(components) == 1)
    except Exception:
        is_single = 1.0

    struct = (is_watertight + is_manifold + is_single) / 3.0

    # --- Topology (genus) ---
    try:
        euler = int(mesh_pred.euler_number)
        # For an orientable closed surface: V − E + F = 2 − 2g  →  g = (2 − euler) / 2
        genus = max(0, (2 - euler) // 2)
    except Exception:
        genus = 1  # pessimistic default

    topo = 1.0 if genus == 0 else max(0.0, 1.0 - genus * 0.1)

    mrs = 0.5 * struct + 0.3 * normal_consistency_score + 0.2 * topo
    return float(np.clip(mrs, 0.0, 1.0))


# Module-level cache: keyed by model_name → (processor, model)
_DINO_CACHE: dict[str, tuple] = {}


def dino_similarity(
    mesh_gt: trimesh.Trimesh,
    mesh_pred: trimesh.Trimesh,
    n_views: int = 8,
    size: int = 224,
    distance: float = 2.2,
    elev_deg: float = 20.0,
    model_name: str = "facebook/dinov2-base",
) -> float | None:
    """
    Mean cosine similarity of DINOv2 CLS features across matched viewpoints.

    Renders both meshes from the same n_views evenly-spaced azimuths and
    computes cosine similarity between the CLS token of each GT/pred pair,
    then returns the mean.  Range (−1, 1]; higher is better (>0.9 is good).

    Returns None when torch or transformers are not installed (install with
    `uv sync --extra dino`).
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image as _PILImage
    except ImportError:
        log.debug("[metrics/dino] torch or transformers not available — skipping")
        return None

    from .renderer import render_mesh_multiview

    azimuths = [i * (360.0 / n_views) for i in range(n_views)]

    try:
        views_gt = render_mesh_multiview(mesh_gt, azimuths, distance=distance, elev_deg=elev_deg, size=size)
        views_pred = render_mesh_multiview(mesh_pred, azimuths, distance=distance, elev_deg=elev_deg, size=size)
    except Exception as exc:
        log.warning("[metrics/dino] rendering failed: %s", exc)
        return None

    if model_name not in _DINO_CACHE:
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            _DINO_CACHE[model_name] = (processor, model)
            log.debug("[metrics/dino] loaded model %s", model_name)
        except Exception as exc:
            log.warning("[metrics/dino] model load failed: %s", exc)
            return None
    processor, model = _DINO_CACHE[model_name]

    sims = []
    with torch.no_grad():
        for img_gt, img_pred in zip(views_gt, views_pred):
            pil_gt = _PILImage.fromarray(img_gt)
            pil_pred = _PILImage.fromarray(img_pred)
            inp = processor(images=[pil_gt, pil_pred], return_tensors="pt")
            out = model(**inp)
            cls = out.last_hidden_state[:, 0]  # (2, D)
            cos = torch.nn.functional.cosine_similarity(cls[0:1], cls[1:2]).item()
            sims.append(cos)

    return float(np.mean(sims)) if sims else None


def cleanup_cost(mesh_pred: trimesh.Trimesh) -> int:
    """
    Count of morphing-blocking issues found in *mesh_pred*.  Range [0, 7].
    Lower is better (0 = clean mesh, ready for OT morphing).

    Issues checked (each contributes 1 to the count):
      1. Multiple connected components (floating debris)
      2. Not watertight (open boundary)
      3. Non-manifold edges (edge shared by >2 faces)
      4. Zero-area faces (area < 1e-10)
      5. Interior vertices (distance to centroid < 0.3 × mean distance)
      6. Euler number ≠ 2 (not genus-0)
      7. Too many vertices (>15 000, infeasible for dense OT)
    """
    cost = 0

    # 1. Multiple connected components
    try:
        components = mesh_pred.split(only_watertight=False)
        if len(components) > 1:
            cost += 1
    except Exception:
        pass

    # 2. Not watertight
    if not mesh_pred.is_watertight:
        cost += 1

    # 3. Non-manifold edges (shared by >2 faces)
    try:
        from collections import Counter
        edge_counts = Counter(
            map(tuple, np.sort(mesh_pred.edges, axis=1).tolist())
        )
        if any(c > 2 for c in edge_counts.values()):
            cost += 1
    except Exception:
        pass

    # 4. Zero-area faces
    try:
        if np.any(mesh_pred.area_faces < 1e-10):
            cost += 1
    except Exception:
        pass

    # 5. Interior vertices
    try:
        centroid = mesh_pred.centroid
        dists = np.linalg.norm(mesh_pred.vertices - centroid, axis=1)
        thresh = dists.mean() * 0.3
        if np.any(dists < thresh):
            cost += 1
    except Exception:
        pass

    # 6. Euler number ≠ 2
    try:
        if int(mesh_pred.euler_number) != 2:
            cost += 1
    except Exception:
        pass

    # 7. Too many vertices for dense OT
    if len(mesh_pred.vertices) > 15_000:
        cost += 1

    return cost


# ------------------------------------------------------------------
# Mesh utilities
# ------------------------------------------------------------------

def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not load Trimesh from {path}")
    return loaded


def _normalise_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre at centroid + scale to unit bounding sphere."""
    v = mesh.vertices.copy().astype(np.float64)
    v -= v.mean(axis=0)
    max_norm = np.linalg.norm(v, axis=1).max()
    if max_norm > 1e-9:
        v /= max_norm
    return trimesh.Trimesh(vertices=v, faces=mesh.faces.copy(), process=False)


def _sample_points(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
