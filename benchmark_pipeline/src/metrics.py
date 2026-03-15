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
  7. DINOv2 Similarity             — mean cosine similarity across 8 matched viewpoints
                                     (requires: uv sync --extra dino; returns None otherwise)
  8. Cleanup Cost                  — count of morphing-blocking issues (0–7)

Public API:
    MetricsCalculator(cfg: MetricsConfig).compute_all(gt_path, pred_path) -> dict
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import KDTree

from .config import MetricsConfig
from shared.mesh_utils import load_mesh, normalise_mesh
from shared.mesh_quality import cleanup_cost  # noqa: F401 — re-exported; defined in shared

log = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Computes all 8 evaluation metrics for a (ground-truth, predicted) mesh pair.

    Parameters
    ----------
    cfg : MetricsConfig
        Typed metrics settings (sample count, F-score threshold, voxel pitch).
    """

    def __init__(self, cfg: MetricsConfig) -> None:
        self.n: int = cfg.n_sample_points
        self.tau: float = cfg.fscore_tau
        self.pitch_frac: float = cfg.voxel_pitch_fraction

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
            mesh_gt = load_mesh(gt_path)
        except Exception as exc:
            raise ValueError(f"Failed to load GT mesh from {gt_path}: {exc}") from exc

        try:
            mesh_pred = load_mesh(pred_path)
        except Exception as exc:
            raise ValueError(f"Failed to load predicted mesh from {pred_path}: {exc}") from exc

        mesh_gt = normalise_mesh(mesh_gt)
        mesh_pred = normalise_mesh(mesh_pred)

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
        mrs = morphing_readiness_score(mesh_pred=mesh_pred, normal_consistency_score=nc)

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
    """Bidirectional mean squared nearest-neighbour distance. Lower is better."""
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
    """Symmetric worst-case NN distance. Catches outliers/floating debris."""
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
    Volumetric IoU via voxel grids.
    Returns None if either mesh is not watertight.
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
    Uses absolute value to handle flipped orientations. Range [0, 1].
    """
    pts_gt, fi_gt = trimesh.sample.sample_surface(mesh_gt, n)
    normals_gt = mesh_gt.face_normals[fi_gt]

    pts_pred, fi_pred = trimesh.sample.sample_surface(mesh_pred, n)
    normals_pred = mesh_pred.face_normals[fi_pred]

    tree_pred = KDTree(pts_pred)
    _, idx = tree_pred.query(pts_gt, k=1, workers=-1)

    matched = normals_pred[idx]
    dots = np.einsum("ij,ij->i", normals_gt, matched)
    return float(np.mean(np.clip(np.abs(dots), 0.0, 1.0)))


def morphing_readiness_score(
    mesh_pred: trimesh.Trimesh,
    normal_consistency_score: float,
) -> float:
    """
    Composite score predicting OT-morphing success. Range [0, 1].

    Weights:
      Structural health (50%): mean([is_watertight, is_manifold, is_single_component])
      Normal consistency (30%): NC score in [0, 1]
      Topology (20%): 1.0 if genus==0, else max(0, 1 − genus×0.1)
    """
    is_watertight = float(mesh_pred.is_watertight)

    try:
        edges_sorted = mesh_pred.edges_sorted
        _, counts = np.unique(
            edges_sorted.view(np.dtype((np.void, edges_sorted.dtype.itemsize * 2))),
            return_counts=True,
        )
        is_manifold = float(np.all(counts <= 2))
    except Exception:
        is_manifold = float(mesh_pred.is_watertight)

    try:
        components = mesh_pred.split(only_watertight=False)
        is_single = float(len(components) == 1)
    except Exception:
        is_single = 1.0

    struct = (is_watertight + is_manifold + is_single) / 3.0

    try:
        genus = max(0, (2 - int(mesh_pred.euler_number)) // 2)
    except Exception:
        genus = 1

    topo = 1.0 if genus == 0 else max(0.0, 1.0 - genus * 0.1)
    mrs = 0.5 * struct + 0.3 * normal_consistency_score + 0.2 * topo
    return float(np.clip(mrs, 0.0, 1.0))


# Module-level cache: model_name → (processor, model)
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
    Mean cosine similarity of DINOv2 CLS features across n_views matched viewpoints.

    Returns None when torch or transformers are not installed
    (install with `uv sync --extra dino`).
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from PIL import Image as _PILImage
    except ImportError:
        log.debug("[metrics/dino] torch or transformers not available — skipping")
        return None

    from shared.renderer import render_mesh_multiview

    azimuths = [i * (360.0 / n_views) for i in range(n_views)]

    try:
        views_gt = render_mesh_multiview(
            mesh_gt, azimuths, distance=distance, elev_deg=elev_deg, size=size
        )
        views_pred = render_mesh_multiview(
            mesh_pred, azimuths, distance=distance, elev_deg=elev_deg, size=size
        )
    except Exception as exc:
        log.warning("[metrics/dino] rendering failed: %s", exc)
        return None

    if model_name not in _DINO_CACHE:
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            _DINO_CACHE[model_name] = (processor, model)
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
            cls = out.last_hidden_state[:, 0]
            cos = torch.nn.functional.cosine_similarity(cls[0:1], cls[1:2]).item()
            sims.append(cos)

    return float(np.mean(sims)) if sims else None


# cleanup_cost is defined in shared.mesh_quality and re-exported via the import above.


# ------------------------------------------------------------------
# Mesh sampling utility
# ------------------------------------------------------------------

def _sample_points(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float64)
