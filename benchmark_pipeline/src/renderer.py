"""
Mesh renderer — produces white-background PNG images.

Render spec:
  - Configurable resolution (default 1024×1024), white background
  - Configurable elevation/azimuth (default 20°/30°)
  - Object fills ~65% of frame (binary-search camera distance)
  - Soft three-point lighting: key (bright), fill (medium), back (rim)

Two backends, selected once at module import time:

  pyrender + OSMesa  — used inside Docker (Linux).
                       Requires: uv sync --extra render
                       Requires: PYOPENGL_PLATFORM=osmesa
                       apt deps: libosmesa6-dev libgl1-mesa-glx

  matplotlib Agg    — headless fallback for macOS dev.
                       Single LightSource approximation.
                       Activated automatically when pyrender is unavailable,
                       or by setting FORCE_MATPLOTLIB=1.

Public API:

    Renderer(cfg: RenderConfig).render(mesh) -> np.ndarray
        Returns a uint8 H×W×3 RGB array.

    render_mesh_multiview(mesh, azimuths, distance, elev_deg, size) -> list[np.ndarray]
        Module-level function used by dino_similarity in metrics.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import trimesh

from .config import RenderConfig

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Backend detection (once at module import)
# ------------------------------------------------------------------

_BACKEND: str


def _detect_backend() -> str:
    if os.environ.get("FORCE_MATPLOTLIB"):
        return "matplotlib"
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", pyopengl_platform)
    try:
        import pyrender  # noqa: F401
        import OpenGL    # noqa: F401
        return "pyrender"
    except (ImportError, Exception):
        return "matplotlib"


_BACKEND = _detect_backend()
log.info("[renderer] backend=%s", _BACKEND)


# ------------------------------------------------------------------
# Renderer class
# ------------------------------------------------------------------

class Renderer:
    """
    Encapsulates rendering configuration and the binary-search camera logic.

    Parameters
    ----------
    cfg : RenderConfig
        Typed render settings (resolution, camera, lighting).
    """

    def __init__(self, cfg: RenderConfig) -> None:
        self._cfg = cfg

    def render(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Render *mesh* to a uint8 H×W×3 RGB array using the stored RenderConfig.

        Binary-searches camera distance until the object fills cfg.fill_fraction
        of the frame (±cfg.fill_tolerance), up to cfg.fill_search_iters iterations.
        """
        cfg = self._cfg
        render_fn = _render_pyrender if _BACKEND == "pyrender" else _render_matplotlib
        light_cfg = {
            "key": cfg.key_light_intensity,
            "fill": cfg.fill_light_intensity,
            "back": cfg.back_light_intensity,
        }
        bg_u8 = tuple(int(c * 255) for c in cfg.bg_color)

        lo, hi = 0.3, 8.0
        img: np.ndarray | None = None

        for _ in range(cfg.fill_search_iters):
            dist = (lo + hi) / 2.0
            img = render_fn(mesh, cfg.width, cfg.height, cfg.elev_deg, cfg.azim_deg,
                            dist, cfg.bg_color, light_cfg)
            fill = _compute_fill_fraction(img, bg_u8)
            log.debug("[renderer] dist=%.3f fill=%.3f", dist, fill)
            if abs(fill - cfg.fill_fraction) <= cfg.fill_tolerance:
                break
            if fill < cfg.fill_fraction:
                hi = dist   # object too small → move closer
            else:
                lo = dist   # object too large → move farther

        if img is None:
            img = render_fn(mesh, cfg.width, cfg.height, cfg.elev_deg, cfg.azim_deg,
                            (lo + hi) / 2.0, cfg.bg_color, light_cfg)

        return img


# ------------------------------------------------------------------
# Module-level multiview helper (used by dino_similarity in metrics.py)
# ------------------------------------------------------------------

def render_mesh_multiview(
    mesh: trimesh.Trimesh,
    azimuths: list[float],
    distance: float = 2.2,
    elev_deg: float = 20.0,
    size: int = 224,
) -> list[np.ndarray]:
    """
    Render *mesh* from multiple azimuth angles at a fixed camera distance.

    Returns a list of uint8 H×W×3 RGB arrays (one per azimuth).

    Parameters
    ----------
    azimuths  : list of azimuth angles in degrees
    distance  : camera distance (2.2 works well for unit-bounding-sphere meshes)
    elev_deg  : camera elevation (degrees)
    size      : output image size in pixels (square)
    """
    from PIL import Image as _PILImage

    render_fn = _render_pyrender if _BACKEND == "pyrender" else _render_matplotlib
    bg = [1.0, 1.0, 1.0]
    light_cfg = {"key": 3.0, "fill": 1.5, "back": 1.0}

    views = []
    for azim in azimuths:
        img = render_fn(mesh, size, size, elev_deg, azim, distance, bg, light_cfg)
        if img.shape[0] != size or img.shape[1] != size:
            img = np.array(
                _PILImage.fromarray(img).resize((size, size), _PILImage.LANCZOS)
            )
        views.append(img)
    return views


# ------------------------------------------------------------------
# Backend: pyrender + OSMesa
# ------------------------------------------------------------------

def _render_pyrender(
    mesh: trimesh.Trimesh,
    width: int,
    height: int,
    elev_deg: float,
    azim_deg: float,
    distance: float,
    bg_color: list[float],
    light_cfg: dict[str, float],
) -> np.ndarray:
    import pyrender  # type: ignore

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(
        bg_color=[*bg_color, 1.0],
        ambient_light=[0.08, 0.08, 0.08],
    )
    scene.add(pr_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.radians(45.0), aspectRatio=width / height)
    scene.add(camera, pose=_look_at_pose(elev_deg, azim_deg, distance))

    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_cfg["key"]),
        pose=_look_at_pose(30, -45, 3.0),
    )
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_cfg["fill"]),
        pose=_look_at_pose(10, 120, 3.0),
    )
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_cfg["back"]),
        pose=_look_at_pose(-20, 210, 3.0),
    )

    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = r.render(scene)
    finally:
        r.delete()

    return color  # uint8 H×W×3


# ------------------------------------------------------------------
# Backend: matplotlib Agg (headless macOS fallback)
# ------------------------------------------------------------------

def _render_matplotlib(
    mesh: trimesh.Trimesh,
    width: int,
    height: int,
    elev_deg: float,
    azim_deg: float,
    distance: float,
    bg_color: list[float],
    light_cfg: dict[str, float],
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    lim = distance
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    ls = LightSource(azdeg=315, altdeg=45)
    ax.plot_trisurf(
        mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
        triangles=mesh.faces,
        color=(0.72, 0.72, 0.78),
        shade=True,
        lightsource=ls,
        linewidth=0,
        antialiased=False,
    )
    ax.view_init(elev=elev_deg, azim=azim_deg)
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(height, width, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


# ------------------------------------------------------------------
# Camera pose utilities
# ------------------------------------------------------------------

def _look_at_pose(elev_deg: float, azim_deg: float, distance: float) -> np.ndarray:
    """4×4 camera-to-world matrix for a camera at (elev, azim, distance) looking at origin."""
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)

    cx = distance * np.cos(elev) * np.cos(azim)
    cy = distance * np.cos(elev) * np.sin(azim)
    cz = distance * np.sin(elev)
    eye = np.array([cx, cy, cz], dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0])

    z_axis = eye - target
    z_norm = np.linalg.norm(z_axis)
    z_axis = z_axis / z_norm if z_norm >= 1e-9 else np.array([0.0, 0.0, 1.0])

    x_axis = np.cross(world_up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm if x_norm >= 1e-9 else np.array([1.0, 0.0, 0.0])

    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


# ------------------------------------------------------------------
# Fill fraction measurement
# ------------------------------------------------------------------

def _compute_fill_fraction(img: np.ndarray, bg_u8: tuple) -> float:
    """Fraction of the frame covered by non-background pixels."""
    mask = ~np.all(img == np.array(bg_u8, dtype=np.uint8), axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    h, w = img.shape[:2]
    bbox_h = int(ys.max()) - int(ys.min()) + 1
    bbox_w = int(xs.max()) - int(xs.min()) + 1
    return max(bbox_h / h, bbox_w / w)
