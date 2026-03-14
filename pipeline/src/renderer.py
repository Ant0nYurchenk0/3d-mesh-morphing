"""
Mesh renderer — produces a 1024×1024 white-background PNG.

Render spec:
  - 1024×1024 pixels, white background
  - 20° elevation, 30° azimuth (configurable via config.yaml)
  - Object fills ~65% of frame (binary-search camera distance)
  - Soft three-point lighting: key (bright), fill (medium), back (rim)

Two backends, selected at module import time:

  pyrender + OSMesa  — used inside Docker (Linux).
                       Requires: uv sync --extra render
                       Requires: PYOPENGL_PLATFORM=osmesa (set in Dockerfile / .env)
                       apt deps: libosmesa6-dev libgl1-mesa-glx

  matplotlib Agg    — headless fallback for macOS dev.
                       Single LightSource approximation (not true three-point),
                       adequate for model input quality.
                       Activated automatically when pyrender is unavailable,
                       or by setting FORCE_MATPLOTLIB=1.

The public API is one function:

    render_mesh(mesh: trimesh.Trimesh, cfg: dict) -> np.ndarray
        Returns a uint8 H×W×3 RGB array (1024×1024×3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import trimesh

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Backend detection (at module load time)
# ------------------------------------------------------------------

_BACKEND: str


def _detect_backend() -> str:
    if os.environ.get("FORCE_MATPLOTLIB"):
        return "matplotlib"
    # Must set PYOPENGL_PLATFORM before importing pyrender/OpenGL
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", pyopengl_platform)
    try:
        import pyrender  # noqa: F401
        import OpenGL   # noqa: F401
        return "pyrender"
    except (ImportError, Exception):
        return "matplotlib"


_BACKEND = _detect_backend()
log.info("[renderer] backend=%s", _BACKEND)


# ------------------------------------------------------------------
# Public entry points
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
    Uses the same backend (pyrender or matplotlib) as render_mesh.

    Parameters
    ----------
    azimuths    : list of azimuth angles in degrees
    distance    : camera distance (2.2 works well for unit-bounding-sphere meshes)
    elev_deg    : camera elevation (degrees)
    size        : output image size in pixels (square); images are resized to
                  exactly size×size after rendering to guarantee uniform shape
    """
    from PIL import Image as _PILImage

    render_fn = _render_pyrender if _BACKEND == "pyrender" else _render_matplotlib
    bg = [1.0, 1.0, 1.0]
    light_cfg = dict(key=3.0, fill=1.5, back=1.0)

    views = []
    for azim in azimuths:
        img = render_fn(mesh, size, size, elev_deg, azim, distance, bg, light_cfg)
        # Guarantee exact size (tight_layout may shift matplotlib canvas by 1px)
        if img.shape[0] != size or img.shape[1] != size:
            img = np.array(
                _PILImage.fromarray(img).resize((size, size), _PILImage.LANCZOS)
            )
        views.append(img)
    return views


def render_mesh(mesh: trimesh.Trimesh, cfg: Any) -> np.ndarray:
    """
    Render *mesh* to a 1024×1024 uint8 RGB array.

    cfg is the full pipeline config dict (or a SimpleNamespace/AttrDict).
    Uses render settings from cfg["render"].
    """
    render_cfg = cfg.get("render", {}) if isinstance(cfg, dict) else cfg.render
    width = _get(render_cfg, "width", 1024)
    height = _get(render_cfg, "height", 1024)
    elev = _get(render_cfg, "elev_deg", 20.0)
    azim = _get(render_cfg, "azim_deg", 30.0)
    target_fill = _get(render_cfg, "fill_fraction", 0.65)
    fill_tol = _get(render_cfg, "fill_tolerance", 0.05)
    n_iters = _get(render_cfg, "fill_search_iters", 8)

    bg = _get(render_cfg, "bg_color", [1.0, 1.0, 1.0])
    key_i = _get(render_cfg, "key_light_intensity", 3.0)
    fill_i = _get(render_cfg, "fill_light_intensity", 1.5)
    back_i = _get(render_cfg, "back_light_intensity", 1.0)

    light_cfg = dict(key=key_i, fill=fill_i, back=back_i)

    render_fn = _render_pyrender if _BACKEND == "pyrender" else _render_matplotlib

    # Binary-search camera distance so object fills ~target_fill of frame
    bg_u8 = tuple(int(c * 255) for c in bg)
    lo, hi = 0.3, 8.0
    img = None

    for _ in range(n_iters):
        dist = (lo + hi) / 2.0
        img = render_fn(mesh, width, height, elev, azim, dist, bg, light_cfg)
        fill = _compute_fill_fraction(img, bg_u8)
        log.debug("[renderer] dist=%.3f fill=%.3f", dist, fill)
        if abs(fill - target_fill) <= fill_tol:
            break
        if fill < target_fill:
            hi = dist  # object too small → move camera closer
        else:
            lo = dist  # object too large → move camera farther

    if img is None:
        img = render_fn(mesh, width, height, elev, azim, (lo + hi) / 2.0, bg, light_cfg)

    return img


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

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.radians(45.0), aspectRatio=width / height)
    camera_pose = _look_at_pose(elev_deg, azim_deg, distance)
    scene.add(camera, pose=camera_pose)

    # Three-point lighting
    # Key: upper-left front
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_cfg["key"]),
        pose=_look_at_pose(30, -45, 3.0),
    )
    # Fill: right
    scene.add(
        pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_cfg["fill"]),
        pose=_look_at_pose(10, 120, 3.0),
    )
    # Back/rim: behind subject
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
    fig_w = width / dpi
    fig_h = height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    v, f = mesh.vertices, mesh.faces

    # Scale the axis limits to simulate camera distance
    # distance=1.0 → axis limit ±1; larger distance → zoom out (larger limits)
    lim = distance
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])

    ls = LightSource(azdeg=315, altdeg=45)
    ax.plot_trisurf(
        v[:, 0], v[:, 1], v[:, 2],
        triangles=f,
        color=(0.72, 0.72, 0.78),
        shade=True,
        lightsource=ls,
        linewidth=0,
        antialiased=False,
    )
    ax.view_init(elev=elev_deg, azim=azim_deg)
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Extract ARGB buffer → RGB
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(height, width, 4)
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


# ------------------------------------------------------------------
# Camera pose utilities
# ------------------------------------------------------------------

def _look_at_pose(elev_deg: float, azim_deg: float, distance: float) -> np.ndarray:
    """
    Compute a 4×4 camera-to-world matrix for a camera at spherical
    coordinates (elev_deg, azim_deg, distance) looking at the origin.
    """
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)

    # Camera position in Cartesian
    cx = distance * np.cos(elev) * np.cos(azim)
    cy = distance * np.cos(elev) * np.sin(azim)
    cz = distance * np.sin(elev)
    eye = np.array([cx, cy, cz], dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    world_up = np.array([0.0, 0.0, 1.0])

    z_axis = eye - target
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-9:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis /= z_norm

    x_axis = np.cross(world_up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-9:
        # Degenerate: camera is directly above/below
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis /= x_norm

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
    """
    Fraction of the frame covered by non-background pixels
    (measured as the larger of the bounding-box height/width ratios).
    Returns 0.0 if the mesh rendered as nothing.
    """
    mask = ~np.all(img == np.array(bg_u8, dtype=np.uint8), axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    h, w = img.shape[:2]
    bbox_h = int(ys.max()) - int(ys.min()) + 1
    bbox_w = int(xs.max()) - int(xs.min()) + 1
    return max(bbox_h / h, bbox_w / w)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
