"""
Mesh downloader / generator with three strategies (tried in order):

  Strategy A ("package"):   thingi10k Python package — semantic search by query.
                            Requires: uv sync --extra dataset
                            Heavy deps: polars, lagrange-open, datasets.

  Strategy B ("http"):      Direct HTTP download from ten-thousand-models.appspot.com
                            by Thingiverse ID. Fails with 404 if the ID is not in
                            the Thingi10K dataset — set thingiverse_id: null in
                            config.yaml to skip this step cleanly.

  Strategy C ("primitive"): Trimesh procedural mesh. Always works. Shapes:
                            gear, star, sphere, torus, cloud.
                            Activated automatically when A+B fail if
                            pipeline.use_primitive_fallback: true (default).

config.yaml controls:
  pipeline.download_strategy:      "auto" | "http_only" | "package_only" | "primitive_only"
  pipeline.use_primitive_fallback: true | false
  shapes.<name>.primitive_type:    gear | star | sphere | torus | cloud
  shapes.<name>.primitive_params:  {<kwargs>}

After obtaining a mesh (any strategy), it is:
  1. Loaded as trimesh.Trimesh (force="mesh")
  2. Normalised to unit bounding sphere, centred at origin
  3. Cached in pipeline/cache/{shape_name}/ to avoid re-downloading
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import requests
import trimesh

log = logging.getLogger(__name__)

_THINGI_DOWNLOAD_URL = "https://ten-thousand-models.appspot.com/download/id/{id}"
_CACHE_ROOT = Path(__file__).parent.parent / "cache"


class Downloader:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        pipe_cfg = cfg.get("pipeline", {})
        self.strategy: str = pipe_cfg.get("download_strategy", "http_only")
        self.use_primitive_fallback: bool = pipe_cfg.get("use_primitive_fallback", True)

    def download(self, shape_name: str, shape_cfg: dict) -> Path:
        """
        Return the local path to the normalised mesh for *shape_name*.
        Downloads/generates and caches on first call; returns cached path on subsequent calls.
        """
        cached = self._find_cached(shape_name)
        if cached is not None:
            log.info("[downloader] cache hit: %s -> %s", shape_name, cached)
            return cached

        log.info("[downloader] acquiring %s (strategy=%s)", shape_name, self.strategy)

        # --- primitive_only: skip all downloads ---
        if self.strategy == "primitive_only":
            return self._primitive_and_cache(shape_name, shape_cfg)

        raw_path: Path | None = None
        last_exc: Exception | None = None

        # Strategy A: thingi10k package
        if self.strategy in ("auto", "package_only"):
            try:
                raw_path = self._download_package(shape_name, shape_cfg)
            except Exception as exc:
                last_exc = exc
                log.warning("[downloader] package strategy failed for %s: %s", shape_name, exc)
                if self.strategy == "package_only":
                    raise

        # Strategy B: HTTP by Thingiverse ID
        if raw_path is None and self.strategy in ("auto", "http_only"):
            tid = shape_cfg.get("thingiverse_id")
            if tid is None:
                log.info(
                    "[downloader] thingiverse_id is null for '%s' - skipping HTTP download",
                    shape_name,
                )
            else:
                try:
                    raw_path = self._download_http(shape_name, shape_cfg)
                except Exception as exc:
                    last_exc = exc
                    log.warning(
                        "[downloader] HTTP download failed for %s (id=%s): %s",
                        shape_name, tid, exc,
                    )

        # Strategy C: trimesh primitive fallback
        if raw_path is None:
            if self.use_primitive_fallback:
                log.info(
                    "[downloader] using primitive fallback for '%s' "
                    "(no valid Thingi10K ID or download failed)",
                    shape_name,
                )
                return self._primitive_and_cache(shape_name, shape_cfg)
            else:
                raise RuntimeError(
                    f"All download strategies failed for '{shape_name}' and "
                    f"use_primitive_fallback is false. Last error: {last_exc}"
                ) from last_exc

        normalised = self._normalise_and_cache(shape_name, raw_path)
        if raw_path != normalised and raw_path.exists():
            raw_path.unlink(missing_ok=True)
        return normalised

    # ------------------------------------------------------------------
    # Strategy A -- thingi10k package
    # ------------------------------------------------------------------

    def _download_package(self, shape_name: str, shape_cfg: dict) -> Path:
        try:
            import thingi10k  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "thingi10k package not installed. Run: uv sync --extra dataset"
            ) from exc

        query = shape_cfg.get("search_query", shape_name)
        log.info("[downloader/package] searching for '%s'", query)
        entries = list(thingi10k.dataset(query=query))
        if not entries:
            raise ValueError(f"No thingi10k results for query: '{query}'")

        entry = entries[0]
        log.info("[downloader/package] found entry: %s", entry.get("file_id", "?"))
        vertices, facets = thingi10k.load_file(entry["file_path"])
        mesh = trimesh.Trimesh(vertices=vertices, faces=facets, process=False)

        tmp_path = _CACHE_ROOT / shape_name / f"{shape_name}_raw.off"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(tmp_path))
        return tmp_path

    # ------------------------------------------------------------------
    # Strategy B -- direct HTTP download
    # ------------------------------------------------------------------

    def _download_http(self, shape_name: str, shape_cfg: dict) -> Path:
        tid = shape_cfg["thingiverse_id"]
        url = _THINGI_DOWNLOAD_URL.format(id=tid)
        log.info("[downloader/http] GET %s", url)

        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()

        ext = _ext_from_response(resp) or shape_cfg.get("file_ext", "stl")
        tmp_path = _CACHE_ROOT / shape_name / f"{shape_name}_raw.{ext}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        log.info("[downloader/http] saved %d bytes to %s", tmp_path.stat().st_size, tmp_path)
        return tmp_path

    # ------------------------------------------------------------------
    # Strategy C -- trimesh primitive
    # ------------------------------------------------------------------

    def _primitive_and_cache(self, shape_name: str, shape_cfg: dict) -> Path:
        primitive_type = shape_cfg.get("primitive_type", shape_name)
        params = shape_cfg.get("primitive_params") or {}
        log.info("[downloader/primitive] generating '%s' (type=%s)", shape_name, primitive_type)

        mesh = _make_primitive(primitive_type, params)
        mesh = _normalise_mesh(mesh)

        cache_dir = _CACHE_ROOT / shape_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{shape_name}.off"
        mesh.export(str(out_path))
        log.info("[downloader/primitive] cached: %s (%d faces)", out_path, len(mesh.faces))
        return out_path

    # ------------------------------------------------------------------
    # Mesh normalisation & caching
    # ------------------------------------------------------------------

    def _normalise_and_cache(self, shape_name: str, raw_path: Path) -> Path:
        log.info("[downloader] normalising mesh from %s", raw_path)
        mesh = _load_as_single_mesh(raw_path)
        mesh = _normalise_mesh(mesh)

        cache_dir = _CACHE_ROOT / shape_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{shape_name}.off"
        mesh.export(str(out_path))
        log.info("[downloader] cached normalised mesh: %s", out_path)
        return out_path

    def _find_cached(self, shape_name: str) -> Path | None:
        for ext in (".off", ".stl", ".obj", ".glb", ".ply"):
            p = _CACHE_ROOT / shape_name / f"{shape_name}{ext}"
            if p.exists():
                return p
        return None


# ------------------------------------------------------------------
# Trimesh primitive generators
# ------------------------------------------------------------------

def _make_primitive(primitive_type: str, params: dict) -> trimesh.Trimesh:
    """Create a trimesh primitive by type name."""
    makers = {
        "sphere": _make_sphere,
        "torus": _make_torus,
        "gear": _make_gear,
        "star": _make_star,
        "cloud": _make_cloud,
    }
    fn = makers.get(primitive_type.lower())
    if fn is None:
        raise ValueError(
            f"Unknown primitive_type '{primitive_type}'. "
            f"Available: {sorted(makers.keys())}"
        )
    return fn(params)


def _make_sphere(params: dict) -> trimesh.Trimesh:
    """Subdivided icosphere -- clean, watertight, genus-0."""
    subdivisions = params.get("subdivisions", 3)
    return trimesh.creation.icosphere(subdivisions=subdivisions)


def _make_torus(params: dict) -> trimesh.Trimesh:
    """Standard torus (donut)."""
    major = params.get("major_radius", 1.0)
    minor = params.get("minor_radius", 0.35)
    return trimesh.creation.torus(major_radius=major, minor_radius=minor)


def _make_gear(params: dict) -> trimesh.Trimesh:
    """
    10-tooth spur gear created by extruding a 2D gear polygon.
    Watertight, genus-0, clearly recognisable as a gear from any angle.
    """
    from shapely.geometry import Polygon  # bundled with trimesh[easy]

    n_teeth = params.get("n_teeth", 10)
    outer_r = params.get("outer_radius", 1.0)
    inner_r = params.get("inner_radius", 0.72)
    height = params.get("height", 0.4)

    angles_outer = np.linspace(0, 2 * np.pi, n_teeth, endpoint=False)
    angles_inner = angles_outer + np.pi / n_teeth

    pts = np.empty((2 * n_teeth, 2))
    pts[0::2] = np.column_stack([outer_r * np.cos(angles_outer), outer_r * np.sin(angles_outer)])
    pts[1::2] = np.column_stack([inner_r * np.cos(angles_inner), inner_r * np.sin(angles_inner)])

    poly = Polygon(pts)
    return trimesh.creation.extrude_polygon(poly, height)


def _make_star(params: dict) -> trimesh.Trimesh:
    """
    5-pointed star extruded into a 3D solid.
    Watertight and clearly star-shaped from the top-down view models prefer.
    """
    from shapely.geometry import Polygon

    n_points = params.get("n_points", 5)
    outer_r = params.get("outer_radius", 1.0)
    inner_r = params.get("inner_radius", 0.4)
    height = params.get("height", 0.3)

    # Start at top (-pi/2) so the star points straight up
    angles_outer = np.linspace(
        -np.pi / 2,
        -np.pi / 2 + 2 * np.pi,
        n_points,
        endpoint=False,
    )
    angles_inner = angles_outer + np.pi / n_points

    pts = np.empty((2 * n_points, 2))
    pts[0::2] = np.column_stack([outer_r * np.cos(angles_outer), outer_r * np.sin(angles_outer)])
    pts[1::2] = np.column_stack([inner_r * np.cos(angles_inner), inner_r * np.sin(angles_inner)])

    poly = Polygon(pts)
    return trimesh.creation.extrude_polygon(poly, height)


def _make_cloud(params: dict) -> trimesh.Trimesh:
    """
    Cloud shape: icosphere with radial sinusoidal noise on vertex positions.
    Creates a recognisably bumpy / organic silhouette without multi-body union.
    Watertight and genus-0.
    """
    noise_scale = params.get("noise_scale", 0.18)
    subdivisions = params.get("subdivisions", 4)

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    v = mesh.vertices.copy()

    rng = np.random.default_rng(42)
    directions = v / np.linalg.norm(v, axis=1, keepdims=True)

    # Multi-frequency noise for a more cloud-like silhouette
    noise = np.zeros(len(v))
    for freq in [2.0, 4.0, 8.0]:
        noise += rng.uniform(-1, 1, len(v)) / freq
    noise *= noise_scale

    v = v + directions * noise[:, np.newaxis]
    return trimesh.Trimesh(vertices=v, faces=mesh.faces.copy(), process=False)


# ------------------------------------------------------------------
# Mesh loading / normalisation utilities
# ------------------------------------------------------------------

def _load_as_single_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh file as a Trimesh, concatenating all geometries in a Scene."""
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
        raise ValueError(f"Could not load a Trimesh from {path} (got {type(loaded)})")

    if len(loaded.faces) == 0:
        raise ValueError(f"Mesh from {path} has no faces")

    return loaded


def _normalise_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Centre at centroid + scale to unit bounding sphere.
    Mirrors the normalisation in Smooth Shape Blends/run.py lines 203-206.
    """
    v = mesh.vertices.copy().astype(np.float64)
    v -= v.mean(axis=0)
    max_norm = np.linalg.norm(v, axis=1).max()
    if max_norm > 1e-9:
        v /= max_norm
    return trimesh.Trimesh(vertices=v, faces=mesh.faces.copy(), process=False)


def _ext_from_response(resp: requests.Response) -> str | None:
    """Extract file extension from Content-Disposition header."""
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename=["\']?([^"\';\s]+)', cd)
    if m:
        return Path(m.group(1)).suffix.lstrip(".").lower() or None
    ct = resp.headers.get("Content-Type", "")
    _ct_map = {
        "model/stl": "stl",
        "model/obj": "obj",
        "application/octet-stream": None,
    }
    return _ct_map.get(ct.split(";")[0].strip())
