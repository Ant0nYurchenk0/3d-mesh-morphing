"""
Mesh downloader / generator with three strategies (tried in order):

  Strategy A ("package"):   thingi10k Python package — semantic search by query.
                            Requires: uv sync --extra dataset

  Strategy B ("http"):      Download by Thingiverse thing ID.
                            First tries ten-thousand-models.appspot.com (thingi10k dataset,
                            10K curated file IDs). If that returns 404, falls back to the
                            Thingiverse API (requires THINGIVERSE_TOKEN env var).

  Strategy C ("primitive"): Trimesh procedural mesh. Always works offline.
                            Shapes: gear, star, sphere, torus, cloud.

config.yaml controls:
  pipeline.download_strategy:      "auto" | "http_only" | "package_only" | "primitive_only"
  pipeline.use_primitive_fallback: true | false
  shapes.<name>.primitive_type:    gear | star | sphere | torus | cloud
  shapes.<name>.primitive_params:  {<kwargs>}

All acquired meshes are normalised to unit bounding sphere and cached in
pipeline/cache/{shape_name}/ to avoid re-downloading.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests
import trimesh

from .config import BenchmarkConfig
from shared.mesh_utils import normalise_mesh

log = logging.getLogger(__name__)

_THINGI_DOWNLOAD_URL = "https://ten-thousand-models.appspot.com/download/id/{id}"
_THINGIVERSE_API_FILES_URL = "https://api.thingiverse.com/things/{thing_id}/files"
_CACHE_ROOT = Path(__file__).parent.parent / "cache"


class Downloader:
    def __init__(self, cfg: BenchmarkConfig) -> None:
        self._cfg = cfg
        self.strategy: str = cfg.pipeline.download_strategy
        self.use_primitive_fallback: bool = cfg.pipeline.use_primitive_fallback

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
                    "[downloader] thingiverse_id is null for '%s' — skipping HTTP download",
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
                    "[downloader] using primitive fallback for '%s'",
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
    # Strategy A — thingi10k package
    # ------------------------------------------------------------------

    def _download_package(self, shape_name: str, shape_cfg: dict) -> Path:
        try:
            import thingi10k  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "thingi10k package not installed. Run: uv sync --extra dataset"
            ) from exc

        # If a specific file_id is given, look it up directly — no CLIP needed.
        thingi10k_file_id = shape_cfg.get("thingi10k_file_id")
        if thingi10k_file_id is not None:
            log.info("[downloader/package] looking up file_id=%s for '%s'", thingi10k_file_id, shape_name)
            thingi10k.init()
            entry = next(
                (e for e in thingi10k.dataset() if e["file_id"] == thingi10k_file_id),
                None,
            )
            if entry is None:
                raise ValueError(
                    f"thingi10k_file_id={thingi10k_file_id} not found in dataset"
                )
        else:
            query = shape_cfg.get("search_query", shape_name)
            log.info("[downloader/package] searching for '%s' (requires CLIP)", query)
            entries = list(thingi10k.dataset(query=query))
            if not entries:
                raise ValueError(f"No thingi10k results for query: '{query}'")
            entry = entries[0]

        log.info("[downloader/package] found entry: file_id=%s, name=%r", entry.get("file_id", "?"), str(entry.get("name", ""))[:60])
        vertices, facets = thingi10k.load_file(entry["file_path"])
        mesh = trimesh.Trimesh(vertices=vertices, faces=facets, process=False)

        tmp_path = _CACHE_ROOT / shape_name / f"{shape_name}_raw.off"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(tmp_path))
        return tmp_path

    # ------------------------------------------------------------------
    # Strategy B — direct HTTP download
    # ------------------------------------------------------------------

    def _download_http(self, shape_name: str, shape_cfg: dict) -> Path:
        tid = shape_cfg["thingiverse_id"]

        # Try thingi10k HTTP endpoint first (works for the 10K curated file IDs).
        url = _THINGI_DOWNLOAD_URL.format(id=tid)
        log.info("[downloader/http] GET %s", url)
        resp = requests.get(url, timeout=60, stream=True)

        if resp.status_code == 404:
            log.info(
                "[downloader/http] thingi10k 404 for id=%s — trying Thingiverse API", tid
            )
            return self._download_thingiverse_api(shape_name, shape_cfg, tid)

        resp.raise_for_status()

        ext = _ext_from_response(resp) or shape_cfg.get("file_ext", "stl")
        tmp_path = _CACHE_ROOT / shape_name / f"{shape_name}_raw.{ext}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        log.info("[downloader/http] saved %d bytes to %s", tmp_path.stat().st_size, tmp_path)
        return tmp_path

    def _download_thingiverse_api(
        self, shape_name: str, shape_cfg: dict, thing_id: int | str
    ) -> Path:
        token = os.environ.get("THINGIVERSE_TOKEN")
        if not token:
            raise RuntimeError(
                f"Thingiverse thing {thing_id} is not in the thingi10k dataset. "
                "Set THINGIVERSE_TOKEN to download directly from the Thingiverse API."
            )

        headers = {"Authorization": f"Bearer {token}"}
        files_url = _THINGIVERSE_API_FILES_URL.format(thing_id=thing_id)
        log.info("[downloader/thingiverse-api] GET %s", files_url)

        resp = requests.get(files_url, headers=headers, timeout=30)
        resp.raise_for_status()
        files = resp.json()

        if not files:
            raise ValueError(f"Thingiverse thing {thing_id} has no downloadable files")

        target_ext = shape_cfg.get("file_ext", "stl").lower()
        file_info = next(
            (f for f in files if f.get("name", "").lower().endswith(f".{target_ext}")),
            files[0],
        )
        name = file_info.get("name", f"{shape_name}.{target_ext}")
        download_url = file_info["download_url"]
        log.info("[downloader/thingiverse-api] downloading file: %s", name)

        dl_resp = requests.get(download_url, headers=headers, timeout=60, stream=True)
        dl_resp.raise_for_status()

        ext = Path(name).suffix.lstrip(".").lower() or target_ext
        tmp_path = _CACHE_ROOT / shape_name / f"{shape_name}_raw.{ext}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as fh:
            for chunk in dl_resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        log.info(
            "[downloader/thingiverse-api] saved %d bytes to %s",
            tmp_path.stat().st_size, tmp_path,
        )
        return tmp_path

    # ------------------------------------------------------------------
    # Strategy C — trimesh primitive
    # ------------------------------------------------------------------

    def _primitive_and_cache(self, shape_name: str, shape_cfg: dict) -> Path:
        primitive_type = shape_cfg.get("primitive_type", shape_name)
        params = shape_cfg.get("primitive_params") or {}
        log.info("[downloader/primitive] generating '%s' (type=%s)", shape_name, primitive_type)

        mesh = _make_primitive(primitive_type, params)
        mesh = normalise_mesh(mesh)

        cache_dir = _CACHE_ROOT / shape_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / f"{shape_name}.off"
        mesh.export(str(out_path))
        log.info("[downloader/primitive] cached: %s (%d faces)", out_path, len(mesh.faces))
        return out_path

    # ------------------------------------------------------------------
    # Normalisation & caching
    # ------------------------------------------------------------------

    def _normalise_and_cache(self, shape_name: str, raw_path: Path) -> Path:
        log.info("[downloader] normalising mesh from %s", raw_path)
        from shared.mesh_utils import load_mesh
        mesh = load_mesh(raw_path)
        mesh = normalise_mesh(mesh)

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
    return trimesh.creation.icosphere(subdivisions=params.get("subdivisions", 3))


def _make_torus(params: dict) -> trimesh.Trimesh:
    return trimesh.creation.torus(
        major_radius=params.get("major_radius", 1.0),
        minor_radius=params.get("minor_radius", 0.35),
    )


def _make_gear(params: dict) -> trimesh.Trimesh:
    from shapely.geometry import Polygon

    n_teeth = params.get("n_teeth", 10)
    outer_r = params.get("outer_radius", 1.0)
    inner_r = params.get("inner_radius", 0.72)
    height = params.get("height", 0.4)

    angles_outer = np.linspace(0, 2 * np.pi, n_teeth, endpoint=False)
    angles_inner = angles_outer + np.pi / n_teeth

    pts = np.empty((2 * n_teeth, 2))
    pts[0::2] = np.column_stack([outer_r * np.cos(angles_outer), outer_r * np.sin(angles_outer)])
    pts[1::2] = np.column_stack([inner_r * np.cos(angles_inner), inner_r * np.sin(angles_inner)])

    return trimesh.creation.extrude_polygon(Polygon(pts), height)


def _make_star(params: dict) -> trimesh.Trimesh:
    from shapely.geometry import Polygon

    n_points = params.get("n_points", 5)
    outer_r = params.get("outer_radius", 1.0)
    inner_r = params.get("inner_radius", 0.4)
    height = params.get("height", 0.3)

    angles_outer = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi, n_points, endpoint=False)
    angles_inner = angles_outer + np.pi / n_points

    pts = np.empty((2 * n_points, 2))
    pts[0::2] = np.column_stack([outer_r * np.cos(angles_outer), outer_r * np.sin(angles_outer)])
    pts[1::2] = np.column_stack([inner_r * np.cos(angles_inner), inner_r * np.sin(angles_inner)])

    return trimesh.creation.extrude_polygon(Polygon(pts), height)


def _make_cloud(params: dict) -> trimesh.Trimesh:
    noise_scale = params.get("noise_scale", 0.18)
    subdivisions = params.get("subdivisions", 4)

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    v = mesh.vertices.copy()

    rng = np.random.default_rng(42)
    directions = v / np.linalg.norm(v, axis=1, keepdims=True)

    noise = np.zeros(len(v))
    for freq in [2.0, 4.0, 8.0]:
        noise += rng.uniform(-1, 1, len(v)) / freq
    noise *= noise_scale

    v = v + directions * noise[:, np.newaxis]
    return trimesh.Trimesh(vertices=v, faces=mesh.faces.copy(), process=False)


# ------------------------------------------------------------------
# HTTP utilities
# ------------------------------------------------------------------

def _ext_from_response(resp: requests.Response) -> str | None:
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
