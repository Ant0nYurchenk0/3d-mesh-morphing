"""
Session and artifact directory management.

Every pipeline run creates a timestamped session folder:
    sessions/YYYY-MM-DD_HHMMSS_{uuid8}/

Under it, each (shape, model) pair gets its own sub-folder:
    sessions/<ts>/gear_trellis/
        original.stl       ← downloaded + normalised GT mesh
        render.png         ← 1024×1024 canonical render
        reconstructed.glb  ← model output (converted to GLB)
        metrics.json       ← computed metrics + metadata

The session root also holds:
    summary.csv    ← aggregated results
    summary.md     ← markdown table
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


class ArtifactDir:
    """Manages files for a single (shape, model) pair."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save helpers — all return the destination Path
    # ------------------------------------------------------------------

    def save_mesh(self, src: Path, ext: str | None = None) -> Path:
        """Copy the original GT mesh into the artifact dir."""
        if ext is None:
            ext = src.suffix.lstrip(".")
        dest = self.path / f"original.{ext}"
        shutil.copy2(src, dest)
        return dest

    def save_render(self, img: np.ndarray) -> Path:
        """Save an H×W×3 uint8 numpy array as render.png."""
        dest = self.path / "render.png"
        Image.fromarray(img).save(dest)
        return dest

    def save_reconstructed(self, src: Path) -> Path:
        """Copy the model-output file as reconstructed.glb (or original ext)."""
        ext = src.suffix or ".glb"
        dest = self.path / f"reconstructed{ext}"
        shutil.copy2(src, dest)
        return dest

    def save_metrics(self, metrics: dict) -> Path:
        """Write metrics dict as pretty-printed JSON."""
        dest = self.path / "metrics.json"
        dest.write_text(json.dumps(metrics, indent=2, default=_json_default))
        return dest

    def save_error(self, error: str) -> Path:
        """Save an error placeholder metrics.json for a failed model call."""
        return self.save_metrics({"error": error})

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def load_metrics(self) -> dict:
        p = self.path / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}

    @property
    def render_path(self) -> Path:
        return self.path / "render.png"

    @property
    def reconstructed_path(self) -> Path | None:
        for ext in (".glb", ".obj", ".ply", ".stl", ".off"):
            p = self.path / f"reconstructed{ext}"
            if p.exists():
                return p
        return None

    @property
    def original_path(self) -> Path | None:
        for ext in (".stl", ".obj", ".off", ".glb", ".ply"):
            p = self.path / f"original{ext}"
            if p.exists():
                return p
        return None


class Session:
    """Represents a single pipeline run with a timestamped directory."""

    def __init__(self, base_dir: Path | str) -> None:
        base = Path(base_dir)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        uid = str(uuid.uuid4())[:8]
        self.dir = base / f"{ts}_{uid}"
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open_existing(cls, path: Path | str) -> "Session":
        """Open an existing session directory without creating a new timestamped one."""
        obj = object.__new__(cls)
        obj.dir = Path(path)
        if not obj.dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {obj.dir}")
        return obj

    # ------------------------------------------------------------------
    # Artifact directories
    # ------------------------------------------------------------------

    def artifact_dir(self, shape_name: str, model_name: str) -> ArtifactDir:
        """Return (creating if needed) the artifact dir for a shape+model pair."""
        key = f"{shape_name}_{model_name}" if model_name else shape_name
        return ArtifactDir(self.dir / key)

    def render_only_dir(self, shape_name: str) -> ArtifactDir:
        """Return the artifact dir used during --dry-run (no model)."""
        return ArtifactDir(self.dir / shape_name)

    # ------------------------------------------------------------------
    # Summary paths
    # ------------------------------------------------------------------

    @property
    def summary_csv_path(self) -> Path:
        return self.dir / "summary.csv"

    @property
    def summary_per_model_csv_path(self) -> Path:
        return self.dir / "summary_per_model.csv"

    @property
    def summary_md_path(self) -> Path:
        return self.dir / "summary.md"

    def __repr__(self) -> str:
        return f"Session(dir={self.dir})"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _json_default(obj):
    """Make numpy scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
