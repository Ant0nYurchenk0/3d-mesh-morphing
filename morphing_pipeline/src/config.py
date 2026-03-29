"""
Typed configuration dataclasses for the morphing pipeline.

Usage:
    cfg = MorphingConfig.from_yaml(Path("config.yaml"))
    cfg.render.width               # 1024
    cfg.image_generation.model     # "gpt-image-1"
    cfg.pipeline.api_delay_seconds # 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from shared.config import RenderConfig  # noqa: F401 — re-exported for callers


@dataclass
class RepairConfig:
    """Configuration for the multi-stage mesh repair pipeline (node 6)."""
    target_faces: int = 10_000
    max_hole_size: int = 100
    min_component_diameter_pct: float = 10.0
    manifoldplus_depth: int = 8                   # ManifoldPlus voxel depth parameter
    manifoldplus_path: str = "manifoldplus"        # path / name of ManifoldPlus binary
    instant_meshes_path: str = "InstantMeshes"    # path / name of Instant Meshes binary


@dataclass
class ImageGenerationConfig:
    """Configuration for OpenAI GPT image generation/editing."""
    model: str = "gpt-image-1"
    size: str = "1024x1024"


@dataclass
class MorphConfig:
    """Configuration for mesh morphing (node 7)."""
    grid_resolution: int = 64   # SDF voxel grid resolution per axis (64³ ≈ 262k pts)
    n_frames: int = 30          # Number of interpolated frames (t = 0 … 1)


@dataclass
class DiffRendConfig:
    """Configuration for differential rendering-based mesh morphing (node 7b)."""
    n_steps: int = 4000          # total optimizer steps
    lr: float = 1e-3             # Adam learning rate
    n_frames: int = 40           # number of checkpoint frames to save
    n_views: int = 20            # camera viewpoints (split across two elevation rings)
    image_size: int = 256        # silhouette render resolution — keep ≤256 for speed
    n_sample_pts: int = 5000     # surface points sampled per mesh for Chamfer distance
    w_chamfer: float = 5.0       # weight: Chamfer distance (primary — must dominate silhouette)
    w_silhouette: float = 0.1    # weight: multi-view silhouette (secondary 2D guidance only)
    w_laplacian: float = 1.0     # weight: Laplacian smoothing (strong — prevents spiky deformations)
    w_normal: float = 0.1        # weight: normal consistency regularizer
    w_edge: float = 0.01         # weight: edge length regularizer (minimal)


@dataclass
class MorphingPipelineConfig:
    """Execution-level settings for the morphing pipeline."""
    api_delay_seconds: float = 2.0
    session_dir: str = "sessions"


@dataclass
class MorphingConfig:
    """Top-level typed configuration for the morphing pipeline."""
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    render: RenderConfig = field(default_factory=RenderConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    morph: MorphConfig = field(default_factory=MorphConfig)
    diff_rend: DiffRendConfig = field(default_factory=DiffRendConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    pipeline: MorphingPipelineConfig = field(default_factory=MorphingPipelineConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "MorphingConfig":
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "MorphingConfig":
        render_d = data.get("render", {})
        repair_d = data.get("repair", {})
        morph_d = data.get("morph", {})
        diff_rend_d = data.get("diff_rend", {})
        img_gen_d = data.get("image_generation", {})
        pipeline_d = data.get("pipeline", {})

        _render_fields = set(RenderConfig.__dataclass_fields__)
        _repair_fields = set(RepairConfig.__dataclass_fields__)
        _morph_fields = set(MorphConfig.__dataclass_fields__)
        _diff_rend_fields = set(DiffRendConfig.__dataclass_fields__)
        _img_gen_fields = set(ImageGenerationConfig.__dataclass_fields__)
        _pipeline_fields = set(MorphingPipelineConfig.__dataclass_fields__)

        return cls(
            models=data.get("models", {}),
            render=RenderConfig(**{k: v for k, v in render_d.items() if k in _render_fields}),
            repair=RepairConfig(**{k: v for k, v in repair_d.items() if k in _repair_fields}),
            morph=MorphConfig(**{k: v for k, v in morph_d.items() if k in _morph_fields}),
            diff_rend=DiffRendConfig(
                **{k: v for k, v in diff_rend_d.items() if k in _diff_rend_fields}
            ),
            image_generation=ImageGenerationConfig(
                **{k: v for k, v in img_gen_d.items() if k in _img_gen_fields}
            ),
            pipeline=MorphingPipelineConfig(
                **{k: v for k, v in pipeline_d.items() if k in _pipeline_fields}
            ),
        )
