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
        img_gen_d = data.get("image_generation", {})
        pipeline_d = data.get("pipeline", {})

        _render_fields = set(RenderConfig.__dataclass_fields__)
        _repair_fields = set(RepairConfig.__dataclass_fields__)
        _img_gen_fields = set(ImageGenerationConfig.__dataclass_fields__)
        _pipeline_fields = set(MorphingPipelineConfig.__dataclass_fields__)

        return cls(
            models=data.get("models", {}),
            render=RenderConfig(**{k: v for k, v in render_d.items() if k in _render_fields}),
            repair=RepairConfig(**{k: v for k, v in repair_d.items() if k in _repair_fields}),
            image_generation=ImageGenerationConfig(
                **{k: v for k, v in img_gen_d.items() if k in _img_gen_fields}
            ),
            pipeline=MorphingPipelineConfig(
                **{k: v for k, v in pipeline_d.items() if k in _pipeline_fields}
            ),
        )
