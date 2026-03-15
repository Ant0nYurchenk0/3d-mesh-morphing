"""
Typed configuration dataclasses for the image-to-3D evaluation pipeline.

Replaces raw dict access with typed, IDE-friendly attributes throughout the codebase.

Usage:
    cfg = BenchmarkConfig.from_yaml(Path("config.yaml"))
    cfg.render.width          # 1024
    cfg.metrics.fscore_tau    # 0.02
    cfg.pipeline.api_delay_seconds  # 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RenderConfig:
    width: int = 1024
    height: int = 1024
    elev_deg: float = 20.0
    azim_deg: float = 30.0
    fill_fraction: float = 0.65
    fill_tolerance: float = 0.05
    fill_search_iters: int = 8
    bg_color: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    key_light_intensity: float = 3.0
    fill_light_intensity: float = 1.5
    back_light_intensity: float = 1.0


@dataclass
class MetricsThresholds:
    chamfer_distance: dict[str, float] = field(
        default_factory=lambda: {"excellent": 0.005, "good": 0.01, "poor": 0.03}
    )
    f_score: dict[str, float] = field(
        default_factory=lambda: {"excellent": 0.90, "good": 0.85, "poor": 0.70}
    )
    morphing_readiness_score: dict[str, float] = field(
        default_factory=lambda: {"excellent": 0.80, "good": 0.65, "poor": 0.40}
    )

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            "chamfer_distance": self.chamfer_distance,
            "f_score": self.f_score,
            "morphing_readiness_score": self.morphing_readiness_score,
        }


@dataclass
class MetricsConfig:
    n_sample_points: int = 10_000
    fscore_tau: float = 0.02
    voxel_pitch_fraction: float = 1.0 / 64
    thresholds: MetricsThresholds = field(default_factory=MetricsThresholds)


@dataclass
class PipelineConfig:
    api_delay_seconds: float = 2.0
    max_retries: int = 1
    session_dir: str = "sessions"
    download_strategy: str = "http_only"
    use_primitive_fallback: bool = True


@dataclass
class BenchmarkConfig:
    shapes: dict[str, dict[str, Any]] = field(default_factory=dict)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    render: RenderConfig = field(default_factory=RenderConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> BenchmarkConfig:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkConfig:
        render_d = data.get("render", {})
        metrics_d = data.get("metrics", {})
        thr_d = metrics_d.get("thresholds", {})
        pipeline_d = data.get("pipeline", {})

        _render_fields = set(RenderConfig.__dataclass_fields__)
        _thr_fields = set(MetricsThresholds.__dataclass_fields__)
        _pipeline_fields = set(PipelineConfig.__dataclass_fields__)

        return cls(
            shapes=data.get("shapes", {}),
            models=data.get("models", {}),
            render=RenderConfig(**{k: v for k, v in render_d.items() if k in _render_fields}),
            metrics=MetricsConfig(
                n_sample_points=metrics_d.get("n_sample_points", 10_000),
                fscore_tau=metrics_d.get("fscore_tau", 0.02),
                voxel_pitch_fraction=metrics_d.get("voxel_pitch_fraction", 1.0 / 64),
                thresholds=MetricsThresholds(
                    **{k: v for k, v in thr_d.items() if k in _thr_fields}
                ),
            ),
            pipeline=PipelineConfig(
                **{k: v for k, v in pipeline_d.items() if k in _pipeline_fields}
            ),
        )
