"""
Shared render configuration dataclass.

Used by both benchmark_pipeline and morphing_pipeline so that the Renderer
class has a single typed configuration definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
