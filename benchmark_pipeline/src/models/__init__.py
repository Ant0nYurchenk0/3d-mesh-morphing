"""
Model client factory.

Usage:
    from src.models import get_model_client
    client = get_model_client("trellis", model_cfg)
    glb_path = client.reconstruct(render_png_path)

Implemented models: trellis, trellis2, hunyuan3d.
To add a new model: create src/models/<name>.py inheriting ImageTo3DClient,
then add it to _REGISTRY below.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from .base import ImageTo3DClient
from .hunyuan3d import Hunyuan3DClient
from .trellis import TRELLISClient
from .trellis2 import TRELLIS2Client

_REGISTRY: dict[str, type[ImageTo3DClient]] = {
    "trellis": TRELLISClient,
    "trellis2": TRELLIS2Client,
    "hunyuan3d": Hunyuan3DClient,
}


def get_model_client(model_name: str, model_cfg: dict) -> ImageTo3DClient:
    """Instantiate a model client by name, pulling HF_TOKEN from the environment."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Copy .env.example → .env and add your HuggingFace token."
        )
    cls = _REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(_REGISTRY.keys())}"
        )
    return cls(model_cfg, hf_token=hf_token)


__all__ = [
    "get_model_client",
    "ImageTo3DClient",
    "TRELLISClient",
    "TRELLIS2Client",
    "Hunyuan3DClient",
]
