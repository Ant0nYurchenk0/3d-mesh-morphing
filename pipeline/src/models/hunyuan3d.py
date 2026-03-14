"""
Hunyuan3D-2 image-to-3D client.

Space: tencent/Hunyuan3D-2
  https://huggingface.co/spaces/tencent/Hunyuan3D-2

Verified API signature for /shape_generation endpoint:
  shape_generation(
      caption: str = "",
      image: FileData,
      mv_image_front: FileData | None = None,
      mv_image_back:  FileData | None = None,
      mv_image_left:  FileData | None = None,
      mv_image_right: FileData | None = None,
      steps: int = 30,
      guidance_scale: float = 5.0,
      seed: int = 1234,
      octree_resolution: int = 256,
      check_box_rembg: bool = True,
      num_chunks: int = 8000,
      randomize_seed: bool = False,
  ) -> (file: filepath, html, stats, actual_seed)

The first element of the returned tuple is the output mesh file path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ImageTo3DClient


class Hunyuan3DClient(ImageTo3DClient):

    def _call_api(self, image_path: Path):
        from gradio_client import handle_file  # type: ignore

        return self.client.predict(
            caption="",
            image=handle_file(str(image_path)),
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=int(self._cfg.get("steps", 30)),
            guidance_scale=float(self._cfg.get("guidance_scale", 5.0)),
            seed=int(self._cfg.get("seed", 1234)),
            octree_resolution=int(self._cfg.get("octree_resolution", 256)),
            check_box_rembg=True,
            num_chunks=int(self._cfg.get("num_chunks", 8000)),
            randomize_seed=False,
            api_name=self.api_name,
        )

    def _extract_output_path(self, result: Any) -> Path:
        # Returns (file, html, stats, actual_seed) — take the first element
        if isinstance(result, (list, tuple)) and len(result) >= 1:
            return super()._extract_output_path(result[0])
        return super()._extract_output_path(result)
