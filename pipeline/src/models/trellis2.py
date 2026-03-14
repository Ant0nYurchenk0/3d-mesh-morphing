"""
TRELLIS.2 image-to-3D client.

Space: microsoft/TRELLIS.2
  https://huggingface.co/spaces/microsoft/TRELLIS.2

Two-step workflow (state is session-bound server-side):
  Step 1 — /image_to_3d: generates SLAT, stores in gr.State server-side.
            Returns HTML multi-view preview (not a file).
  Step 2 — /extract_glb: reads stored state, exports GLB.
            Returns (model_viewer_path, glb_download_path).

TRELLIS.2 has a different API from TRELLIS v1:
  - No multiimages parameter
  - Adds: resolution, ss_guidance_rescale, ss_rescale_t
  - Adds: shape_slat_* and tex_slat_* guidance/sampling params
  - /extract_glb uses decimation_target (int face count) not mesh_simplify (float ratio)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ImageTo3DClient


def _extract_local_path(result: Any) -> str:
    """Pull the local filesystem path out of a gradio_client file result."""
    if isinstance(result, dict):
        return result.get("path") or result.get("tmp_path") or str(result)
    return str(result)


class TRELLIS2Client(ImageTo3DClient):

    def _call_api(self, image_path: Path):
        from gradio_client import handle_file  # type: ignore

        # Step 1: generate (state stored server-side, result is HTML preview)
        self.client.predict(
            image=handle_file(str(image_path)),
            seed=int(self._cfg.get("seed", 42)),
            resolution=str(self._cfg.get("resolution", "1024")),
            ss_guidance_strength=float(self._cfg.get("ss_guidance_strength", 7.5)),
            ss_guidance_rescale=float(self._cfg.get("ss_guidance_rescale", 0.7)),
            ss_sampling_steps=int(self._cfg.get("ss_sampling_steps", 12)),
            ss_rescale_t=float(self._cfg.get("ss_rescale_t", 5.0)),
            shape_slat_guidance_strength=float(self._cfg.get("shape_slat_guidance_strength", 7.5)),
            shape_slat_guidance_rescale=float(self._cfg.get("shape_slat_guidance_rescale", 0.5)),
            shape_slat_sampling_steps=int(self._cfg.get("shape_slat_sampling_steps", 12)),
            shape_slat_rescale_t=float(self._cfg.get("shape_slat_rescale_t", 3.0)),
            tex_slat_guidance_strength=float(self._cfg.get("tex_slat_guidance_strength", 1.0)),
            tex_slat_guidance_rescale=float(self._cfg.get("tex_slat_guidance_rescale", 0.0)),
            tex_slat_sampling_steps=int(self._cfg.get("tex_slat_sampling_steps", 12)),
            tex_slat_rescale_t=float(self._cfg.get("tex_slat_rescale_t", 3.0)),
            api_name="/image_to_3d",
        )

        # Step 2: extract GLB from server-side state (same session)
        return self.client.predict(
            decimation_target=int(self._cfg.get("decimation_target", 300000)),
            texture_size=int(self._cfg.get("texture_size", 2048)),
            api_name="/extract_glb",
        )

    def _extract_output_path(self, result: Any) -> Path:
        # /extract_glb returns (model_viewer_path, glb_download_path) — take [1]
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return super()._extract_output_path(result[1])
        return super()._extract_output_path(result)
