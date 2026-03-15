"""
TRELLIS image-to-3D client.

Space: trellis-community/TRELLIS  (community fork of microsoft/TRELLIS-image-large)
  https://huggingface.co/spaces/trellis-community/TRELLIS

This Space combines generation + GLB extraction into a single endpoint and
exposes /start_session to work around the session-directory bug (the original
microsoft/TRELLIS fails because demo.load never fires for API clients).

Two-step workflow:
  Step 1 — /start_session: creates the per-session TMP_DIR on the server.
            Takes no user arguments; must be called before generation.
  Step 2 — /generate_and_extract_glb: runs TRELLIS inference + exports GLB.
            Returns (video_path, glb_viewer_path, glb_download_path).

Both calls MUST use the same Client instance (same session hash).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import ImageTo3DClient

log = logging.getLogger(__name__)


class TRELLISClient(ImageTo3DClient):

    def _call_api(self, image_path: Path):
        from gradio_client import handle_file  # type: ignore

        # Step 1: init session directory on the server
        log.debug("[TRELLISClient] calling /start_session")
        self.client.predict(api_name="/start_session")

        # Step 2: generate 3D + extract GLB in one combined call
        log.debug("[TRELLISClient] calling /generate_and_extract_glb")
        result = self.client.predict(
            image=handle_file(str(image_path)),
            multiimages=[],
            seed=int(self._cfg.get("seed", 42)),
            ss_guidance_strength=float(self._cfg.get("ss_guidance_strength", 7.5)),
            ss_sampling_steps=int(self._cfg.get("ss_sampling_steps", 12)),
            slat_guidance_strength=float(self._cfg.get("slat_guidance_strength", 3.0)),
            slat_sampling_steps=int(self._cfg.get("slat_sampling_steps", 12)),
            multiimage_algo="stochastic",
            mesh_simplify=float(self._cfg.get("mesh_simplify", 0.95)),
            texture_size=int(self._cfg.get("texture_size", 1024)),
            api_name="/generate_and_extract_glb",
        )
        log.debug("[TRELLISClient] result: %r", result)
        return result

    def _extract_output_path(self, result: Any) -> Path:
        # Returns (video_path, glb_viewer_path, glb_download_path) — take [2]
        if isinstance(result, (list, tuple)) and len(result) >= 3:
            return super()._extract_output_path(result[2])
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            return super()._extract_output_path(result[1])
        return super()._extract_output_path(result)
