"""
OpenAI GPT-based image generation and editing.

Used by two pipeline nodes:
  - enhance_image_node (node 1): identity regeneration — GPT reproduces the
    input image as faithfully as possible (preserves all details).
  - morph_image_node (node 4): guided morphing — GPT edits the base image
    according to a user-supplied text prompt.

Both operations use the OpenAI images.edit() endpoint which accepts an existing
image and a text prompt, and returns a GPT-generated image.

Supported models (via OPENAI_MODEL in config.yaml or MorphingConfig):
  - "gpt-image-1"  (default) — latest GPT image model, returns b64_json
  - "dall-e-2"               — older model, returns URL, requires mask for edit
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Prompt used by node 1 to ask GPT to reproduce the image without changes.
_ENHANCE_PROMPT = (
    "Reproduce this image exactly as it appears. Preserve every detail including "
    "colors, shapes, textures, lighting, composition, and style with the highest "
    "possible fidelity. Do not add, remove, or alter anything."
)


class GPTImageGenerator:
    """
    OpenAI image editing client for the morphing pipeline.

    Parameters
    ----------
    api_key : str
        OpenAI API key (from OPENAI_API_KEY environment variable).
    model : str
        OpenAI image model name. Default: "gpt-image-1".
    size : str
        Output image dimensions. Default: "1024x1024".
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-image-1",
        size: str = "1024x1024",
    ) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._size = size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enhance(self, image_path: Path, output_path: Path) -> Path:
        """
        Node 1 — identity regeneration.

        Sends the image to GPT with a prompt asking it to reproduce the image
        without any changes. Useful for normalising the input before morphing.

        Parameters
        ----------
        image_path : Path
            Source image (PNG/JPEG, ≤25 MB for gpt-image-1).
        output_path : Path
            Destination path for the generated PNG.

        Returns
        -------
        Path
            The saved output image path (same as *output_path*).
        """
        log.info("[GPTImageGenerator] enhance: %s -> %s", image_path, output_path)
        return self._edit(image_path, _ENHANCE_PROMPT, output_path)

    def morph(self, image_path: Path, prompt: str, output_path: Path) -> Path:
        """
        Node 4 — guided morphing.

        Sends the base image to GPT with the user's morphing prompt, returning
        an edited image that represents the desired visual transformation.

        Parameters
        ----------
        image_path : Path
            Base image to morph (PNG/JPEG).
        prompt : str
            Natural-language description of the desired change
            (loaded from the --prompt-file CLI argument).
        output_path : Path
            Destination path for the generated PNG.

        Returns
        -------
        Path
            The saved output image path (same as *output_path*).
        """
        log.info("[GPTImageGenerator] morph: %s + prompt -> %s", image_path, output_path)
        return self._edit(image_path, prompt, output_path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _edit(self, image_path: Path, prompt: str, output_path: Path) -> Path:
        """Call images.edit() and save the result to *output_path*."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(image_path, "rb") as img_f:
            response = self._client.images.edit(
                model=self._model,
                image=img_f,
                prompt=prompt,
                n=1,
                size=self._size,
            )

        img_data = response.data[0]

        if getattr(img_data, "b64_json", None):
            img_bytes = base64.b64decode(img_data.b64_json)
        elif getattr(img_data, "url", None):
            import requests
            img_bytes = requests.get(img_data.url, timeout=60).content
        else:
            raise ValueError(
                f"[GPTImageGenerator] unexpected response format: {repr(img_data)[:200]}"
            )

        output_path.write_bytes(img_bytes)
        log.info("[GPTImageGenerator] saved %d bytes -> %s", len(img_bytes), output_path)
        return output_path
