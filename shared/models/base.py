"""
Abstract base class for all image-to-3D model clients.

Each model subclass must implement _call_api(image_path) and may optionally
override _extract_glb_path(result) if its Space returns a non-standard structure.

Retry logic, HF authentication, logging, and the gradio_client connection
are all handled here so individual model classes stay minimal.

Usage (from subclass):
    class TRELLISClient(ImageTo3DClient):
        def _call_api(self, image_path):
            from gradio_client import handle_file
            return self.client.predict(
                image=handle_file(str(image_path)),
                api_name=self.api_name,
            )
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ImageTo3DClient(ABC):
    """
    Base class for HuggingFace Space image-to-3D clients.

    Parameters
    ----------
    cfg : dict
        The model's config block from config.yaml
        (e.g. cfg["models"]["trellis"]).
    hf_token : str
        HuggingFace API token. Required for gated or private Spaces.
    """

    def __init__(self, cfg: dict, hf_token: str) -> None:
        self.space_id: str = cfg["space_id"]
        self.api_name: str = cfg.get("api_name", "/predict")
        self.output_format: str = cfg.get("output_format", "glb")
        self.hf_token: str = hf_token
        self.delay_seconds: float = float(cfg.get("delay_seconds", 2.0))
        self.max_retries: int = int(cfg.get("max_retries", 1))
        self._cfg = cfg
        self.__client = None

    # ------------------------------------------------------------------
    # Lazy gradio_client initialisation
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self.__client is None:
            from gradio_client import Client  # type: ignore
            log.info("[%s] connecting to Space %s", self.__class__.__name__, self.space_id)
            self.__client = Client(self.space_id, token=self.hf_token)
        return self.__client

    def view_api(self) -> None:
        """Print the Space's full API signature. Call this to verify api_name."""
        self.client.view_api()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reconstruct(self, image_path: Path) -> Path:
        """
        Reconstruct a 3D mesh from *image_path*.

        Retries once on failure (total 2 attempts) with an exponential
        back-off of 2× the configured delay.

        Returns the local filesystem path to the downloaded output file
        (typically a .glb, but may vary by model).
        """
        attempts = self.max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                log.info(
                    "[%s] calling %s (attempt %d/%d)",
                    self.__class__.__name__, self.api_name, attempt + 1, attempts
                )
                result = self._call_api(image_path)
                output_path = self._extract_output_path(result)
                log.info("[%s] output at: %s", self.__class__.__name__, output_path)
                return output_path

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "[%s] attempt %d failed: %s: %s",
                    self.__class__.__name__, attempt + 1,
                    type(exc).__name__, exc,
                )
                if attempt < attempts - 1:
                    sleep_t = self.delay_seconds * (2 ** attempt)
                    log.info("[%s] retrying in %.1fs...", self.__class__.__name__, sleep_t)
                    time.sleep(sleep_t)

        raise RuntimeError(
            f"[{self.__class__.__name__}] all {attempts} attempts failed. "
            f"Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Abstract / overrideable
    # ------------------------------------------------------------------

    @abstractmethod
    def _call_api(self, image_path: Path) -> Any:
        """
        Call the Gradio Space API and return the raw result.
        Use `from gradio_client import handle_file` for file inputs.
        """

    def _extract_output_path(self, result: Any) -> Path:
        """
        Extract a local filesystem Path from the gradio_client result.

        gradio_client downloads remote files automatically; the result
        contains the local temp path in various shapes depending on the
        Space and gradio version:

          - str              → direct file path
          - dict             → look for "value", "tmp_path", "path", "url" keys
          - list / tuple     → first element that looks like a file path
          - nested dict      → recursively search for a .glb / .obj / .ply path
        """
        path = self._find_path_in_result(result)
        if path is None:
            raise ValueError(
                f"[{self.__class__.__name__}] Could not extract a file path from result: "
                f"{repr(result)[:500]}"
            )
        p = Path(str(path))
        if not p.exists():
            raise FileNotFoundError(
                f"[{self.__class__.__name__}] gradio_client returned path that does not exist: {p}"
            )
        return p

    # ------------------------------------------------------------------
    # Path extraction helpers
    # ------------------------------------------------------------------

    def _find_path_in_result(self, result: Any, depth: int = 0) -> str | None:
        """Recursively search *result* for a file path (string or dict value)."""
        if depth > 5:
            return None

        if isinstance(result, str):
            if result and (Path(result).exists() or result.startswith("/")):
                return result
            return None

        if isinstance(result, dict):
            # Prefer keys that look like file references
            for key in ("value", "tmp_path", "path", "url", "model", "glb", "mesh", "output"):
                val = result.get(key)
                if val is not None:
                    found = self._find_path_in_result(val, depth + 1)
                    if found:
                        return found
            # Fall back: search all values
            for val in result.values():
                found = self._find_path_in_result(val, depth + 1)
                if found:
                    return found

        if isinstance(result, (list, tuple)):
            for item in result:
                found = self._find_path_in_result(item, depth + 1)
                if found:
                    return found

        return None
