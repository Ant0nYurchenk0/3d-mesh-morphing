"""
Hunyuan3D-2 on-premise Gradio inference server.

API-compatible with the tencent/Hunyuan3D-2 HF Space, so
Hunyuan3DClient (shared/models/hunyuan3d.py) works unchanged when
endpoint_url points here.

Endpoint:
  /shape_generation — image-to-3D shape generation
                      returns (mesh_file, html, stats, actual_seed)
                      Hunyuan3DClient uses result[0] as the output path.

Environment variables:
  HF_TOKEN           — HuggingFace token (required for gated/private models)
  MODEL_ID           — HF repo (default: tencent/Hunyuan3D-2)
  MODEL_SUBFOLDER    — subfolder inside the repo (default: hunyuan3d-dit-v2-0)
  HF_HOME            — weight cache directory (default: ~/.cache/huggingface)
  PORT               — Gradio server port (default: 7860)
"""

from __future__ import annotations

import logging
import os
import random
import tempfile

import gradio as gr
import huggingface_hub
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MODEL_ID", "tencent/Hunyuan3D-2")
MODEL_SUBFOLDER = os.environ.get("MODEL_SUBFOLDER", "hunyuan3d-dit-v2-0")
PORT = int(os.environ.get("PORT", 7860))

_pipeline = None


def _load_pipeline() -> None:
    global _pipeline

    if HF_TOKEN:
        log.info("Logging in to HuggingFace Hub …")
        huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)

    # Pre-download the full repo so sub-model paths resolve locally
    # and weights persist across container restarts via $HF_HOME
    local_path = huggingface_hub.snapshot_download(
        MODEL_ID,
        allow_patterns=[
        "hunyuan3d-dit-v2-0/*",
        "*.json",           # repo-level configs
        ]
        )
    log.info("Model downloaded to: %s", local_path)

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: PLC0415

    _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        local_path,
        subfolder=MODEL_SUBFOLDER,
        torch_dtype=torch.float16,
    )
    _pipeline.to("cuda")
    log.info("Pipeline ready.")


# ---------------------------------------------------------------------------
# Background removal helper
# ---------------------------------------------------------------------------

def _remove_background(image):
    """Return image with background removed using rembg (if installed)."""
    try:
        from rembg import remove  # noqa: PLC0415
        return remove(image)
    except ImportError:
        log.warning("rembg not installed; skipping background removal.")
        return image


# ---------------------------------------------------------------------------
# Endpoint handler
# ---------------------------------------------------------------------------

def _shape_generation(
    caption,
    image,
    mv_image_front,
    mv_image_back,
    mv_image_left,
    mv_image_right,
    steps,
    guidance_scale,
    seed,
    octree_resolution,
    check_box_rembg,
    num_chunks,
    randomize_seed,
):
    actual_seed = random.randint(0, 2**31 - 1) if randomize_seed else int(seed)
    log.info(
        "Inference  seed=%d  steps=%s  guidance_scale=%s  octree_resolution=%s",
        actual_seed, steps, guidance_scale, octree_resolution,
    )

    if check_box_rembg and image is not None:
        image = _remove_background(image)

    generator = torch.Generator(device="cuda").manual_seed(actual_seed)

    # Collect multi-view images (None entries are ignored by the pipeline).
    mv_images = [mv_image_front, mv_image_back, mv_image_left, mv_image_right]
    mv_images = [img for img in mv_images if img is not None] or None

    result = _pipeline(
        image=image,
        caption=caption or "",
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        octree_resolution=int(octree_resolution),
        num_chunks=int(num_chunks),
        **({"mv_images": mv_images} if mv_images is not None else {}),
    )

    # Pipeline returns a list; take the first mesh.
    mesh = result[0] if isinstance(result, (list, tuple)) else result

    glb_path = tempfile.mktemp(suffix=".glb")
    mesh.export(glb_path)
    log.info("GLB written: %s", glb_path)

    # Match HF Space return signature: (file, html, stats, actual_seed)
    return glb_path, "", "", actual_seed


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:

        # /shape_generation ───────────────────────────────────────────────
        _caption         = gr.Textbox(value="",     label="caption",           visible=False)
        _image           = gr.Image(type="pil",     label="image",             visible=False)
        _mv_front        = gr.Image(type="pil",     label="mv_image_front",    visible=False)
        _mv_back         = gr.Image(type="pil",     label="mv_image_back",     visible=False)
        _mv_left         = gr.Image(type="pil",     label="mv_image_left",     visible=False)
        _mv_right        = gr.Image(type="pil",     label="mv_image_right",    visible=False)
        _steps           = gr.Number(value=30,      label="steps",             visible=False)
        _guidance_scale  = gr.Number(value=5.0,     label="guidance_scale",    visible=False)
        _seed            = gr.Number(value=1234,    label="seed",              visible=False)
        _octree_res      = gr.Number(value=256,     label="octree_resolution", visible=False)
        _rembg           = gr.Checkbox(value=True,  label="check_box_rembg",   visible=False)
        _num_chunks      = gr.Number(value=8000,    label="num_chunks",        visible=False)
        _rand_seed       = gr.Checkbox(value=False, label="randomize_seed",    visible=False)

        _out_file  = gr.File(    visible=False)
        _out_html  = gr.HTML(    visible=False)
        _out_stats = gr.Textbox( visible=False)
        _out_seed  = gr.Number(  visible=False)

        _btn = gr.Button(visible=False)
        _btn.click(
            fn=_shape_generation,
            inputs=[
                _caption, _image,
                _mv_front, _mv_back, _mv_left, _mv_right,
                _steps, _guidance_scale, _seed,
                _octree_res, _rembg, _num_chunks, _rand_seed,
            ],
            outputs=[_out_file, _out_html, _out_stats, _out_seed],
            api_name="shape_generation",
        )

    return demo


if __name__ == "__main__":
    _load_pipeline()
    demo = _build_demo()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
    )