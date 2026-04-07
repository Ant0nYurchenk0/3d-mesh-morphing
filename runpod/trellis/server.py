"""
TRELLIS on-premise Gradio inference server.

API-compatible with the trellis-community/TRELLIS HF Space, so
TRELLISClient (shared/models/trellis.py) works unchanged when
endpoint_url points here.

Endpoints:
  /start_session            — no-op session initialiser (API compatibility)
  /generate_and_extract_glb — TRELLIS inference + GLB export
                              returns (video_path, glb_viewer_path, glb_download_path)

Environment variables:
  HF_TOKEN  — HuggingFace token (required for gated/private models)
  MODEL_ID  — HF repo to load (default: JeffreyXiang/TRELLIS-image-large)
  HF_HOME   — weight cache directory (default: ~/.cache/huggingface)
              mount a RunPod network volume here to persist weights across restarts
  PORT      — Gradio server port (default: 7860)
"""

from __future__ import annotations

import logging
import os
import tempfile

import gradio as gr
import huggingface_hub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MODEL_ID", "JeffreyXiang/TRELLIS-image-large")
PORT = int(os.environ.get("PORT", 7860))

_pipeline = None


def _load_pipeline() -> None:
    global _pipeline

    if HF_TOKEN:
        log.info("Logging in to HuggingFace Hub …")
        huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)

    log.info("Downloading / loading TRELLIS pipeline: %s …", MODEL_ID)
    from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: PLC0415

    _pipeline = TrellisImageTo3DPipeline.from_pretrained(MODEL_ID)
    _pipeline.cuda()
    log.info("Pipeline ready.")


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

def _start_session():
    """No-op; kept for API compatibility with the community HF Space."""
    pass


def _generate_and_extract_glb(
    image,            # PIL.Image  (via gr.Image(type="pil"))
    multiimages,      # list[file] — ignored; kept for API compatibility
    seed,
    ss_guidance_strength,
    ss_sampling_steps,
    slat_guidance_strength,
    slat_sampling_steps,
    multiimage_algo,  # str — ignored
    mesh_simplify,
    texture_size,
):
    from trellis.utils import postprocessing_utils  # noqa: PLC0415

    log.info(
        "Inference  seed=%s  ss_steps=%s/%s  slat_steps=%s/%s",
        seed, ss_sampling_steps, ss_guidance_strength,
        slat_sampling_steps, slat_guidance_strength,
    )

    outputs = _pipeline.run(
        image,
        seed=int(seed),
        formats=["gaussian", "mesh"],
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": int(ss_sampling_steps),
            "cfg_strength": float(ss_guidance_strength),
        },
        slat_sampler_params={
            "steps": int(slat_sampling_steps),
            "cfg_strength": float(slat_guidance_strength),
        },
    )

    glb_path = tempfile.mktemp(suffix=".glb")
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=float(mesh_simplify),
        texture_size=int(texture_size),
        verbose=False,
    )
    glb.export(glb_path)
    log.info("GLB written: %s", glb_path)

    # Match HF Space return signature: (video_path, glb_viewer_path, glb_download_path)
    # Only glb_download_path is used by TRELLISClient._extract_output_path
    return None, None, glb_path


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
# Component labels must match the keyword argument names used by TRELLISClient
# (gradio_client maps predict() kwargs to components by label).

def _build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:

        # /start_session ── no-op ─────────────────────────────────────────
        _s_btn = gr.Button(visible=False)
        _s_btn.click(
            fn=_start_session,
            inputs=[],
            outputs=[],
            api_name="start_session",
        )

        # /generate_and_extract_glb ──────────────────────────────────────
        _img         = gr.Image(type="pil",            label="image",                 visible=False)
        _multi       = gr.File(file_count="multiple",  label="multiimages",           visible=False)
        _seed        = gr.Number(value=42,             label="seed",                  visible=False)
        _ss_guid     = gr.Number(value=7.5,            label="ss_guidance_strength",  visible=False)
        _ss_steps    = gr.Number(value=12,             label="ss_sampling_steps",     visible=False)
        _slat_guid   = gr.Number(value=3.0,            label="slat_guidance_strength", visible=False)
        _slat_steps  = gr.Number(value=12,             label="slat_sampling_steps",   visible=False)
        _algo        = gr.Textbox(value="stochastic",  label="multiimage_algo",       visible=False)
        _simplify    = gr.Number(value=0.95,           label="mesh_simplify",         visible=False)
        _tex_size    = gr.Number(value=1024,           label="texture_size",          visible=False)

        _video   = gr.Video(    visible=False)
        _viewer  = gr.Model3D(  visible=False)
        _glb     = gr.File(     visible=False)

        _g_btn = gr.Button(visible=False)
        _g_btn.click(
            fn=_generate_and_extract_glb,
            inputs=[
                _img, _multi, _seed,
                _ss_guid, _ss_steps,
                _slat_guid, _slat_steps,
                _algo, _simplify, _tex_size,
            ],
            outputs=[_video, _viewer, _glb],
            api_name="generate_and_extract_glb",
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
        show_error=True,
    )
