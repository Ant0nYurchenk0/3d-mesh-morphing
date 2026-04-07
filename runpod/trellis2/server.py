"""
TRELLIS.2 on-premise Gradio inference server.

API-compatible with the microsoft/TRELLIS.2 HF Space, so
TRELLIS2Client (shared/models/trellis2.py) works unchanged when
endpoint_url points here.

Two-step workflow (mirrors the HF Space):
  /image_to_3d — runs full TRELLIS.2 inference; stores outputs in Gradio
                 session state; returns an HTML preview (ignored by client)
  /extract_glb — reads session state; exports GLB;
                 returns (model_viewer_path, glb_download_path)

Environment variables:
  HF_TOKEN  — HuggingFace token (required for gated/private models)
  MODEL_ID  — HF repo to load (default: microsoft/TRELLIS.2-4B)
  HF_HOME   — weight cache directory (default: ~/.cache/huggingface)
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
MODEL_ID = os.environ.get("MODEL_ID", "microsoft/TRELLIS.2-4B")
PORT = int(os.environ.get("PORT", 7860))

_pipeline = None


def _load_pipeline() -> None:
    global _pipeline

    if HF_TOKEN:
        log.info("Logging in to HuggingFace Hub …")
        huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)

    # Download the full repo first so sub-model paths resolve locally
    local_path = huggingface_hub.snapshot_download(MODEL_ID)
    log.info("Model downloaded to: %s", local_path)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    _pipeline = Trellis2ImageTo3DPipeline.from_pretrained(local_path)
    _pipeline.cuda()
    log.info("Pipeline ready.")


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

def _image_to_3d(
    image,
    seed,
    resolution,
    ss_guidance_strength,
    ss_guidance_rescale,
    ss_sampling_steps,
    ss_rescale_t,
    shape_slat_guidance_strength,
    shape_slat_guidance_rescale,
    shape_slat_sampling_steps,
    shape_slat_rescale_t,
    tex_slat_guidance_strength,
    tex_slat_guidance_rescale,
    tex_slat_sampling_steps,
    tex_slat_rescale_t,
):
    """Run TRELLIS.2 inference; store mesh in session state."""
    # Map resolution string to pipeline_type: "512" → "512", "1024" → "1024_cascade", "1536" → "1536_cascade"
    _res = str(resolution).strip()
    pipeline_type = "1024_cascade" if _res == "1024" else ("1536_cascade" if _res == "1536" else _res)
    log.info("Inference  seed=%s  pipeline_type=%s", seed, pipeline_type)

    # TRELLIS.2 run() signature differs from TRELLIS:
    #   - no `formats` param (always returns an O-Voxel mesh)
    #   - resolution via `pipeline_type` not `resolution`
    #   - sampler param keys are `guidance_strength`/`guidance_rescale` (not cfg_*)
    #   - second stage key is `shape_slat_sampler_params` (not `slat_sampler_params`)
    mesh = _pipeline.run(
        image,
        seed=int(seed),
        preprocess_image=True,
        pipeline_type=pipeline_type,
        sparse_structure_sampler_params={
            "steps": int(ss_sampling_steps),
            "guidance_strength": float(ss_guidance_strength),
            "guidance_rescale": float(ss_guidance_rescale),
            "rescale_t": float(ss_rescale_t),
        },
        shape_slat_sampler_params={
            "steps": int(shape_slat_sampling_steps),
            "guidance_strength": float(shape_slat_guidance_strength),
            "guidance_rescale": float(shape_slat_guidance_rescale),
            "rescale_t": float(shape_slat_rescale_t),
        },
        tex_slat_sampler_params={
            "steps": int(tex_slat_sampling_steps),
            "guidance_strength": float(tex_slat_guidance_strength),
            "guidance_rescale": float(tex_slat_guidance_rescale),
            "rescale_t": float(tex_slat_rescale_t),
        },
    )[0]

    mesh.simplify(16777216)  # nvdiffrast face count limit

    log.info("Inference complete; storing mesh in session state.")
    return mesh, "<p>Generation complete.</p>"


def _extract_glb(mesh_state, decimation_target, texture_size):
    """Export GLB from session-state mesh; return (viewer_path, glb_path)."""
    if mesh_state is None:
        raise ValueError(
            "Session state is empty — call /image_to_3d before /extract_glb."
        )

    import o_voxel  # noqa: PLC0415

    log.info(
        "Exporting GLB  decimation_target=%s  texture_size=%s",
        decimation_target, texture_size,
    )

    glb_path = tempfile.mktemp(suffix=".glb")

    # TRELLIS.2 uses o_voxel.postprocess.to_glb() with mesh attributes
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh_state.vertices,
        faces=mesh_state.faces,
        attr_volume=mesh_state.attrs,
        coords=mesh_state.coords,
        attr_layout=mesh_state.layout,
        voxel_size=mesh_state.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(decimation_target),
        texture_size=int(texture_size),
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
    )

    glb.export(glb_path, extension_webp=True)
    log.info("GLB written: %s", glb_path)

    # Match HF Space return signature: (model_viewer_path, glb_download_path)
    return None, glb_path


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def _build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:

        _state = gr.State(value=None)

        # /image_to_3d ────────────────────────────────────────────────────
        _img                  = gr.Image(type="pil", label="image",                        visible=False)
        _seed                 = gr.Number(value=42,  label="seed",                         visible=False)
        _resolution           = gr.Textbox(value="1024", label="resolution",               visible=False)
        _ss_guid              = gr.Number(value=7.5,  label="ss_guidance_strength",        visible=False)
        _ss_rescale           = gr.Number(value=0.7,  label="ss_guidance_rescale",         visible=False)
        _ss_steps             = gr.Number(value=12,   label="ss_sampling_steps",           visible=False)
        _ss_rescale_t         = gr.Number(value=5.0,  label="ss_rescale_t",                visible=False)
        _shape_slat_guid      = gr.Number(value=7.5,  label="shape_slat_guidance_strength", visible=False)
        _shape_slat_rescale   = gr.Number(value=0.5,  label="shape_slat_guidance_rescale", visible=False)
        _shape_slat_steps     = gr.Number(value=12,   label="shape_slat_sampling_steps",   visible=False)
        _shape_slat_rescale_t = gr.Number(value=3.0,  label="shape_slat_rescale_t",        visible=False)
        _tex_slat_guid        = gr.Number(value=1.0,  label="tex_slat_guidance_strength",  visible=False)
        _tex_slat_rescale     = gr.Number(value=0.0,  label="tex_slat_guidance_rescale",   visible=False)
        _tex_slat_steps       = gr.Number(value=12,   label="tex_slat_sampling_steps",     visible=False)
        _tex_slat_rescale_t   = gr.Number(value=3.0,  label="tex_slat_rescale_t",          visible=False)

        _html_preview = gr.HTML(visible=False)

        _btn_gen = gr.Button(visible=False)
        _btn_gen.click(
            fn=_image_to_3d,
            inputs=[
                _img, _seed, _resolution,
                _ss_guid, _ss_rescale, _ss_steps, _ss_rescale_t,
                _shape_slat_guid, _shape_slat_rescale, _shape_slat_steps, _shape_slat_rescale_t,
                _tex_slat_guid, _tex_slat_rescale, _tex_slat_steps, _tex_slat_rescale_t,
            ],
            outputs=[_state, _html_preview],
            api_name="image_to_3d",
        )

        # /extract_glb ────────────────────────────────────────────────────
        _decimation = gr.Number(value=300000, label="decimation_target", visible=False)
        _tex_size   = gr.Number(value=2048,   label="texture_size",      visible=False)

        _viewer = gr.Model3D(visible=False)
        _glb    = gr.File(visible=False)

        _btn_ext = gr.Button(visible=False)
        _btn_ext.click(
            fn=_extract_glb,
            inputs=[_state, _decimation, _tex_size],
            outputs=[_viewer, _glb],
            api_name="extract_glb",
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