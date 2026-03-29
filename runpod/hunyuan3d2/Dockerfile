# ============================================================
# Hunyuan3D-2 — RunPod self-hosted inference pod
# ============================================================
# Model:   tencent/Hunyuan3D-2 (shape generation, no texture)
# Package: hy3dgen (PyPI) — no GitHub clone required
# GPU:     RTX 4090 (24 GB) or A40 recommended; A100 for safety margin
# VRAM:    ~6 GB for shape-only inference
#
# Build:
#   docker build --build-arg HF_TOKEN=<token> \
#     -t your-user/hunyuan3d2:latest \
#     -f runpod/hunyuan3d2/Dockerfile .
#
# The model weights are downloaded during build and baked into the image,
# so the pod starts instantly with no download delay.
# To stream weights at runtime instead (smaller image), remove the
# RUN huggingface-cli download ... line and set HF_TOKEN at pod launch.
# ============================================================

FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────
# gradio pinned to match version used in the HF Space
RUN pip install --no-cache-dir \
        hy3dgen \
        "gradio==4.44.0" \
        "huggingface_hub[cli]"

# ── Model weights ────────────────────────────────────────────
# Downloaded at build time and baked into the image layer.
# Pass HF_TOKEN as a build arg; it is NOT stored in the final image.
ARG HF_TOKEN
RUN python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download("tencent/Hunyuan3D-2", local_dir="/models/hunyuan3d2", token=os.environ["HF_TOKEN"])
PYEOF

# ── App setup ────────────────────────────────────────────────
WORKDIR /app

# gradio_app.py is included in the weights repo alongside the model files
RUN cp /models/hunyuan3d2/gradio_app.py /app/gradio_app.py

# Point hy3dgen at the pre-downloaded weights directory
ENV HUNYUAN3D_MODEL_PATH=/models/hunyuan3d2

# ── Expose Gradio port ───────────────────────────────────────
EXPOSE 8080

# ── Entry point ──────────────────────────────────────────────
# --server_name 0.0.0.0 required so RunPod's proxy can reach the app
CMD ["python", "gradio_app.py", "--server_name", "0.0.0.0", "--server_port", "8080"]
