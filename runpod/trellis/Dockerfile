# ============================================================
# TRELLIS — RunPod self-hosted inference pod
# ============================================================
# Model:   microsoft/TRELLIS-image-large
# Code:    github.com/microsoft/TRELLIS (+ submodules)
# GPU:     A100 40 GB recommended (16 GB absolute minimum)
# VRAM:    ~16 GB at default settings
#
# Build (from repo root — build context must include the TRELLIS submodule):
#   docker build --build-arg HF_TOKEN=<token> \
#     -t your-user/trellis:latest \
#     -f runpod/trellis/Dockerfile .
#
# WARNING: This build compiles several CUDA extensions (spconv, nvdiffrast,
# kaolin, diffoctreerast). Expect 30–60 min on first build.
# Use Docker layer caching — do not pass --no-cache unless necessary.
# ============================================================

# Use devel image for CUDA headers needed by extension compilation
FROM --platform=linux/amd64 pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        build-essential \
        cmake \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ── Clone TRELLIS with submodules ────────────────────────────
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /app/trellis
WORKDIR /app/trellis

# ── Python dependencies ──────────────────────────────────────
# Install xformers and flash-attn before requirements.txt to ensure
# CUDA-compatible wheels are used.
RUN pip install --no-cache-dir \
        "xformers==0.0.27.post2" \
        "flash-attn>=2.6.0" \
        --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

# ── Compile CUDA extensions ──────────────────────────────────
# TRELLIS's setup.sh installs these; replicated here for Docker.
# Each extension is in a submodule under extensions/.
RUN pip install --no-cache-dir \
        "spconv-cu124" \
        "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git"

# kaolin (needed for mesh operations)
RUN pip install --no-cache-dir \
        "kaolin @ https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu124/kaolin-0.17.0-cp310-cp310-linux_x86_64.whl"

# ── Gradio app dependencies ───────────────────────────────────
RUN pip install --no-cache-dir "gradio==4.44.0" "huggingface_hub[cli]"

# ── Model weights ────────────────────────────────────────────
ARG HF_TOKEN
RUN python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
snapshot_download("microsoft/TRELLIS-image-large", local_dir="/models/trellis", token=os.environ["HF_TOKEN"])
PYEOF

# ── Point app at pre-downloaded weights ──────────────────────
ENV TRELLIS_MODEL_ROOT=/models/trellis

# ── Expose Gradio port ───────────────────────────────────────
EXPOSE 8080

# ── Entry point ──────────────────────────────────────────────
# app.py is the Gradio demo included in the TRELLIS repo.
# It exposes /start_session + /generate_and_extract_glb, matching TRELLISClient.
CMD ["python", "app.py", "--server_name", "0.0.0.0", "--server_port", "8080"]
