# ── Hunyuan3D-2 on-premise inference server ──────────────────────────────────
#
# Build:
#   docker build -t your-user/hunyuan3d2-server:latest .
#
# Run:
#   docker run --gpus all -p 7860:7860 \
#     -e HF_TOKEN=hf_xxx \
#     -v /runpod-volume:/workspace \
#     your-user/hunyuan3d2-server:latest
#
# Notes:
#   • Requires a GPU with ≥16 GB VRAM (shape only) or ≥24 GB (shape + texture).
#   • Model weights (~10 GB) are downloaded on first run into $HF_HOME.
#   • We only need shape generation, so texture CUDA extensions
#     (custom_rasterizer, differentiable_renderer) are NOT compiled.
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Limit parallel CUDA compilation jobs to avoid OOM during build
ENV MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential \
        ninja-build \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Hunyuan3D-2 ─────────────────────────────────────────────────────────────

# Clone the repo
RUN git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2.git /opt/Hunyuan3D-2

# Install the repo's own pinned requirements FIRST, then the package
RUN pip install --no-cache-dir -r /opt/Hunyuan3D-2/requirements.txt \
    && pip install --no-cache-dir -e /opt/Hunyuan3D-2

# ── Gradio server deps ──────────────────────────────────────────────────────

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Application ──────────────────────────────────────────────────────────────

COPY server.py /app/server.py
WORKDIR /app

# Persist weights across RunPod restarts via network volume
ENV HF_HOME=/workspace/hf-cache

EXPOSE 7860

CMD ["python", "server.py"]