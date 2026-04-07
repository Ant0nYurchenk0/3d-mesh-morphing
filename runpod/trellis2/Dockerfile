# ── TRELLIS.2 on-premise inference server ───────────────────────────────────
#
# Build:
#   docker build --platform linux/amd64 -t your-user/trellis2-server:latest .
#
# Run (weights downloaded at first start; mount a volume to cache them):
#   docker run --gpus all -p 7860:7860 \
#     -e HF_TOKEN=hf_xxx \
#     -v /path/to/hf-cache:/hf-cache \
#     your-user/trellis2-server:latest
#
# Notes:
#   • CUDA extensions (cumesh, o-voxel, flexgemm, nvdiffrast, nvdiffrec,
#     flash-attn) are compiled from source during image build (~30 min).
#   • Model weights (~20 GB) are downloaded on first run into $HF_HOME.
#     Use a RunPod network volume at /hf-cache to persist them.
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# Target GPU architectures for CUDA extension compilation.
# Covers the most common RunPod GPUs: T4 / A100 / A10G / L40S / H100.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# System libraries required for headless GL, image processing, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        build-essential \
        ninja-build \
        libjpeg-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────

# 1. flash-attn (default attention backend for TRELLIS.2)
RUN pip install --no-cache-dir "flash-attn==2.7.3" --no-build-isolation

# 2. nvdiffrast v0.4.0 — differentiable rasteriser
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    MAX_JOBS=2 pip install --no-cache-dir --no-build-isolation /tmp/extensions/nvdiffrast

# 3. nvdiffrec — split-sum PBR renderer (JeffreyXiang fork, renderutils branch)
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    MAX_JOBS=2 pip install --no-cache-dir --no-build-isolation /tmp/extensions/nvdiffrec

# 4. TRELLIS.2 — clone the repo (includes trellis2 package + o-voxel submodule)
RUN git clone --recurse-submodules --depth 1 -b main \
        https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS2

# 5. cumesh — CUDA-accelerated mesh utilities
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh && \
    MAX_JOBS=2 pip install --no-cache-dir --no-build-isolation /tmp/extensions/CuMesh 

# 6. flexgemm — efficient sparse convolution via Triton
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM && \
    pip install --no-cache-dir --no-build-isolation /tmp/extensions/FlexGEMM

# 7. o-voxel — O-Voxel representation (from the cloned TRELLIS.2 repo)
RUN cp -r /opt/TRELLIS2/o-voxel /tmp/extensions/o-voxel && \
    pip install --no-cache-dir --no-build-isolation /tmp/extensions/o-voxel

# 8. utils3d — geometry utilities (pinned commit used by setup.sh)
RUN pip install --no-cache-dir \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Patch TRELLIS.2 image_feature_extractor: DINOv3ViTModel nests its encoder under
# `.model`, so layers are at `.model.layer`, not `.layer` directly.
# Bug in TRELLIS.2 relative to current transformers API.
RUN sed -i 's/self\.model\.layer/self.model.model.layer/g' \
        /opt/TRELLIS2/trellis2/modules/image_feature_extractor.py

# 9. TRELLIS.2 runtime dependencies (from setup.sh --basic)
RUN pip install --no-cache-dir \
        imageio \
        imageio-ffmpeg \
        tqdm \
        easydict \
        opencv-python-headless \
        trimesh \
        transformers \
        tensorboard \
        pandas \
        lpips \
        zstandard \
        kornia \
        timm \
        pillow-simd \
        safetensors \
        accelerate \
        einops \
        scipy



# 10. Gradio server + remaining runtime dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Application ──────────────────────────────────────────────────────────────

COPY server.py /app/server.py
WORKDIR /app

# Expose the trellis2 package to Python
ENV PYTHONPATH=/opt/TRELLIS2

# Model weights are downloaded at container startup into $HF_HOME.
ENV HF_HOME=/hf-cache

EXPOSE 7860

CMD ["python", "server.py"]