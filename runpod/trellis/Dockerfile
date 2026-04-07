# ── TRELLIS on-premise inference server ─────────────────────────────────────
#
# Build (add --platform because RunPod pods are linux/amd64; required on Apple Silicon):
#   docker build --platform linux/amd64 -t your-user/trellis-server:latest .
#
# Run (weights downloaded at first start; mount a volume to cache them):
#   docker run --gpus all -p 7860:7860 \
#     -e HF_TOKEN=hf_xxx \
#     -v /path/to/hf-cache:/hf-cache \
#     your-user/trellis-server:latest
#
# Notes:
#   • flash-attn is compiled from source during image build (~20 min).
#     To speed this up, replace the flash-attn pip install line with a
#     pre-built wheel from https://github.com/Dao-MixedPrecision/flash-attention/releases
#   • Model weights (~10 GB) are downloaded on first run into $HF_HOME.
#     Use a RunPod network volume at /hf-cache to persist them.
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# Target GPU architectures for CUDA extension compilation.
# Required when building cross-platform (e.g. Apple Silicon → linux/amd64)
# because no GPU is available to auto-detect the architecture.
# Covers the most common RunPod GPUs: T4 / A100 / A10G / L40S / H100.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# System libraries required for headless GL and build tools
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

# ── Python dependencies ──────────────────────────────────────────────────────

# 1. xformers — memory-efficient attention (pre-built for torch 2.1.2 + cu121)
RUN pip install --no-cache-dir \
        "xformers==0.0.23.post1" \
        --index-url https://download.pytorch.org/whl/cu121

# 2. flash-attn — pin to 2.5.9.post1, the last version compatible with torch 2.1.x.
#    flash-attn ≥2.6 uses std::optional which clashes with torch 2.1's c10::optional.
RUN pip install --no-cache-dir "flash-attn==2.5.9.post1" --no-build-isolation

# 3. spconv — sparse convolutions (cu120 wheel works with CUDA 12.x)
RUN pip install --no-cache-dir spconv-cu120

# 4. kaolin — NVIDIA 3D deep learning library
RUN pip install --no-cache-dir \
        kaolin \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html

# 5. utils3d — geometry utilities; pin to the exact commit used by TRELLIS setup.sh
RUN pip install --no-cache-dir \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# 6. TRELLIS — clone the repo (no setup.py/pyproject.toml; import via PYTHONPATH)
RUN git clone --recurse-submodules --depth 1 \
        https://github.com/microsoft/TRELLIS.git /opt/TRELLIS

# 7. TRELLIS runtime dependencies — mirrors setup.sh --basic exactly, plus extras
#    setup.sh --basic: pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless
#                      scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista
#                      pymeshfix igraph transformers
#    extras: plyfile safetensors accelerate lpips diffusers (needed by TRELLIS pipeline)
RUN pip install --no-cache-dir \
        pillow \
        imageio \
        imageio-ffmpeg \
        tqdm \
        easydict \
        opencv-python-headless \
        scipy \
        ninja \
        rembg \
        onnxruntime \
        trimesh \
        open3d \
        xatlas \
        pyvista \
        pymeshfix \
        igraph \
        transformers \
        plyfile \
        safetensors \
        accelerate \
        lpips \
        diffusers

# 8. diff-gaussian-rasterization — mip-splatting Gaussian renderer (setup.sh --mipgaussian)
RUN git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting \
    && pip install --no-cache-dir /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ \
    && rm -rf /tmp/mip-splatting

# 9. nvdiffrast — differentiable rasterization (requires --no-build-isolation)
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git"

# 10. Gradio server + version pins (transformers<4.47, diffusers<0.30, numpy<2.0)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Application ──────────────────────────────────────────────────────────────

COPY server.py /app/server.py
WORKDIR /app

# Expose the TRELLIS package to Python
ENV PYTHONPATH=/opt/TRELLIS

# Model weights are downloaded at container startup into $HF_HOME.
# Override by mounting a volume: -v /host/hf-cache:/hf-cache
ENV HF_HOME=/hf-cache

EXPOSE 7860

CMD ["python", "server.py"]
