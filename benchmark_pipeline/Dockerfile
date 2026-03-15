# ============================================================
# Image-to-3D Evaluation Pipeline — Docker image
# ============================================================
# Base: python:3.12-slim (Debian Bookworm)
#
# What this image includes:
#   - All pipeline Python dependencies (trimesh, scipy, gradio-client, etc.)
#   - pyrender + PyOpenGL with OSMesa for headless 3D rendering
#   - System libs: libosmesa6, libgl1, libglx-mesa0 (Debian Bookworm)
#   - uv for Python package management
#
# GPU is NOT required — all model inference runs via HuggingFace Spaces API.
#
# Build:
#   docker compose build
#
# Run:
#   docker compose run pipeline --shapes gear --models trellis --dry-run
#   docker compose run pipeline                           # full matrix
# ============================================================

FROM python:3.12-slim

# ── System dependencies ─────────────────────────────────────
# NOTE: python:3.12-slim is based on Debian Bookworm.
# libgl1-mesa-glx was removed in Bookworm; use libgl1 + libglx-mesa0 instead.
# libosmesa6 provides the OSMesa runtime (libosmesa6-dev adds headers, not needed at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OSMesa: software OpenGL for pyrender headless rendering (Bookworm packages)
        libosmesa6 \
        libgl1 \
        libglx-mesa0 \
        # Required by trimesh[easy] and other scientific Python libs
        libglib2.0-0 \
        libgomp1 \
        # X11 stubs (needed by some OpenGL libs even in headless mode)
        libxrender1 \
        libxext6 \
        # For downloading meshes and installing uv
        curl \
        # Needed by some trimesh loaders (git:// URIs)
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Environment ─────────────────────────────────────────────
# Tell PyOpenGL to use OSMesa (must be set before pyrender is imported)
ENV PYOPENGL_PLATFORM=osmesa

# Disable Python output buffering for real-time logs
ENV PYTHONUNBUFFERED=1

# ── Install uv ──────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Working directory ────────────────────────────────────────
WORKDIR /app/benchmark_pipeline

# ── Dependency layer (cached unless pyproject.toml changes) ──
# uv.lock is NOT committed (see .gitignore). uv lock resolves it fresh here;
# the resulting lock is baked into the image layer so the image is self-contained.
# To regenerate locally:  cd benchmark_pipeline && uv lock
COPY benchmark_pipeline/pyproject.toml ./

# Resolve + lock deps, then install (render = pyrender/PyOpenGL, dino = torch+transformers)
# torch on ARM64 Linux is CPU-only (no CUDA), so the wheel is ~300 MB not ~2 GB.
RUN uv lock && uv sync --extra render --extra dino --no-dev 2>&1

# ── Source code ──────────────────────────────────────────────
COPY benchmark_pipeline/ .
# shared/ utilities used by both pipelines
COPY shared/ /app/shared/

# Make shared/ importable as a top-level package
ENV PYTHONPATH=/app

# Create output directories (will be bind-mounted in compose)
RUN mkdir -p sessions cache

# ── Entry point ──────────────────────────────────────────────
ENTRYPOINT ["/app/benchmark_pipeline/.venv/bin/python", "run_pipeline.py"]

# Default command — runs full matrix (overridden in compose or on CLI)
CMD []
