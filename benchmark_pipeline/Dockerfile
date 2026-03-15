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
WORKDIR /app/pipeline

# ── Dependency layer (cached unless pyproject.toml changes) ──
# Copy manifests first so Docker can cache this layer
COPY pipeline/pyproject.toml ./
# uv.lock may not exist on first build; the wildcard handles that
COPY pipeline/uv.lock* ./

# Install core deps + render optional (pyrender + PyOpenGL)
# --no-dev skips dev/test deps if any are added later
RUN uv sync --extra render --no-dev 2>&1

# ── Source code ──────────────────────────────────────────────
COPY pipeline/ .

# Create output directories (will be bind-mounted in compose)
RUN mkdir -p sessions cache

# ── Entry point ──────────────────────────────────────────────
ENTRYPOINT ["uv", "run", "python", "run_pipeline.py"]

# Default command — runs full matrix (overridden in compose or on CLI)
CMD []
