# ============================================================
# 3D Mesh Morphing Pipeline — Docker image
# ============================================================
# Base: python:3.12-slim (Debian Bookworm)
#
# What this image includes:
#   - All pipeline Python dependencies (trimesh, openai, gradio-client, etc.)
#   - pyrender + PyOpenGL with OSMesa for headless 3D mesh rendering (node 3)
#   - System libs: libosmesa6, libgl1, libglx-mesa0 (Debian Bookworm)
#   - uv for Python package management
#
# GPU is NOT required — all model inference runs via HuggingFace Spaces API.
# OpenAI API calls are made via the openai Python SDK.
#
# Build:
#   docker compose build
#
# Run (image input):
#   docker compose run morphing --input-image /data/photo.png --prompt-file /data/prompt.txt
#
# Run (mesh input):
#   docker compose run morphing --input-mesh /data/object.glb --prompt-file /data/prompt.txt
# ============================================================

FROM python:3.12-slim

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OSMesa: software OpenGL for pyrender headless rendering
        libosmesa6 \
        libgl1 \
        libglx-mesa0 \
        # Required by trimesh[easy] and other scientific Python libs
        libglib2.0-0 \
        libgomp1 \
        # X11 stubs (needed by some OpenGL libs even in headless mode)
        libxrender1 \
        libxext6 \
        # Required by pymeshlab plugins (modern OpenGL dispatch + font rendering)
        libopengl0 \
        libharfbuzz0b \
        # For uv install and ManifoldPlus / Instant Meshes build
        curl \
        git \
        cmake \
        build-essential \
        # ManifoldPlus runtime and build deps
        libeigen3-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libtbb-dev \
        # Instant Meshes build deps (nanogui → GLFW → X11 + OpenGL headers)
        libgl-dev \
        libx11-dev \
        libxi-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        # xvfb: virtual framebuffer for running Instant Meshes headlessly
        xvfb \
        # xauth: required by xvfb-run to create Xauthority cookies
        xauth \
    && rm -rf /var/lib/apt/lists/*

# ── Build ManifoldPlus from source ──────────────────────────
# ManifoldPlus: watertight manifold repair (node 6 stage 2)
# https://github.com/hjwdzh/ManifoldPlus
RUN git clone --depth=1 --recurse-submodules https://github.com/hjwdzh/ManifoldPlus /tmp/ManifoldPlus \
    && ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen \
    && cmake -S /tmp/ManifoldPlus -B /tmp/ManifoldPlus/build \
             -DCMAKE_BUILD_TYPE=Release \
             -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    && cmake --build /tmp/ManifoldPlus/build --parallel "$(nproc)" \
    && install -m755 /tmp/ManifoldPlus/build/manifold /usr/local/bin/manifoldplus \
    && rm -rf /tmp/ManifoldPlus

# ── Build Instant Meshes from source ────────────────────────
# Instant Meshes: uniform remeshing for clean topology (node 6 stage 3)
# Runs via xvfb-run because nanogui/GLFW initialises a display even in batch mode.
# https://github.com/wjakob/instant-meshes
#
# ARM64 note: the bundled TBB (ext/tbb) is too old to have aarch64 machine
# primitives (__TBB_machine_fetchadd4 etc.).  We inject a thin compatibility
# header that maps the missing symbols to GCC __sync built-ins.  This is done
# via a cmake initial-cache file so the multi-word CXXFLAGS value is handled
# correctly without shell-quoting issues.
RUN git clone --depth=1 --recurse-submodules https://github.com/wjakob/instant-meshes /tmp/InstantMeshes \
    # ARM64 fix: ittnotify_config.h uses __TBB_machine_fetchadd4 before tbb_machine.h
    # is pulled in, producing "not declared" errors on aarch64.  The machine primitives
    # ARE defined in gcc_generic.h (which tbb_machine.h includes), but the inclusion
    # happens too late for ittnotify_config.h.  Prepend the missing include so the
    # machine header is seen first; include guards make the later re-inclusion a no-op.
    && sed -i '1s|^|#include <tbb/tbb_machine.h>\n|' \
           /tmp/InstantMeshes/ext/tbb/src/tbb/tools_api/ittnotify_config.h \
    && cmake -S /tmp/InstantMeshes -B /tmp/InstantMeshes/build \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_CXX_FLAGS="-Wno-changes-meaning" \
    && cmake --build /tmp/InstantMeshes/build --parallel "$(nproc)" \
    && install -m755 "/tmp/InstantMeshes/build/Instant Meshes" /usr/local/bin/InstantMeshes \
    && rm -rf /tmp/InstantMeshes

# ── Environment ─────────────────────────────────────────────
# Tell PyOpenGL to use OSMesa (must be set before pyrender is imported)
ENV PYOPENGL_PLATFORM=osmesa

# Disable Python output buffering for real-time logs
ENV PYTHONUNBUFFERED=1

# ── Install uv ──────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Working directory ────────────────────────────────────────
WORKDIR /app/morphing_pipeline

# ── Dependency layer (cached unless pyproject.toml changes) ──
# uv.lock is NOT committed (see .gitignore). uv lock resolves it fresh here;
# the resulting lock is baked into the image layer so the image is self-contained.
# To regenerate locally:  cd morphing_pipeline && uv lock
COPY morphing_pipeline/pyproject.toml ./

# Resolve + lock deps, then install (render = pyrender/PyOpenGL, repair = pymeshlab)
RUN uv lock && uv sync --extra render --extra repair --no-dev 2>&1

# ── Source code ──────────────────────────────────────────────
COPY morphing_pipeline/ .
# shared/ utilities used by both pipelines
COPY shared/ /app/shared/

# Make shared/ importable as a top-level package
ENV PYTHONPATH=/app

# Create session output directory (bind-mounted in compose)
RUN mkdir -p sessions

# ── Entry point ──────────────────────────────────────────────
ENTRYPOINT ["/app/morphing_pipeline/.venv/bin/python", "run_morphing.py"]

# Default: show help (override at runtime with actual arguments)
CMD ["--help"]
