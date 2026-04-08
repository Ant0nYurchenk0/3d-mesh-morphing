# RunPod — On-Premise Inference Servers

Three image-to-3D models deployed as standard RunPod Pod deployments, each
exposing a Gradio API on port 7860. The existing `shared/models/` clients
work unchanged — point `endpoint_url` at the pod instead of the HF Space.

## Models

| Folder | Model | HF Repo | Min VRAM |
|---|---|---|---|
| `trellis/` | TRELLIS-image-large | `JeffreyXiang/TRELLIS-image-large` | 16 GB |
| `trellis2/` | TRELLIS.2-4B | `microsoft/TRELLIS.2-4B` | 24 GB |
| `hunyuan3d2/` | Hunyuan3D-2 | `tencent/Hunyuan3D-2` | 16 GB |

---

## How It Works

Each subfolder contains three files:

- **`Dockerfile`** — builds the image. Installs all CUDA extensions, clones
  the model repo (TRELLIS/TRELLIS.2), and copies `server.py`.
- **`server.py`** — Gradio `Blocks` app. Downloads model weights from
  HuggingFace at container startup, loads them into GPU memory, then serves
  inference requests. The Gradio endpoints mirror the HF Space API so
  `shared/models/` clients need no changes.
- **`requirements.txt`** — pinned Python packages (Gradio, version pins for
  numpy/diffusers/transformers to match torch compatibility).

### Startup sequence (inside the container)

1. `server.py` logs in to HuggingFace Hub using `HF_TOKEN`.
2. Weights are downloaded into `$HF_HOME` (default `/hf-cache`). If a network
   volume is mounted there, weights persist across pod restarts.
3. The pipeline is loaded onto the GPU.
4. Gradio starts on `0.0.0.0:7860`.

### API endpoints

| Server | Endpoint | Description |
|---|---|---|
| trellis | `POST /start_session` | No-op session init (API compat) |
| trellis | `POST /generate_and_extract_glb` | One-shot inference → GLB |
| trellis2 | `POST /image_to_3d` | Runs inference, stores mesh in session state |
| trellis2 | `POST /extract_glb` | Exports GLB from session state |
| hunyuan3d2 | `POST /shape_generation` | One-shot inference → GLB |

---

## Building and Pushing

Build for `linux/amd64` (required when building on Apple Silicon):

```bash
# TRELLIS
docker build --platform linux/amd64 \
  -t <dockerhub-user>/trellis-server:latest \
  ./runpod/trellis
docker push <dockerhub-user>/trellis-server:latest

# TRELLIS.2
docker build --platform linux/amd64 \
  -t <dockerhub-user>/trellis2-server:latest \
  ./runpod/trellis2
docker push <dockerhub-user>/trellis2-server:latest

# Hunyuan3D-2
docker build --platform linux/amd64 \
  -t <dockerhub-user>/hunyuan3d2-server:latest \
  ./runpod/hunyuan3d2
docker push <dockerhub-user>/hunyuan3d2-server:latest
```

> **Build time:** ~30–45 min per image (flash-attn and several CUDA extensions
> are compiled from source). The resulting images are ~20–25 GB.

---

## Deploying on RunPod

### Pod settings

| Setting | Value |
|---|---|
| Container image | `<dockerhub-user>/trellis-server:latest` (or trellis2/hunyuan3d2) |
| Expose TCP port | `7860` |
| GPU | A10G (24 GB) for TRELLIS.2; A10G or A100 for others |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace token — needed to download gated models |
| `HF_HOME` | No | Weight cache dir (default `/hf-cache`). Mount a network volume here. |
| `MODEL_ID` | No | Override the default HF repo ID |
| `PORT` | No | Override Gradio port (default `7860`) |

### Network volume (recommended)

Mount a RunPod network volume at `/hf-cache` to persist model weights across
pod restarts. Without it, weights (~10–20 GB) are re-downloaded every time the
pod starts.

In the RunPod UI: **Edit Pod → Volume → Mount path: `/hf-cache`**

### Pod URL

Once the pod is running, the Gradio API is reachable at:

```
https://<pod-id>-7860.proxy.runpod.net
```

---

## Connecting the Benchmark Pipeline

Set `endpoint_url` in `benchmark_pipeline/config.yaml` for each model:

```yaml
models:
  trellis:
    endpoint_url: "https://<pod-id>-7860.proxy.runpod.net"

  trellis2:
    endpoint_url: "https://<pod-id>-7860.proxy.runpod.net"

  hunyuan3d:
    endpoint_url: "https://<pod-id>-7860.proxy.runpod.net"
```

Leave `endpoint_url` empty (`""`) to fall back to the public HF Space.

---

## Dependency Notes

### TRELLIS and TRELLIS.2

TRELLIS has no `setup.py` / `pyproject.toml` and cannot be installed via pip.
The Dockerfile clones the repo to `/opt/TRELLIS` and sets `PYTHONPATH` to make
it importable. Dependencies follow `setup.sh --basic` exactly:

```
pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy
ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph
transformers
```

Plus the optional extensions used by inference: `xformers`, `flash-attn`
(pinned to `2.5.9.post1` for torch 2.1.x compatibility), `spconv-cu120`,
`kaolin`, `utils3d` (pinned commit from `setup.sh`),
`diff-gaussian-rasterization` (from mip-splatting), and `nvdiffrast`.

`flash-attn ≥ 2.6` uses `std::optional` which clashes with torch 2.1's
`c10::optional` — hence the pin.

TRELLIS.2's `image_feature_extractor.py` accesses `self.model.layer` but the
current transformers `DINOv3ViTModel` nests layers under `.model.layer`. The
Dockerfile patches this with `sed` at build time.

### Hunyuan3D-2

The `hy3dgen` package (from the Hunyuan3D-2 repo) pulls in `numpy 2.x` and
`diffusers ≥ 0.30`, both incompatible with torch 2.1. The Dockerfile
force-downgrades these immediately after `pip install -e /opt/Hunyuan3D-2`
before `requirements.txt` runs.

`Hunyuan3DDiTFlowMatchingPipeline.to("cuda")` is an in-place operation that
returns `None` — do not reassign `_pipeline = _pipeline.to("cuda")`.
