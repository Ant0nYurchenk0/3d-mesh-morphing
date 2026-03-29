# RunPod Deployment Guide

Self-hosting models on RunPod eliminates dependency on HuggingFace Space availability (PAUSED state, GPU task aborts, shared queues). Each model runs as a pod with a Gradio endpoint; the benchmark pipeline connects to it via `endpoint_url` in `config.yaml`.

## How it works

`gradio_client.Client` accepts a URL in addition to a HF Space ID. Set `endpoint_url` in a model's config block to route all calls to your RunPod pod instead of HF Spaces:

```yaml
# benchmark_pipeline/config.yaml
hunyuan3d:
  space_id: "tencent/Hunyuan3D-2"       # used only when endpoint_url is empty
  endpoint_url: "https://<pod-id>-8080.proxy.runpod.net"
```

Leave `endpoint_url: ""` to keep using the HF Space.

---

## Model 1: Hunyuan3D-2 — start here

**Why first**: simplest to containerize (PyPI package), currently 0% success on HF (Space PAUSED).

| | |
|---|---|
| HF model repo | `tencent/Hunyuan3D-2` |
| PyPI package | `hy3dgen` — no GitHub clone needed |
| Gradio endpoint | `/shape_generation` |
| Min VRAM | 6 GB (shape only, no texture) |
| Recommended GPU | RTX 4090 (24 GB) or A40 |

### Docker image

The Dockerfile lives at `runpod/hunyuan3d2/Dockerfile` in this repo.

Build and push:
```bash
docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -t your-dockerhub-user/hunyuan3d2:latest \
  -f runpod/hunyuan3d2/Dockerfile .

docker push your-dockerhub-user/hunyuan3d2:latest
```

### Deploy on RunPod

1. Go to **RunPod → Pods → Deploy**.
2. Select **RTX 4090** or **A40** template.
3. Set container image to `your-dockerhub-user/hunyuan3d2:latest`.
4. Set environment variable: `HF_TOKEN=<your token>` (needed if weights weren't baked at build time).
5. Expose **HTTP port 8080**.
6. Start the pod. Wait for the Gradio app to load (~2–3 min on first run as weights download).
7. Copy the pod proxy URL: `https://<pod-id>-8080.proxy.runpod.net`.

### Update config

```yaml
hunyuan3d:
  space_id: "tencent/Hunyuan3D-2"
  endpoint_url: "https://<pod-id>-8080.proxy.runpod.net"
```

### Verify endpoint

```python
from gradio_client import Client
c = Client("https://<pod-id>-8080.proxy.runpod.net")
c.view_api()   # should show /shape_generation
```

---

## Model 2: TRELLIS

| | |
|---|---|
| HF model repo | `microsoft/TRELLIS-image-large` |
| Inference code | Requires cloning `github.com/microsoft/TRELLIS` + submodules |
| Gradio endpoints | `/start_session` + `/generate_and_extract_glb` |
| Min VRAM | 16 GB |
| Recommended GPU | A100 40 GB |

The community fork (`trellis-community/TRELLIS`) is used in the benchmark pipeline to avoid a session-directory bug in the original Microsoft Space. When self-hosting, that bug doesn't exist, so you can use the original `app.py` from the Microsoft repo.

### Docker image

The Dockerfile lives at `runpod/trellis/Dockerfile` in this repo.

Build and push:
```bash
docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  -t your-dockerhub-user/trellis:latest \
  -f runpod/trellis/Dockerfile .

docker push your-dockerhub-user/trellis:latest
```

> **Note**: The TRELLIS Dockerfile compiles several CUDA extensions during build (`spconv`, `nvdiffrast`, `kaolin`, etc). Expect a 20–40 min build. Use `--no-cache` only when necessary.

### Deploy on RunPod

Same steps as Hunyuan3D-2, but select **A100 40 GB** and expose port 8080.

### Update config

```yaml
trellis:
  space_id: "trellis-community/TRELLIS"
  endpoint_url: "https://<pod-id>-8080.proxy.runpod.net"
```

The existing `TRELLISClient` calls `/start_session` first. If you use the Microsoft `app.py` directly, confirm whether `/start_session` is present; if not, remove that call from `shared/models/trellis.py`.

---

## Model 3: TRELLIS.2

TRELLIS.2 uses a two-step stateful API (`/image_to_3d` stores a `gr.State` server-side, then `/extract_glb` reads it). Self-hosting requires the same session-bound Gradio app as the Microsoft Space.

**Status**: Defer until Hunyuan3D-2 and TRELLIS are validated on RunPod.

To proceed:
1. Inspect the Space source at `https://huggingface.co/spaces/microsoft/TRELLIS.2/tree/main`.
2. Identify the `app.py` and the model checkpoint it loads.
3. Build a Dockerfile similar to TRELLIS but pointing at the TRELLIS.2 checkpoint.

---

## Cost reference

| GPU | RunPod price | 18-call benchmark (~30 min) |
|---|---|---|
| RTX 4090 (24 GB) | ~$0.74/hr | ~$0.37 |
| A40 (48 GB) | ~$0.76/hr | ~$0.38 |
| A100 40 GB | ~$2.09/hr | ~$1.05 |
| A100 80 GB | ~$2.49/hr | ~$1.25 |

Use **on-demand** for occasional benchmark runs. Reserve instances only if running daily.

---

## Stopping pods

Stop pods between runs to avoid idle charges. The model weights are preserved in the container; restart the pod to resume without re-downloading.

If you baked weights into the image at build time (via `RUN huggingface-cli download ...`), you can stop and restart without any HF token or download time.
