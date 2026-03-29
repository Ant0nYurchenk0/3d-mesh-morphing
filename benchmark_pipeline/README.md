# Benchmark Pipeline

Evaluates image-to-3D reconstruction models by rendering reference meshes, reconstructing them, and computing quality metrics.

## Setup

```bash
cp .env.example .env   # add HF_TOKEN
docker compose build
```

## Running

```bash
# Full matrix (all shapes × all models)
docker compose run benchmark_pipeline

# Subset of shapes and models
docker compose run benchmark_pipeline \
  --shapes shape_37384,shape_37416 \
  --models trellis,trellis2

# Dry run: download and render only, no model API calls
docker compose run benchmark_pipeline --dry-run --shapes shape_37384

# Re-run metrics on an existing session (no API calls)
docker compose run benchmark_pipeline \
  --recompute-metrics /app/benchmark_pipeline/sessions/<run>

# List available shapes and models
docker compose run benchmark_pipeline --list
```

Session artifacts are written to `sessions/YYYY-MM-DD_HHMMSS_<uid>/`.

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--shapes LIST` | all | Comma-separated shape names to evaluate |
| `--models LIST` | all | Comma-separated model names to evaluate |
| `--dry-run` | off | Download + render only; skip model calls and metrics |
| `--recompute-metrics DIR` | — | Re-run metrics on an existing session directory |
| `--list` | — | Print available shapes and models then exit |

Available shapes and models are defined in `config.yaml`.

## Session artifacts

```
sessions/<run>/
  <shape>_<model>/
    original.off      # normalised ground-truth mesh
    render.png        # canonical 1024×1024 render
    reconstructed.glb # model output
    metrics.json      # 8 quality metrics
  summary.csv
  summary_per_model.csv
  summary.md
```

## Models

| Key | Space | Notes |
|---|---|---|
| `trellis` | `trellis-community/TRELLIS` | Community fork; exposes `/start_session` to fix session-directory bug |
| `trellis2` | `microsoft/TRELLIS.2` | Two-step API: `/image_to_3d` → `/extract_glb` |
| `hunyuan3d` | `tencent/Hunyuan3D-2` | Single-step `/shape_generation`; Space prone to PAUSED state |

Each model block in `config.yaml` accepts an optional `endpoint_url` field. When non-empty it overrides the HF Space and points the client at a self-hosted instance (e.g. RunPod). See [RUNPOD.md](RUNPOD.md) for deployment instructions.

## Metrics

| Metric | Direction | Notes |
|---|---|---|
| Chamfer Distance (CD) | ↓ lower is better | Mean squared nearest-neighbour distance |
| F-Score @ τ=0.02 | ↑ higher is better | Fraction of surface points within 2% of unit sphere |
| Hausdorff Distance | ↓ | Worst-case surface deviation |
| Volume IoU | ↑ | Voxel intersection-over-union; shown as `—` for non-watertight meshes |
| Normal Consistency | ↑ | Mean surface normal alignment |
| Morphing Readiness Score (MRS) | ↑ | Composite: structure + normals + topology |
| DINO Similarity | ↑ | DINOv2 render embedding cosine similarity; requires `--extra dino` (installed in Docker) |
| Cleanup Cost | ↓ | Integer count of mesh quality issues [0–7] |

All meshes are normalised to a unit bounding sphere before metric computation.

## RunPod deployment

See [RUNPOD.md](RUNPOD.md) for instructions on deploying each model as a self-hosted RunPod pod, eliminating dependency on HuggingFace Space availability.
