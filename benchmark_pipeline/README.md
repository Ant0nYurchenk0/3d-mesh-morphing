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
  --shapes gear,sphere \
  --models trellis,hunyuan3d

# Dry run: download and render only, no model API calls
docker compose run benchmark_pipeline --dry-run --shapes gear

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

Available shapes: `gear`, `star`, `torus`, `sphere`, `cloud`

Available models: `trellis`, `trellis2`, `hunyuan3d`, `triposg`, `spar3d`

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
