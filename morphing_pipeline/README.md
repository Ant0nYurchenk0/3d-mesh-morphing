# Morphing Pipeline

Generates a morphed 3D mesh from an input image or mesh using GPT image editing and an image-to-3D model.

## Setup

```bash
cp .env.example .env   # add OPENAI_API_KEY and HF_TOKEN
docker compose build
```

## Running

Place input files in `data/` and a prompt in a `.txt` file.

```bash
# Mesh input (render → morph image → reconstruct → repair)
docker compose run morphing \
  --input-mesh /data/object.off \
  --prompt-file /data/prompt.txt

# Image input (enhance → base mesh → morph image → reconstruct → repair)
docker compose run morphing \
  --input-image /data/photo.png \
  --prompt-file /data/prompt.txt
```

Session artifacts are written to `sessions/YYYY-MM-DD_HHMMSS_<uid>/`.

## Parameters

| Flag | Default | Description |
|---|---|---|
| `--input-mesh PATH` | — | Input mesh (OFF/GLB/OBJ/STL) |
| `--input-image PATH` | — | Input image (PNG/JPEG) |
| `--prompt-file PATH` | — | `.txt` file with morphing prompt |
| `--model NAME` | `trellis` | Image-to-3D model: `trellis`, `trellis2`, `hunyuan3d` |
| `--skip-enhance` | off | Skip GPT identity copy (node 1); use raw input image |
| `--skip-base-mesh` | off | Skip base mesh generation (node 2) |
| `--remesh` | off | Run Instant Meshes as repair stage 3 (requires binary in image) |
| `--list` | — | Print available models and exit |

## Repair stages (node 6)

| Stage | Tool | Condition |
|---|---|---|
| 1 | PyMeshLab | always |
| 2 | ManifoldPlus | if `cleanup_cost > 0` after stage 1 |
| 3 | Instant Meshes | only with `--remesh` |

Each stage produces an intermediate file (`repair_stage1.glb`, etc.) in the session directory. A `repair_report.json` with cost metrics is also saved.

## Session artifacts

```
sessions/<run>/
  render.png          # mesh input only: render of input mesh
  enhanced_image.png  # image input only: GPT-reproduced input
  morphed_image.png   # GPT-edited image
  target_mesh.glb     # image-to-3D reconstruction
  repaired_mesh.glb   # after repair pipeline
  repair_report.json  # cost_before, cost_final, stages_run, face_count
```
