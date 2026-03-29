# Morphing Pipeline

Generates a 3D mesh morphing animation from an input image or mesh. The pipeline uses GPT image editing to produce a target concept, reconstructs it as a 3D mesh, then morphs the base mesh toward the target via one of two methods: SDF interpolation (fast, CPU-friendly) or differentiable rendering (GPU, gradient-based).

## Pipeline overview

```
input mesh / image
      │
      ▼
[node 1] enhance_image   — GPT identity copy (normalises lighting/style; skippable)
      │
      ▼
[node 2] image_to_base_mesh  — image-to-3D reconstruction of base shape
      │
      ├── (mesh input path)
      ▼
[node 3] render_mesh     — renders the input mesh to a PNG for GPT
      │
      ▼
[node 4] morph_image     — GPT edits the image according to the prompt
      │
      ▼
[node 5] target_mesh     — image-to-3D reconstruction of morphed concept
      │
      ▼
[node 6] repair_mesh     — cleans up the target mesh (PyMeshLab → ManifoldPlus → Instant Meshes)
      │
      ▼
[node 7] morph_meshes    — produces transition frames (GLBs) from base → target
```

## Setup

```bash
cp .env.example .env   # add OPENAI_API_KEY and HF_TOKEN
docker compose build
```

## Running

Place input files in `data/` and a prompt in a `.txt` file.

```bash
# Mesh input — SDF morphing (default, fast)
docker compose run morphing \
  --input-mesh /data/object.off \
  --prompt-file /data/prompt.txt

# Mesh input — differential renderer (GPU, better quality)
docker compose run morphing \
  --input-mesh /data/object.off \
  --prompt-file /data/prompt.txt \
  --morph-method differential

# Image input
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
| `--morph-method` | `sdf` | Morphing algorithm: `sdf` or `differential` |
| `--skip-enhance` | off | Skip GPT identity copy (node 1); use raw input image |
| `--skip-base-mesh` | off | Skip base mesh generation (node 2) |
| `--remesh` | off | Run Instant Meshes as repair stage 3 (requires binary in image) |
| `--list` | — | Print available models and exit |

## Morphing methods (node 7)

### `--morph-method sdf` (default)

Both meshes are voxelised into signed distance fields on a 3D grid, then linearly interpolated at evenly spaced t values from 0 to 1. Each interpolated SDF is converted back to a mesh via marching cubes and exported as a GLB frame. Fast and CPU-friendly; works well when both meshes have similar topology.

Config keys under `morph:` in `config.yaml`:
- `grid_resolution` — SDF voxel grid side length (default 64; higher = more detail, slower)
- `n_frames` — number of interpolated GLB frames (default 30)

### `--morph-method differential`

Gradient-based geometry optimization using PyTorch3D. Runs in two phases:

**Phase 1 — Optimize deformation (source → target)**

The base mesh is first subdivided until it has at least ~2000 vertices (giving the optimizer enough degrees of freedom). A trainable per-vertex offset tensor is optimized with SGD+momentum to minimize a combined loss:

| Loss | Weight | Role |
|---|---|---|
| Chamfer distance | `w_chamfer` (5.0) | Primary 3D shape fidelity — point-cloud match to target surface |
| Multi-view silhouette | `w_silhouette` (0.1) | 2D boundary alignment across 20 camera views |
| Laplacian smoothing | `w_laplacian` (1.0) | Prevents spiky local deformations |
| Normal consistency | `w_normal` (0.1) | Keeps adjacent face normals smooth |
| Edge length | `w_edge` (0.01) | Weak guard against degenerate/collapsed triangles |

Silhouette loss uses a `SoftSilhouetteShader` (sigma=5e-3, ~1.8px blur at 256px) across 3 elevation rings of cameras, giving gradient signal to vertices not on the silhouette boundary. Regularization weights decay over training to allow coarse alignment first, then fine detail.

**Phase 2 — Interpolate to animation frames**

Once the final per-vertex offsets are known, the animation is produced by smoothly interpolating from zero to the full offsets using a Hermite smoothstep curve. This guarantees temporally uniform motion regardless of the optimizer's path.

Config keys under `diff_rend:` in `config.yaml`:
- `n_steps` — optimizer steps (default 4000; reduce to 100–500 for quick CPU tests)
- `lr` — SGD learning rate (default 0.001)
- `n_frames` — animation frames to export (default 40)
- `n_views` — silhouette camera viewpoints (default 20, split across 3 elevation rings)
- `image_size` — silhouette render resolution in px (default 256)
- `n_sample_pts` — Chamfer point-cloud samples per mesh (default 5000)
- `w_chamfer`, `w_silhouette`, `w_laplacian`, `w_normal`, `w_edge` — loss weights

Requires PyTorch3D (GPU build included in `Dockerfile.gpu`). CPU fallback is supported but slow.

### Canonical orientation (both methods)

Before morphing, both meshes are independently:
1. Centred and scaled to fit in [-1, 1]³
2. PCA-aligned to a Y-up canonical frame (largest variance axis → Y/height, 2nd largest → X/width, smallest → Z/depth)

This ensures both meshes stand in the same pose before optimization, so silhouette views from matching azimuths see semantically equivalent geometry. The rotation matrix is forced to det = +1 (proper rotation, not reflection) to prevent face winding reversal that would invert all normals.

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
  render.png            # mesh input only: render of input mesh
  enhanced_image.png    # image input only: GPT-reproduced input
  morphed_image.png     # GPT-edited image
  target_mesh.glb       # image-to-3D reconstruction
  repaired_mesh.glb     # after repair pipeline
  repair_report.json    # cost_before, cost_final, stages_run, face_count
  transition/
    frame_0000.glb      # morph frame 0 (= base mesh)
    frame_NNNN.glb      # morph frame N (= target shape)
    morph_info.json     # method, n_steps, loss weights, device used
```

## Docker (GPU — RunPod)

The GPU image (`Dockerfile.gpu`) uses `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` as base and builds PyTorch3D from source with CUDA support.

```bash
# Build and push
docker build -f morphing_pipeline/Dockerfile.gpu \
  -t yurchenkoanton/morphing-pipeline:gpu .
docker push yurchenkoanton/morphing-pipeline:gpu

# On RunPod (SSH)
docker pull yurchenkoanton/morphing-pipeline:gpu
docker run -d --gpus all --name morphing \
  -e OPENAI_API_KEY=<key> \
  yurchenkoanton/morphing-pipeline:gpu

docker exec morphing python run_morphing.py \
  --input-mesh data/gear.off \
  --prompt-file data/gear_smooth_prompt.txt \
  --morph-method differential
```

Download results (RunPod SSH doesn't support SCP — use base64 pipe):
```bash
# On RunPod
docker exec morphing bash -c "tar czf - sessions/ | base64" \
  | grep -E '^[A-Za-z0-9+/=]+' > frames.b64

# On Mac
base64 -d frames.b64 | tar xzf -
```
