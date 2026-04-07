# Image-to-3D Evaluation — Session Summary

## Per-shape results

| Shape | Model | CD ↓ | F-Score ↑ | Hausdorff ↓ | Vol IoU ↑ | Normal Cons ↑ | MRS ↑ | DINO ↑ | Cost ↓ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| gear | hunyuan3d | 0.24856 | 0.03735 | 0.78973 | 0.16924 | 0.25593 | 0.77678 | 0.77038 | 3 | poor |
| gear | trellis | 0.24375 | 0.04834 | 0.76638 | — | 0.28768 | 0.45297 | 0.77414 | 4 | poor |
| gear | trellis2 | 0.22860 | 0.04163 | 0.76137 | — | 0.26437 | 0.44598 | 0.78284 | 5 | poor |
| sphere | hunyuan3d | 0.00091 | 0.57305 | 0.06218 | — | 0.99817 | 0.49945 | 0.96384 | 6 | poor |
| sphere | trellis | 0.00280 | 0.13505 | 0.07350 | — | 0.99775 | 0.66599 | 0.96364 | 3 | poor |
| sphere | trellis2 | 0.00193 | 0.20001 | 0.07316 | — | 0.99792 | 0.66604 | 0.92871 | 4 | poor |
| star | hunyuan3d | 0.20158 | 0.04835 | 0.86962 | — | 0.30150 | 0.29045 | 0.73379 | 7 | poor |
| star | trellis | 0.19726 | 0.04856 | 0.88448 | — | 0.29555 | 0.45533 | 0.73682 | 4 | poor |
| star | trellis2 | 0.18601 | 0.04940 | 0.83756 | — | 0.29888 | 0.45633 | 0.75521 | 6 | poor |
| torus | hunyuan3d | 0.35215 | 0.04180 | 0.97617 | 0.12229 | 0.45370 | 0.66944 | 0.67668 | 4 | poor |
| torus | trellis | 0.33555 | 0.03039 | 0.95814 | — | 0.46096 | 0.50496 | 0.60187 | 4 | poor |
| torus | trellis2 | 0.36911 | 0.03786 | 0.97326 | — | 0.44891 | 0.50134 | 0.65383 | 4 | poor |

**Thresholds (unit-sphere normalised):**
- CD: excellent < 0.005, good < 0.010, poor > 0.030
- F-Score: excellent > 0.90, good > 0.85, poor < 0.70
- MRS: excellent > 0.80, good > 0.65, poor < 0.40

*Volume IoU shown as `—` when either mesh is not watertight.*
*DINO shown as `—` when torch/transformers not installed (`uv sync --extra dino`).*

## Per-model mean (successful shapes only)

| Model | N | CD ↓ | F-Score ↑ | Hausdorff ↓ | Vol IoU ↑ | Normal Cons ↑ | MRS ↑ | DINO ↑ | Cost ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hunyuan3d | 4 | 0.20080 | 0.17514 | 0.67442 | 0.14577 | 0.50233 | 0.55903 | 0.78617 | 5.0 |
| trellis | 4 | 0.19484 | 0.06559 | 0.67063 | — | 0.51048 | 0.51981 | 0.76912 | 3.8 |
| trellis2 | 4 | 0.19641 | 0.08223 | 0.66134 | — | 0.50252 | 0.51742 | 0.78015 | 4.8 |

*Means computed only over shapes where the model succeeded (no error).*