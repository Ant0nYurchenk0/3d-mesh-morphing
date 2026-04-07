# Image-to-3D Evaluation — Session Summary

## Per-shape results

| Shape | Model | CD ↓ | F-Score ↑ | Hausdorff ↓ | Vol IoU ↑ | Normal Cons ↑ | MRS ↑ | DINO ↑ | Cost ↓ | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| shape_37384 | hunyuan3d | 0.08052 | 0.04202 | 0.54666 | 0.37937 | 0.54436 | 0.86331 | 0.61422 | 2 | poor |
| shape_37384 | trellis | 0.07256 | 0.05306 | 0.55891 | — | 0.61996 | 0.55266 | 0.61691 | 3 | poor |
| shape_37384 | trellis2 | 0.05591 | 0.04779 | 0.41130 | — | 0.71429 | 0.58095 | 0.66103 | 5 | poor |
| shape_37416 | hunyuan3d | 0.08698 | 0.07679 | 0.79843 | — | 0.48497 | 0.67882 | 0.62235 | 5 | poor |
| shape_37416 | trellis | 0.04276 | 0.15265 | 0.38191 | — | 0.55267 | 0.53247 | 0.69250 | 4 | poor |
| shape_37416 | trellis2 | 0.06469 | 0.14661 | 0.67745 | — | 0.51521 | 0.52123 | 0.55245 | 5 | poor |
| shape_38554 | hunyuan3d | 0.16968 | 0.05622 | 0.73505 | — | 0.53166 | 0.59283 | 0.68278 | 5 | poor |
| shape_38554 | trellis | 0.15198 | 0.04770 | 0.68698 | — | 0.49125 | 0.51404 | 0.67102 | 4 | poor |
| shape_38554 | trellis2 | 0.17570 | 0.11489 | 0.75132 | — | 0.52342 | 0.52369 | 0.69334 | 6 | poor |
| shape_38741 | hunyuan3d | 0.31844 | 0.04490 | 0.96009 | 0.04473 | 0.28640 | 0.58592 | 0.66300 | 4 | poor |
| shape_38741 | trellis | 0.32774 | 0.05998 | 0.95850 | — | 0.33578 | 0.46740 | 0.65770 | 4 | poor |
| shape_38741 | trellis2 | 0.33337 | 0.06205 | 0.96767 | — | 0.32687 | 0.46473 | 0.65159 | 5 | poor |
| shape_44498 | hunyuan3d | 0.22775 | 0.04580 | 0.83135 | 0.11708 | 0.37271 | 0.81181 | 0.59256 | 3 | poor |
| shape_44498 | trellis | 0.29603 | 0.04511 | 0.91108 | — | 0.25554 | 0.44333 | 0.54453 | 4 | poor |
| shape_44498 | trellis2 | 0.25240 | 0.06249 | 0.90062 | — | 0.28626 | 0.45254 | 0.50192 | 5 | poor |
| shape_48013 | hunyuan3d | 0.11250 | 0.11602 | 0.65334 | — | 0.57748 | 0.37324 | 0.66162 | 6 | poor |
| shape_48013 | trellis | 0.10247 | 0.09030 | 0.60260 | — | 0.51354 | 0.52073 | 0.64858 | 4 | poor |
| shape_48013 | trellis2 | 0.09409 | 0.15684 | 0.63004 | — | 0.54554 | 0.53033 | 0.63626 | 5 | poor |

**Thresholds (unit-sphere normalised):**
- CD: excellent < 0.005, good < 0.010, poor > 0.030
- F-Score: excellent > 0.90, good > 0.85, poor < 0.70
- MRS: excellent > 0.80, good > 0.65, poor < 0.40

*Volume IoU shown as `—` when either mesh is not watertight.*
*DINO shown as `—` when torch/transformers not installed (`uv sync --extra dino`).*

## Per-model mean (successful shapes only)

| Model | N | CD ↓ | F-Score ↑ | Hausdorff ↓ | Vol IoU ↑ | Normal Cons ↑ | MRS ↑ | DINO ↑ | Cost ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hunyuan3d | 6 | 0.16598 | 0.06362 | 0.75415 | 0.18039 | 0.46627 | 0.65099 | 0.63942 | 4.2 |
| trellis | 6 | 0.16559 | 0.07480 | 0.68333 | — | 0.46146 | 0.50510 | 0.63854 | 3.8 |
| trellis2 | 6 | 0.16269 | 0.09844 | 0.72307 | — | 0.48527 | 0.51225 | 0.61610 | 5.2 |

*Means computed only over shapes where the model succeeded (no error).*