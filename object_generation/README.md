````markdown
# Gaussian-Splatting Pipeline – `train_everything.sh`

Automates per-object Gaussian training **and** world-space warping for one video sequence.

---

## 1.  Quick Start

```bash
# activate your environment first if needed
chmod +x train_everything.sh
CUDA_VISIBLE_DEVICES=0 ./train_everything.sh
````

`train_everything.sh` will:

1. train canonical Gaussians for every masked object
2. fill any missing time steps
3. warp each object into COLMAP world space
4. export combined `.npy` tensors for downstream code

---

## 2.  Required Folders

| Variable (edit in script) | What it points to                            |
| ------------------------- | -------------------------------------------- |
| `INPUT_DIR`               | RGB frames (`*.png` / `*.jpg`)               |
| `MASK_DIR`                | Per-object masks (one sub-folder per object) |
| `DEPTH_DIR`               | Depth maps aligned to `INPUT_DIR`            |
| `CAM_PARAM_PATH`          | `cam_params.pkl` (extrinsics)                |
| `INTRINSIC_PATH`          | `cameras.txt` (intrinsics)                   |

---

## 3.  Outputs

```
Gaussian_output/scene_6/
└── object_##/
    ├── canonical/   # object-centric checkpoints (*.pkl)
    ├── world/       # world-space checkpoints + tensors
    └── vis/         # optional debug renders (if VISDIR set)
```

*Key artefacts*
`world/combined_xyzrgb.npy` – merged tensor for rendering / analysis.

---

## 4.  Tune-Me Params

| Name       | Default              | Meaning               |
| ---------- | -------------------- | --------------------- |
| `STEPSIZE` | 30                   | Optimiser step size   |
| `ITERS`    | 1000                 | Iterations per object |
| `CONFIG`   | `configs/image.yaml` | Base training config  |

Adjust for quality vs. speed trade-offs.

---

## 5.  Troubleshooting

* **Path errors** – verify every directory constant in the header.
* **Alignment drift** – confirm depth & camera files come from the same COLMAP run.
* **GPU OOM** – lower `ITERS` or reduce frame count.

Happy splatting!

```
```
