# Video Preprocessing Pipeline Builder (with Optional Temporal Projections)

## Overview

This project provides a **video preprocessing pipeline builder** for large-scale computer vision experiments.

The script reads a directory of videos, converts all videos to **grayscale**, and automatically generates **multiple preprocessing pipelines** by combining different image processing operations (e.g. CLAHE, gamma correction, denoising).  
Optionally, after preprocessing each video frame-by-frame, the script can apply a **temporal projection** across frames (e.g. **max**, **mean**, **sum**, **std**, **argmax**, **quantile**) to produce a single 2D image per video.

All outputs are stored efficiently as **disk-backed NumPy arrays (`.dat` memmap files)** with accompanying **JSON metadata**, plus a global **manifest** for full reproducibility.

The design prioritises:
- Reproducibility
- Memory efficiency
- Large video dataset support
- Traceability  

## Features

- Reads videos from a folder (`.avi`, `.mp4`, configurable)
- Converts frames to grayscale
- Enforces consistent frame count and resolution
- Generates **all valid combinations and permutations** of preprocessing operations
- Applies each pipeline to all videos
- Optional **temporal projection** after preprocessing:
  - Sum projection
  - Max projection
  - Mean projection
  - Standard deviation projection
  - Argmax (peak frame index per pixel)
  - Quantile projection (configurable `q`)
- Stores results as **NumPy memmap (`.dat`) files**
- Produces:
  - Per-output metadata JSON
  - Global manifest JSON
  - `video_index.csv` (index map for video traceability)

## Pipeline Logic

Stages:

1. Equalisation
2. Gamma correction
3. Filtering
4. (Optional) Projection

- At most one operation per stage
- All permutations generated → many pipelines possible


## Input Data

- A directory containing video files
- All videos must:
  - Be readable by OpenCV
  - Have the same spatial resolution
  - Be compatible with the selected frame handling policy

##  Installation Requirements
- **Python**: `>=3.9`
- **Required libraries**:
  - `numpy`
  - `opencv-python`
## 🚀 Usage

### Basic Run (Default Settings)

```bash
python framesFusion.py
```

This will:

- Read videos from `./data`
- Accept `.avi` files
- Use strict frame matching
- Generate all preprocessing pipelines
- Save results to `./output`
- Output arrays will have shape `(N, T, H, W)` (no projection)

 

### Example: Custom Dataset & Parameters (No Projection)

```bash
python framesFusion.py   --data_path /test   --output_path ./outputs/run_01   --ext .avi .mp4   --frame_policy pad   --gamma_low 0.7   --gamma_high 1.3   --clahe_low_clip 1.0   --clahe_high_clip 3.0   --median_ksize 5   --bilateral_d 5   --bilateral_sigma_color 50   --bilateral_sigma_space 75   --nlmeans_h 12   --dtype float32
```

 

### Example: Enable Projections (Projected Output Only)

```bash
python framesFusion.py   --data_path /test   --output_path ./outputs/run_proj   --ext .avi .mp4   --frame_policy pad   --projection_mode projected_only   --enable_projection proj_max proj_mean proj_quantile   --quantile_q 0.75   --dtype uint8
```

 

### Example: Store Both Full Videos & Projected Outputs

```bash
python framesFusion.py   --data_path ./data   --output_path ./outputs/run_both   --projection_mode both   --enable_projection proj_max   --dtype uint8
```

 

### Example: Full Preprocessing + All Projections (Most Complete Run)

```bash
python framesFusion.py   --data_path /test   --output_path ./outputs/run_full_all   --ext .avi .mp4   --frame_policy pad   --max_videos 0    --gamma_low 0.7   --gamma_high 1.3   --clahe_low_clip 1.0   --clahe_low_grid 16   --clahe_high_clip 3.0   --clahe_high_grid 4   --median_ksize 5   --bilateral_d 5   --bilateral_sigma_color 50   --bilateral_sigma_space 75   --nlmeans_h 12   --nlmeans_template 3   --nlmeans_search 7   --projection_mode both   --enable_projection proj_sum proj_max proj_mean proj_std proj_argmax proj_quantile   --quantile_q 0.75   --dtype float32
```
 
 

## Command-Line Arguments

### **Input / Output**
| Argument       | Type       | Default       | Description                                |
| ------------|-----------|--------------|-------------------------------------------|
| `--data_path` | `str`     | `./data`     | Path to input video directory            |
| `--output_path`| `str`    | `./output`| Path to output directory                 |
| `--ext`       | `list[str]`| `.avi`      | Video extensions to include (e.g. `.avi .mp4`) |

 

### **Dataset Handling**
| Argument       | Type   | Default | Description                                      |
|---------------|--------|---------|--------------------------------------------------|
| `--frame_policy`| `str`| `strict`| Frame mismatch handling: `strict`, `crop`, `pad` |
| `--max_videos` | `int` | `0`     | If >0, process only first N videos (debug)      |

 

### **Preprocessing Ops**
Valid names for `--enable` / `--disable`:

- **Equalisation**: `clahe_low`, `clahe_high`
- **Gamma**: `gamma_low`, `gamma_high`
- **Filters**: `median_blur`, `bilateral`, `nlmeans` 

 

### **Projection Settings**
| Argument                | Type       | Default | Description                                  |
|-------------------------|-----------|---------|----------------------------------------------|
| `--projection_mode`     | `str`     | `none` | Output mode: `none`, `projected_only`, `both` |
| `--enable_projection`   | `list[str]`| `None` | Projection ops to include                   |
| `--disable_projection`  | `list[str]`| `None` | Projection ops to exclude                   |
| `--quantile_q`          | `float`   | `0.75` | Quantile for `proj_quantile`                |

Valid projection names: `proj_sum`, `proj_max`, `proj_mean`, `proj_std`, `proj_argmax`, `proj_quantile`

 

### **Hyperparameters**
- **Gamma**: `--gamma_low`, `--gamma_high`
- **CLAHE**: `--clahe_low_clip`, `--clahe_low_grid`, `--clahe_high_clip`, `--clahe_high_grid`
- **Filters**: `--median_ksize`, `--bilateral_d`, `--bilateral_sigma_color`, `--bilateral_sigma_space`, `--nlmeans_h`, `--nlmeans_template`, `--nlmeans_search`

 

## Outputs

- **Traceability**: `video_index.csv` maps array index → original filename
- **Per Pipeline**:
  - `projection_mode=none`: `(N, T, H, W)`
  - `projected_only`: `(N, H, W)`
  - `both`: `__full.dat` and `__proj-<ALIAS>.dat`
- **Global Manifest**: `pipelines_manifest_<timestamp>.json`

 
