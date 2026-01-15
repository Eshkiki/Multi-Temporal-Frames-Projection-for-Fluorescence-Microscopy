# Multi-Temporal Frames Projection for Dynamic Processes Fusion in Fluorescence Microscopy
 A Video Preprocessing Pipeline Builder
## Overview
This project provides a **video preprocessing pipeline builder** for large-scale computer vision experiments.
The script reads a directory of videos, converts all videos to **grayscale**, and automatically generates **multiple preprocessing pipelines** by combining different image processing operations (e.g. contrast enhancement, gamma correction, denoising).  
Each pipeline is applied to every video, and the results are stored efficiently as **disk-backed NumPy arrays**.  
Detailed **JSON metadata** is produced to ensure full reproducibility.
The design prioritises:
- Reproducibility
- Memory efficiency
- Large video dataset support
## Features

- Reads videos from a folder (`.avi`, `.mp4`, configurable)
- Converts frames to grayscale
- Enforces consistent frame count and resolution
- Generates **all valid combinations and permutations** of preprocessing operations
- Applies each pipeline to all videos
- Stores results as **NumPy memmap (`.dat`) files**
- Produces per-pipeline metadata and a global manifest

## Input Data

- A directory containing video files
- All videos must:
  - Be readable by OpenCV
  - Have the same spatial resolution
  - Be compatible with the selected frame handling policy
## Installation Requirements
- Python 3.9+
- Required libraries:
  - `numpy`
  - `opencv-python`
## Usage
### Basic run (default settings)
```bash
python main_git.py
```
or
```bash
python main_git.py  --data_path D:\Dataset\MASI\tets --output_path ./outputs/run_01 --ext .avi .mp4 --frame_policy pad --gamma_low 0.7 --gamma_high 1.3 --clahe_low_clip 1.0 --clahe_high_clip 3.0 --median_ksize 5 --bilateral_d 5 --bilateral_sigma_color 50 --bilateral_sigma_space 75 --nlmeans_h 12 --dtype float32
```
