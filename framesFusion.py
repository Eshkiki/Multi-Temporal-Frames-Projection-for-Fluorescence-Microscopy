"""
Video preprocessing pipeline builder + optional temporal projections.

What this script does
- Reads a directory of videos
- Converts to grayscale
- Validates consistent (H, W) across videos
- Enforces consistent frame count T using a frame policy (strict/crop/pad)
- Generates preprocessing pipelines from stages (equalisation, gamma, filter)
- Optionally applies a temporal projection over frames after preprocessing
- Saves results as disk-backed NumPy memmaps (.dat) + JSON metadata
- Writes a global manifest JSON and an explicit index map (video_index.csv)

Outputs (depending on projection_mode)
- projection_mode=none:
  - one .dat per pipeline with shape (N, T, H, W)
- projection_mode=projected_only:
  - one .dat per pipeline with shape (N, H, W)
- projection_mode=both:
  - two .dat per pipeline:
      * __full.dat  with shape (N, T, H, W)
      * __proj-<ALIAS>.dat with shape (N, H, W)

Also produces:
- one JSON metadata per output .dat
- pipelines_manifest_<timestamp>.json
- video_index.csv mapping (index -> filename -> full path)


"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import product, permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2


# -----------------------------
# Helpers
# -----------------------------

def show_error_message_box(msg: str, exit_code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(exit_code)


def check_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        show_error_message_box(f"Cannot create output directory '{path}': {e}")


def list_videos(data_path: Path, exts: Sequence[str]) -> List[Path]:
    if not data_path.exists():
        show_error_message_box(f"Data path does not exist: {data_path}")
    if not data_path.is_dir():
        show_error_message_box(f"Data path is not a directory: {data_path}")

    exts_lower = {e.lower() for e in exts}
    files = [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() in exts_lower]
    files.sort()

    if not files:
        show_error_message_box(f"No video files found in {data_path} with extensions: {sorted(exts_lower)}")
    return files


def safe_video_read_gray(video_path: Path) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open video")

    frames: List[np.ndarray] = []
    h = w = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV returns BGR for color videos; handle grayscale or color robustly
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if h is None:
            h, w = gray.shape[:2]

        frames.append(gray)

    cap.release()

    if h is None or w is None or len(frames) == 0:
        raise RuntimeError("Video contains no readable frames")

    return frames, (h, w)


def adjust_frames(frames: List[np.ndarray], T: int, policy: str) -> Optional[np.ndarray]:
    """Return array (T,H,W) or None if strict mismatch."""
    n = len(frames)
    if n == T:
        return np.stack(frames, axis=0)

    if policy == "strict":
        return None

    if policy == "crop":
        if n < T:
            return None  # cannot crop up
        return np.stack(frames[:T], axis=0)

    if policy == "pad":
        if n > T:
            return np.stack(frames[:T], axis=0)
        pad_count = T - n
        if pad_count <= 0:
            return np.stack(frames[:T], axis=0)
        last = frames[-1]
        padded = frames + [last] * pad_count
        return np.stack(padded, axis=0)

    raise ValueError(f"Unknown frame policy: {policy}")


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
    # If likely [0,1], scale; else clamp.
    if x.size > 0 and x.max() <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


# -----------------------------
# Operation definitions (frame ops)
# -----------------------------

@dataclass(frozen=True)
class Operation:
    name: str
    alias: str
    stage: str
    params: Dict[str, Any]

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GammaCorrection(Operation):
    def apply(self, img: np.ndarray) -> np.ndarray:
        x = to_uint8(img)
        gamma = float(self.params["gamma"])
        if gamma <= 0:
            gamma = 1.0
        inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(x, table)


class CLAHE(Operation):
    def apply(self, img: np.ndarray) -> np.ndarray:
        x = to_uint8(img)
        clip = float(self.params["clipLimit"])
        grid = tuple(self.params["tileGridSize"])
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        return clahe.apply(x)


class MedianBlur(Operation):
    def apply(self, img: np.ndarray) -> np.ndarray:
        x = to_uint8(img)
        k = int(self.params["ksize"])
        if k <= 0:
            k = 1
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(x, k)


class BilateralFilter(Operation):
    def apply(self, img: np.ndarray) -> np.ndarray:
        x = to_uint8(img)
        d = int(self.params["d"])
        sc = float(self.params["sigmaColor"])
        ss = float(self.params["sigmaSpace"])
        return cv2.bilateralFilter(x, d, sc, ss)


class NlMeans(Operation):
    def apply(self, img: np.ndarray) -> np.ndarray:
        x = to_uint8(img)
        h = float(self.params["h"])
        tw = int(self.params["templateWindowSize"])
        sw = int(self.params["searchWindowSize"])
        return cv2.fastNlMeansDenoising(x, None, h, tw, sw)


 
# -----------------------------
# Projection operation definitions (video-level ops)
# -----------------------------

@dataclass(frozen=True)
class ProjectionOperation(Operation):
    """Projection over time axis; input is (T,H,W), output is (H,W)."""
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # Not used for projections, but kept for interface completeness
    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ProjectionOperation does not support apply(img); use apply_video(video_arr).")


class SumProjection(ProjectionOperation):
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        return np.sum(video_arr, axis=0)


class MaxProjection(ProjectionOperation):
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        return np.max(video_arr, axis=0)


class MeanProjection(ProjectionOperation):
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        return np.mean(video_arr, axis=0)


class StdProjection(ProjectionOperation):
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        return np.std(video_arr, axis=0)


class ArgmaxProjection(ProjectionOperation):
    """
    Returns per-pixel index of the peak frame (0..T-1).
    If stored as uint8 and T>256, we scale indices to 0..255 for compatibility.
    """
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        return np.argmax(video_arr, axis=0)


class QuantileProjection(ProjectionOperation):
    def apply_video(self, video_arr: np.ndarray) -> np.ndarray:
        q = float(self.params["q"])
        return np.quantile(video_arr, q, axis=0)


# -----------------------------
# Registry and pipeline generation
# -----------------------------

def build_registry(args: argparse.Namespace) -> Dict[str, List[Operation]]:
    # Stages: equalisation, gamma, filter, projection
    reg: Dict[str, List[Operation]] = {"equalisation": [], "gamma": [], "filter": [], "projection": []}

    # Equalisation
    reg["equalisation"].append(CLAHE(
        name="clahe_low",
        alias="CL",
        stage="equalisation",
        params={"clipLimit": args.clahe_low_clip, "tileGridSize": [args.clahe_low_grid, args.clahe_low_grid]},
    ))
    reg["equalisation"].append(CLAHE(
        name="clahe_high",
        alias="CH",
        stage="equalisation",
        params={"clipLimit": args.clahe_high_clip, "tileGridSize": [args.clahe_high_grid, args.clahe_high_grid]},
    ))

    # Gamma
    reg["gamma"].append(GammaCorrection(
        name="gamma_low",
        alias="GL",
        stage="gamma",
        params={"gamma": args.gamma_low},
    ))
    reg["gamma"].append(GammaCorrection(
        name="gamma_high",
        alias="GH",
        stage="gamma",
        params={"gamma": args.gamma_high},
    ))

    # Filters
    reg["filter"].append(MedianBlur(
        name="median_blur",
        alias="MB",
        stage="filter",
        params={"ksize": args.median_ksize},
    ))
    reg["filter"].append(BilateralFilter(
        name="bilateral",
        alias="BF",
        stage="filter",
        params={"d": args.bilateral_d, "sigmaColor": args.bilateral_sigma_color, "sigmaSpace": args.bilateral_sigma_space},
    ))
    reg["filter"].append(NlMeans(
        name="nlmeans",
        alias="NF",
        stage="filter",
        params={"h": args.nlmeans_h, "templateWindowSize": args.nlmeans_template, "searchWindowSize": args.nlmeans_search},
    ))

  

    # Projections (video-level)
    reg["projection"].append(SumProjection(name="proj_sum", alias="SUM", stage="projection", params={}))
    reg["projection"].append(MaxProjection(name="proj_max", alias="MAX", stage="projection", params={}))
    reg["projection"].append(MeanProjection(name="proj_mean", alias="MEAN", stage="projection", params={}))
    reg["projection"].append(StdProjection(name="proj_std", alias="STD", stage="projection", params={}))
    reg["projection"].append(ArgmaxProjection(name="proj_argmax", alias="ARGMAX", stage="projection", params={}))
    reg["projection"].append(QuantileProjection(
        name="proj_quantile", alias="Q", stage="projection", params={"q": args.quantile_q}
    ))

    return reg


def filter_registry(
    reg: Dict[str, List[Operation]],
    enabled: Optional[Sequence[str]],
    disabled: Optional[Sequence[str]],
) -> Dict[str, List[Operation]]:
    enabled_set = set(enabled) if enabled else None
    disabled_set = set(disabled) if disabled else set()

    out: Dict[str, List[Operation]] = {}
    for stage, ops in reg.items():
        stage_ops = []
        for op in ops:
            if op.name in disabled_set:
                continue
            if enabled_set is not None and op.name not in enabled_set:
                continue
            stage_ops.append(op)
        out[stage] = stage_ops
    return out


@dataclass
class Pipeline:
    pipeline_id: str
    alias: str
    ops: List[Operation]  # ordered list


def build_pipelines(
    reg: Dict[str, List[Operation]],
    stages: Sequence[str],
    required_stages: Optional[Sequence[str]] = None,
) -> List[Pipeline]:
    """
    Generate all combinations with:
    - at most one op from each stage
    - missing stages allowed via None, except required_stages where None is not allowed
    - order matters (all permutations of chosen ops)
    """
    required = set(required_stages or [])

    stage_choices: List[List[Optional[Operation]]] = []
    for s in stages:
        ops = reg.get(s, [])
        if s in required:
            # must choose exactly one op from this stage
            stage_choices.append(list(ops))
        else:
            stage_choices.append([None] + list(ops))

    raw = list(product(*stage_choices))

    pipelines: List[Pipeline] = []
    seen_alias = set()

    for choice in raw:
        chosen = [op for op in choice if op is not None]
        if not chosen:
            continue

        for perm in permutations(chosen, r=len(chosen)):
            ops = list(perm)
            alias = "_" + "_".join(op.alias for op in ops)

            if alias in seen_alias:
                continue
            seen_alias.add(alias)

            pid = f"p{len(pipelines):04d}"
            pipelines.append(Pipeline(pipeline_id=pid, alias=alias, ops=ops))

    return pipelines


# -----------------------------
# Processing
# -----------------------------

def apply_frame_ops(video_arr: np.ndarray, ops: List[Operation]) -> np.ndarray:
    """
    video_arr: (T,H,W)
    returns: (T,H,W) uint8
    """
    T = video_arr.shape[0]
    out = np.empty_like(video_arr, dtype=np.uint8)

    for t in range(T):
        img = video_arr[t]
        for op in ops:
            img = op.apply(img)
        out[t] = to_uint8(img)

    return out


def apply_projection(processed_video: np.ndarray, proj_op: ProjectionOperation, out_dtype: np.dtype) -> np.ndarray:
    """
    processed_video: (T,H,W) uint8
    returns: (H,W) with dtype consistent with out_dtype (uint8 or float32)
    """
    proj = proj_op.apply_video(processed_video)

    # Special case: argmax can exceed uint8 range if T is large; scale if needed
    if proj_op.name == "proj_argmax" and out_dtype == np.uint8:
        T = processed_video.shape[0]
        if T <= 1:
            proj_scaled = np.zeros_like(proj, dtype=np.uint8)
        else:
            # scale 0..T-1 into 0..255
            proj_scaled = np.round((proj.astype(np.float32) / (T - 1)) * 255.0)
            proj_scaled = np.clip(proj_scaled, 0, 255).astype(np.uint8)
        return proj_scaled

    # For other projections, convert to uint8 safely
    if out_dtype == np.uint8:
        return to_uint8(proj)

    # float32 output: normalise into [0,1] if looks like 0..255
    x = proj.astype(np.float32)
    # If it looks like uint8-like intensity, scale down
    if x.size > 0 and x.max() > 1.5:
        x = x / 255.0
    x = np.clip(x, 0.0, 1.0)
    return x


def write_video_index_map(out_path: Path, valid_videos: List[Path]) -> str:
    video_index_path = out_path / "video_index.csv"
    with open(video_index_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "filename", "full_path"])
        for i, vp in enumerate(valid_videos):
            w.writerow([i, vp.name, str(vp.resolve())])
    return video_index_path.name


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build preprocessing pipelines for a video dataset; optionally apply temporal projections; save one array per pipeline."
    )

    # I/O
    parser.add_argument("--data_path", type=str, default="./data", help="Input folder containing videos.")
    parser.add_argument("--output_path", type=str, default="./output", help="Output folder.")
    parser.add_argument("--ext", type=str, nargs="+", default=[".avi"], help="Video extensions to include (e.g., .avi .mp4).")

    # Dataset handling
    parser.add_argument("--frame_policy", type=str, choices=["strict", "crop", "pad"], default="strict",
                        help="How to handle videos with different frame counts vs inferred T.")
    parser.add_argument("--max_videos", type=int, default=0, help="If >0, process only first N videos (debug).")

    # Frame-level pipeline selection
    parser.add_argument("--enable", type=str, nargs="*", default=None,
                        help="If provided, only these operation names will be used (e.g., gamma_low median_blur).")
    parser.add_argument("--disable", type=str, nargs="*", default=None,
                        help="Operation names to exclude.")


    # Projection configuration
    parser.add_argument("--projection_mode", type=str, choices=["none", "projected_only", "both"], default="none",
                        help="none: store (N,T,H,W). projected_only: store (N,H,W). both: store both outputs.")
    parser.add_argument("--enable_projection", type=str, nargs="*", default=None,
                        help="If provided, only these projection ops are used (e.g., proj_max proj_mean proj_quantile).")
    parser.add_argument("--disable_projection", type=str, nargs="*", default=None,
                        help="Projection ops to exclude.")
    parser.add_argument("--quantile_q", type=float, default=0.75, help="Quantile value used by proj_quantile (default 0.75).")

    # Hyperparameters
    parser.add_argument("--gamma_low", type=float, default=0.75)
    parser.add_argument("--gamma_high", type=float, default=1.25)

    parser.add_argument("--clahe_low_clip", type=float, default=1.0)
    parser.add_argument("--clahe_low_grid", type=int, default=16)
    parser.add_argument("--clahe_high_clip", type=float, default=4.0)
    parser.add_argument("--clahe_high_grid", type=int, default=4)

    parser.add_argument("--median_ksize", type=int, default=5)

    parser.add_argument("--bilateral_d", type=int, default=3)
    parser.add_argument("--bilateral_sigma_color", type=float, default=25.0)
    parser.add_argument("--bilateral_sigma_space", type=float, default=50.0)

    parser.add_argument("--nlmeans_h", type=float, default=10.0)
    parser.add_argument("--nlmeans_template", type=int, default=3)
    parser.add_argument("--nlmeans_search", type=int, default=7)

    # Storage
    parser.add_argument("--dtype", type=str, choices=["uint8", "float32"], default="uint8",
                        help="Output dtype for stored arrays (uint8 recommended).")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_path = Path(args.output_path)
    check_dir(out_path)

    videos = list_videos(data_path, args.ext)
    if args.max_videos and args.max_videos > 0:
        videos = videos[: args.max_videos]

    # Infer T, H, W from first valid video
    inferred_T = None
    H = W = None

    valid_videos: List[Path] = []
    skipped: List[Dict[str, str]] = []

    # First pass: infer T/H/W and validate all videos with policy
    for vp in videos:
        try:
            frames, (h, w) = safe_video_read_gray(vp)
            if inferred_T is None:
                inferred_T = len(frames)
                H, W = h, w

            if (h, w) != (H, W):
                skipped.append({"video": vp.name, "reason": f"shape mismatch: {(h, w)} != {(H, W)}"})
                continue

            arr = adjust_frames(frames, inferred_T, args.frame_policy)
            if arr is None:
                skipped.append({"video": vp.name, "reason": f"frame mismatch ({len(frames)} vs {inferred_T}) under policy '{args.frame_policy}'"})
                continue

            valid_videos.append(vp)

        except Exception as e:
            skipped.append({"video": vp.name, "reason": str(e)})

    if inferred_T is None or H is None or W is None:
        show_error_message_box("Could not infer frame count/shape from any readable video.")

    if not valid_videos:
        show_error_message_box("No valid videos after applying frame/shape checks. Check frame_policy or input files.")

    N = len(valid_videos)
    T = inferred_T

    # Build and filter registries
    base_reg = build_registry(args)

    # Filter frame-level ops with --enable/--disable
    reg = filter_registry(base_reg, args.enable, args.disable)

    # Filter projection ops with --enable_projection/--disable_projection
    proj_reg = {"projection": base_reg["projection"]}
    proj_reg = filter_registry(proj_reg, args.enable_projection, args.disable_projection)
    reg["projection"] = proj_reg["projection"]

    if args.projection_mode != "none" and len(reg["projection"]) == 0:
        show_error_message_box("projection_mode is not 'none' but no projection operations are enabled. Check --enable_projection/--disable_projection.")

    # Decide stages and required stages
    stages = ["equalisation", "gamma", "filter"]
    required_stages: List[str] = []

    if args.projection_mode != "none":
        stages.append("projection")
        required_stages.append("projection")  # force a projection op in every pipeline

    pipelines = build_pipelines(reg, stages=stages, required_stages=required_stages)
    if not pipelines:
        show_error_message_box("No pipelines generated. Check enable/disable arguments or projection settings.")

    # Decide output dtype
    out_dtype = np.uint8 if args.dtype == "uint8" else np.float32

    # Run metadata
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Save explicit index mapping for traceability
    video_index_file = write_video_index_map(out_path, valid_videos)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "data_path": str(data_path.resolve()),
        "output_path": str(out_path.resolve()),
        "video_count_total": len(videos),
        "video_count_valid": N,
        "frame_policy": args.frame_policy,
        "T": T, "H": H, "W": W,
        "extensions": args.ext,
        "skipped": skipped,
        "args": vars(args),
        "video_index_file": video_index_file,
        "valid_videos": [{"index": i, "filename": vp.name, "full_path": str(vp.resolve())} for i, vp in enumerate(valid_videos)],
        "pipelines": [],
    }

    def make_safe_name(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_").replace(" ", "")

    # For each pipeline, create memmap(s) and fill video-by-video
    for p_i, pipe in enumerate(pipelines):
        # Split ops: frame ops vs projection op
        frame_ops: List[Operation] = [op for op in pipe.ops if op.stage in ("equalisation", "gamma", "filter")]
        proj_ops: List[ProjectionOperation] = [op for op in pipe.ops if op.stage == "projection"]  # type: ignore

        proj_op: Optional[ProjectionOperation] = proj_ops[0] if proj_ops else None

        # Build base name
        base_name = f"{pipe.pipeline_id}{pipe.alias}"
        base_name = make_safe_name(base_name)

        # Determine file outputs based on projection_mode
        outputs_for_this_pipeline: List[Dict[str, Any]] = []

        if args.projection_mode == "none":
            # One output: full video tensor
            dat_name = f"{base_name}.dat"
            json_name = f"{base_name}.json"
            mm_shape = (N, T, H, W)
            mm = np.memmap(str(out_path / dat_name), dtype=out_dtype, mode="w+", shape=mm_shape)

            for v_i, vp in enumerate(valid_videos):
                try:
                    frames, _ = safe_video_read_gray(vp)
                    arr = adjust_frames(frames, T, args.frame_policy)
                    if arr is None:
                        raise RuntimeError("Unexpected mismatch after validation pass")

                    processed_video = apply_frame_ops(arr, frame_ops)  # (T,H,W) uint8

                    if out_dtype == np.float32:
                        mm[v_i] = processed_video.astype(np.float32) / 255.0
                    else:
                        mm[v_i] = processed_video

                except Exception as e:
                    show_error_message_box(f"Failed processing video '{vp.name}' for pipeline '{base_name}': {e}")

            mm.flush()

            pipe_meta = {
                "pipeline_id": pipe.pipeline_id,
                "alias": pipe.alias,
                "ops": [{"name": op.name, "alias": op.alias, "stage": op.stage, "params": op.params} for op in pipe.ops],
                "shape": list(mm_shape),
                "dtype": args.dtype,
                "file": dat_name,
                "output_kind": "full",
            }
            with open(out_path / json_name, "w", encoding="utf-8") as f:
                json.dump(pipe_meta, f, indent=2)

            outputs_for_this_pipeline.append(pipe_meta)

            print(f"[{p_i+1}/{len(pipelines)}] saved: {dat_name} shape={mm_shape} ops={[op.alias for op in pipe.ops]}")

        else:
            # projected_only or both requires a projection op (enforced earlier)
            if proj_op is None:
                show_error_message_box(f"Projection mode is '{args.projection_mode}' but pipeline '{base_name}' has no projection op.")

            proj_suffix = f"proj-{proj_op.alias}"
            proj_base = make_safe_name(f"{base_name}__{proj_suffix}")

            # If both, write full first
            if args.projection_mode == "both":
                dat_name_full = f"{base_name}__full.dat"
                json_name_full = f"{base_name}__full.json"
                mm_shape_full = (N, T, H, W)
                mm_full = np.memmap(str(out_path / dat_name_full), dtype=out_dtype, mode="w+", shape=mm_shape_full)

                # We'll also compute projection in the same pass to avoid rereading videos twice
                dat_name_proj = f"{proj_base}.dat"
                json_name_proj = f"{proj_base}.json"
                mm_shape_proj = (N, H, W)
                mm_proj = np.memmap(str(out_path / dat_name_proj), dtype=out_dtype, mode="w+", shape=mm_shape_proj)

                for v_i, vp in enumerate(valid_videos):
                    try:
                        frames, _ = safe_video_read_gray(vp)
                        arr = adjust_frames(frames, T, args.frame_policy)
                        if arr is None:
                            raise RuntimeError("Unexpected mismatch after validation pass")

                        processed_video = apply_frame_ops(arr, frame_ops)  # (T,H,W) uint8
                        projected = apply_projection(processed_video, proj_op, out_dtype)  # (H,W)

                        if out_dtype == np.float32:
                            mm_full[v_i] = processed_video.astype(np.float32) / 255.0
                            # projected already float32 in [0,1] by apply_projection
                            mm_proj[v_i] = projected.astype(np.float32)
                        else:
                            mm_full[v_i] = processed_video
                            mm_proj[v_i] = projected.astype(np.uint8)

                    except Exception as e:
                        show_error_message_box(f"Failed processing video '{vp.name}' for pipeline '{base_name}': {e}")

                mm_full.flush()
                mm_proj.flush()

                pipe_meta_full = {
                    "pipeline_id": pipe.pipeline_id,
                    "alias": pipe.alias,
                    "ops": [{"name": op.name, "alias": op.alias, "stage": op.stage, "params": op.params} for op in pipe.ops],
                    "shape": list(mm_shape_full),
                    "dtype": args.dtype,
                    "file": dat_name_full,
                    "output_kind": "full",
                }
                with open(out_path / json_name_full, "w", encoding="utf-8") as f:
                    json.dump(pipe_meta_full, f, indent=2)

                pipe_meta_proj = {
                    "pipeline_id": pipe.pipeline_id,
                    "alias": pipe.alias,
                    "projection": {"name": proj_op.name, "alias": proj_op.alias, "params": proj_op.params},
                    "ops": [{"name": op.name, "alias": op.alias, "stage": op.stage, "params": op.params} for op in pipe.ops],
                    "shape": list(mm_shape_proj),
                    "dtype": args.dtype,
                    "file": dat_name_proj,
                    "output_kind": "projected",
                }
                with open(out_path / json_name_proj, "w", encoding="utf-8") as f:
                    json.dump(pipe_meta_proj, f, indent=2)

                outputs_for_this_pipeline.extend([pipe_meta_full, pipe_meta_proj])

                print(f"[{p_i+1}/{len(pipelines)}] saved: {dat_name_full} shape={mm_shape_full} ops={[op.alias for op in pipe.ops]}")
                print(f"[{p_i+1}/{len(pipelines)}] saved: {dat_name_proj} shape={mm_shape_proj} projection={proj_op.alias}")

            else:
                # projected_only: only projected output
                dat_name = f"{proj_base}.dat"
                json_name = f"{proj_base}.json"
                mm_shape = (N, H, W)
                mm = np.memmap(str(out_path / dat_name), dtype=out_dtype, mode="w+", shape=mm_shape)

                for v_i, vp in enumerate(valid_videos):
                    try:
                        frames, _ = safe_video_read_gray(vp)
                        arr = adjust_frames(frames, T, args.frame_policy)
                        if arr is None:
                            raise RuntimeError("Unexpected mismatch after validation pass")

                        processed_video = apply_frame_ops(arr, frame_ops)  # (T,H,W) uint8
                        projected = apply_projection(processed_video, proj_op, out_dtype)  # (H,W)

                        mm[v_i] = projected

                    except Exception as e:
                        show_error_message_box(f"Failed processing video '{vp.name}' for pipeline '{base_name}': {e}")

                mm.flush()

                pipe_meta = {
                    "pipeline_id": pipe.pipeline_id,
                    "alias": pipe.alias,
                    "projection": {"name": proj_op.name, "alias": proj_op.alias, "params": proj_op.params},
                    "ops": [{"name": op.name, "alias": op.alias, "stage": op.stage, "params": op.params} for op in pipe.ops],
                    "shape": list(mm_shape),
                    "dtype": args.dtype,
                    "file": dat_name,
                    "output_kind": "projected",
                }
                with open(out_path / json_name, "w", encoding="utf-8") as f:
                    json.dump(pipe_meta, f, indent=2)

                outputs_for_this_pipeline.append(pipe_meta)

                print(f"[{p_i+1}/{len(pipelines)}] saved: {dat_name} shape={mm_shape} projection={proj_op.alias} ops={[op.alias for op in pipe.ops]}")

        manifest["pipelines"].append({
            "pipeline_id": pipe.pipeline_id,
            "alias": pipe.alias,
            "ops": [{"name": op.name, "alias": op.alias, "stage": op.stage, "params": op.params} for op in pipe.ops],
            "outputs": outputs_for_this_pipeline,
        })

    # Save global manifest
    manifest_path = out_path / f"pipelines_manifest_{run_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Manifest: {manifest_path.name}")
    print(f"Valid videos: {N} | Skipped: {len(skipped)} | Pipelines: {len(pipelines)}")
    print(f"Video index map: {video_index_file}")


if __name__ == "__main__":
    main()
