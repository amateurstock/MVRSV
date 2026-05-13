#!/usr/bin/env python3
"""
Research-oriented River Surface Velocity Estimation Server

Core improvements over the previous version:
1. Source FPS and processing FPS are separated.
2. Velocity uses frame_dt, not UI/display FPS.
3. PIV uses interrogation-window template matching instead of Farneback-only dense flow.
4. Velocity is estimated with robust median + MAD outlier rejection.
5. YOLO is throttled to reduce FPS drops.
6. CSV validation logging is built in.
7. Model reset is avoided unless the source changes.

Notes for research reporting:
- The PIV method below is a practical cross-correlation/window matching PIV approximation.
- For publication-grade PIV, consider OpenPIV or a full FFT-based cross-correlation pipeline.
"""

import csv
import gc
import math as mt
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from flask import Flask, Response, jsonify, render_template, request, abort
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ============================================================
# APP AND PATHS
# ============================================================

app = Flask(__name__)

VIDEO_SRC = "./test/AlpineStabilised.avi"
MORPHOLOGY_MODEL_PATH = "models/morphology_model.pt"
TRACER_MODEL_PATH = "models/tracer_model.pt"
MODEL_FOLDER = Path("./models")
UPLOAD_FOLDER = Path("./uploads")
LOG_FOLDER = Path("./logs")
VALIDATION_CSV = LOG_FOLDER / "river_velocity_validation_log.csv"
ALLOWED_VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
ALLOWED_MODEL_EXTENSIONS = {".pt", ".onnx"}
MAX_CAMERA_SCAN_INDEX = 10
PROFILE_HISTORY_LIMIT = 120

UPLOAD_FOLDER.mkdir(exist_ok=True)
LOG_FOLDER.mkdir(exist_ok=True)

lock = threading.Lock()

# ============================================================
# GLOBAL STATE
# ============================================================

current_video_src = VIDEO_SRC
current_source_kind = "video"
requested_seek_frame = None
active_stream_token = 0
selected_morphology_model_path = MORPHOLOGY_MODEL_PATH
selected_tracer_model_path = TRACER_MODEL_PATH

playback_info = {
    "source_kind": current_source_kind,
    "current_frame": 0,
    "total_frames": 0,
    "fps": 0.0,
    "duration": 0.0,
    "is_paused": False,
}

profile_history = deque(maxlen=PROFILE_HISTORY_LIMIT)
profile_last = {}
validation_rows = deque(maxlen=2000)

# ============================================================
# RESEARCH-TUNED PARAMETERS
# ============================================================

global_vars = {
    # Camera and orthorectification
    "cam_fov": 72.4,
    "is_ortho": False,
    "H": None,
    "ortho_status": "VGCP calibration optional; using raw scale",
    "f_p": 1.0,
    "px_scale": 50,                 # pixels per meter in orthorectified output
    "meters_per_pixel": 0.02,       # used directly without VGCPs; overwritten to 1 / px_scale when H is ready

    # Video and display
    "video_w": 1920,
    "video_h": 1080,
    "target_sz": 640,
    "stream_max_width": 640,
    "target_stream_fps": 20.0,
    "is_paused": False,

    # YOLO throttling
    "morphology_threshold": 0.35,
    "tracer_threshold": 0.25,
    "morphology_imgsz": 512,
    "tracer_imgsz": 640,
    "tracer_target_kind": "all",
    "morphology_interval": 60,
    "detect_interval": 10,
    "morphology_mask_erode": 2,

    # Tracer optical flow
    "min_tracked_points": 2,
    "tracer_trail_length": 24,

    # Velocity validity
    "max_velocity": 8.0,
    "min_velocity": 0.01,
    "velocity_smoothing_window": 8,

    # PIV settings - tuned for stability more than raw FPS
    "enable_piv": True,
    "piv_interval": 5,
    "piv_grid_step": 40,
    "piv_window_size": 64,
    "piv_search_size": 112,
    "piv_min_corr": 0.45,
    "piv_min_vectors": 4,
    "piv_max_displacement_px": 80,

    # STIV fallback - simplified line correlation
    "enable_stiv": True,
    "stiv_history": 48,
    "stiv_start_x": 0.05,
    "stiv_start_y": 0.50,
    "stiv_end_x": 0.95,
    "stiv_end_y": 0.50,

    # Model selection
    "morphology_model_path": MORPHOLOGY_MODEL_PATH,
    "tracer_model_path": TRACER_MODEL_PATH,
}

global_vgcps = []

# ============================================================
# MODEL STATE
# ============================================================

morphology_model = None
tracer_model = None
morphology_model_loaded_path = None
tracer_model_loaded_path = None
inference_device = None
inference_device_label = None
onnx_cuda_available = None


def get_inference_device():
    global inference_device, inference_device_label
    if inference_device is None:
        inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        inference_device_label = (
            f"{inference_device} {torch.cuda.get_device_name(0)}"
            if inference_device.startswith("cuda")
            else "cpu"
        )
        print(f"YOLO inference device: {inference_device_label}")
    return inference_device


def get_inference_device_label():
    get_inference_device()
    return inference_device_label or "unknown"


def reset_yolo_models():
    global morphology_model, tracer_model, morphology_model_loaded_path, tracer_model_loaded_path
    morphology_model = None
    tracer_model = None
    morphology_model_loaded_path = None
    tracer_model_loaded_path = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_onnx_model(path):
    return Path(str(path)).suffix.lower() == ".onnx"


def can_use_onnx_cuda():
    global onnx_cuda_available
    if onnx_cuda_available is not None:
        return onnx_cuda_available
    try:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available in ONNX Runtime.")

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "cuda_probe", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10

        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            onnx.save(model, f.name)
            session = ort.InferenceSession(
                f.name,
                providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
            )
            onnx_cuda_available = "CUDAExecutionProvider" in session.get_providers()
    except Exception as exc:
        print(f"ONNX CUDA unavailable: {exc}")
        onnx_cuda_available = False
    return onnx_cuda_available


def get_model_predict_device(path):
    if is_onnx_model(path):
        if not can_use_onnx_cuda():
            raise RuntimeError(
                f"ONNX model '{path}' requires CUDAExecutionProvider, but ONNX CUDA is unavailable."
            )
        return "cuda:0"
    return get_inference_device()


def predict_yolo_model(model, frame, conf, imgsz, model_path):
    global onnx_cuda_available
    device = get_model_predict_device(model_path)
    try:
        return model.predict(
            frame,
            conf=conf,
            imgsz=imgsz,
            device=device,
            half=device.startswith("cuda"),
            verbose=False,
        )[0]
    except Exception as exc:
        if is_onnx_model(model_path) and device.startswith("cuda"):
            onnx_cuda_available = False
            raise RuntimeError(f"ONNX CUDA inference failed for '{model_path}'.") from exc
        raise


def list_available_models():
    models = []
    for path in sorted(MODEL_FOLDER.glob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED_MODEL_EXTENSIONS:
            models.append(str(path))
    return models


def validate_model_path(path, default_path):
    candidate = Path(str(path or default_path))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(MODEL_FOLDER.resolve())
    except (OSError, ValueError):
        return default_path
    if resolved.suffix.lower() not in ALLOWED_MODEL_EXTENSIONS:
        return default_path
    if not resolved.exists() or not resolved.is_file():
        return default_path
    return str(resolved.relative_to(Path.cwd()))


def get_morphology_model():
    global morphology_model, morphology_model_loaded_path
    with lock:
        path = selected_morphology_model_path
    if morphology_model is None or morphology_model_loaded_path != path:
        morphology_model = YOLO(path, task="segment")
        morphology_model_loaded_path = path
        if not is_onnx_model(path):
            morphology_model.to(get_inference_device())
    return morphology_model


def get_tracer_model():
    global tracer_model, tracer_model_loaded_path
    with lock:
        path = selected_tracer_model_path
    if tracer_model is None or tracer_model_loaded_path != path:
        tracer_model = YOLO(path, task="detect")
        tracer_model_loaded_path = path
        if not is_onnx_model(path):
            tracer_model.to(get_inference_device())
    return tracer_model


def cull_streams(reset_models=False):
    """Restart active streams. Avoid model reset unless source/model changes."""
    global active_stream_token, requested_seek_frame
    with lock:
        active_stream_token += 1
        requested_seek_frame = None
        profile_history.clear()
        profile_last.clear()
    if reset_models:
        reset_yolo_models()
    return active_stream_token


def jsonify_safe_settings():
    settings = {}

    for key, value in global_vars.items():
        if key == "H":
            settings[key] = value.tolist() if value is not None else None
        elif isinstance(value, np.generic):
            settings[key] = value.item()
        else:
            settings[key] = value

    return settings

# ============================================================
# DATA CLASSES
# ============================================================

class VGCP:
    def __init__(self, data):
        self.x = float(data.get("x", 0) or 0)
        self.y = float(data.get("y", 0) or 0)
        self.r = float(data.get("r", 0) or 0)


@dataclass
class PipelineState:
    prev_gray: MatLike | None = None
    track_pts: MatLike | None = None
    track_labels: list[str] = field(default_factory=list)
    tracer_trails: list[list[tuple[int, int]]] = field(default_factory=list)
    morphology_polygons: list[np.ndarray] = field(default_factory=list)
    morphology_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    stiv_rows: deque = field(default_factory=deque)
    velocity_history: deque = field(default_factory=deque)
    latest_piv_velocity: float | None = None
    latest_stiv_velocity: float | None = None
    latest_tracer_velocity: float | None = None
    latest_tracer_count: int = 0
    last_payload: bytes | None = None
    last_frame_shape: tuple[int, int] | None = None
    last_tracer_detection_frame: int = -1_000_000
    prev_frame_index: int | None = None
    prev_capture_time: float | None = None

    def reset_tracking(self):
        self.prev_gray = None
        self.track_pts = None
        self.track_labels = []
        self.tracer_trails = []
        self.last_tracer_detection_frame = -1_000_000
        self.prev_frame_index = None
        self.prev_capture_time = None
        self.latest_tracer_velocity = None
        self.latest_tracer_count = 0

    def reset_for_shape(self, shape):
        if shape == self.last_frame_shape:
            return
        self.reset_tracking()
        self.morphology_polygons = []
        self.morphology_boxes = []
        self.stiv_rows.clear()
        self.velocity_history.clear()
        self.latest_piv_velocity = None
        self.latest_stiv_velocity = None
        self.last_frame_shape = shape

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_float(data, key, default, min_value=None, max_value=None):
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def get_int(data, key, default, min_value=None, max_value=None):
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        value = int(default)
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def get_bool(data, key, default=False):
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def is_allowed_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def resize_for_stream(frame):
    max_width = int(global_vars["stream_max_width"])
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def get_capture_metadata(cap, source_kind):
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if source_kind == "video" else 0
    duration = total_frames / fps if source_kind == "video" and fps > 0 and total_frames > 0 else 0.0
    return {
        "source_kind": source_kind,
        "current_frame": 0,
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
    }


def update_playback_info(**kwargs):
    with lock:
        playback_info.update(kwargs)


def record_stage(timings, name, started_at):
    now = time.perf_counter()
    timings[name] = timings.get(name, 0.0) + (now - started_at) * 1000.0
    return now


def record_profile(timings, frame_counter, frame_shape):
    total_ms = (time.perf_counter() - timings.pop("_frame_started_at")) * 1000.0
    timings["total_ms"] = total_ms
    timings["frame"] = int(frame_counter)
    if frame_shape is not None:
        timings["frame_w"] = int(frame_shape[1])
        timings["frame_h"] = int(frame_shape[0])
    with lock:
        profile_last.clear()
        profile_last.update(timings)
        profile_history.append(dict(timings))


def summarize_profile():
    with lock:
        samples = list(profile_history)
        last = dict(profile_last)
    stage_names = sorted({key for sample in samples for key in sample if key.endswith("_ms")})
    summary = {}
    for name in stage_names:
        values = [float(sample[name]) for sample in samples if name in sample]
        if values:
            summary[name] = {
                "last_ms": round(float(last.get(name, values[-1])), 3),
                "avg_ms": round(float(sum(values) / len(values)), 3),
                "max_ms": round(float(max(values)), 3),
                "samples": len(values),
            }
    return {"sample_count": len(samples), "last_frame": int(last.get("frame", 0) or 0), "stages": summary}


def init_cam_dims():
    with lock:
        src = current_video_src
        kind = current_source_kind
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    success, frame = cap.read()
    if not success or frame is None:
        cap.release()
        raise RuntimeError("Cannot read frame from source.")
    frame = resize_for_stream(frame)
    h, w = frame.shape[:2]
    metadata = get_capture_metadata(cap, kind)
    with lock:
        global_vars["video_w"] = w
        global_vars["video_h"] = h
        playback_info.update(metadata)
    cap.release()

# ============================================================
# ORTHORECTIFICATION
# ============================================================

def calc_f_p(w_fov, w_px):
    return w_px / (2 * mt.tan(mt.radians(w_fov / 2)))


def calc_homo(vgcps, focal_px, img_w, img_h):
    if len(vgcps) < 4:
        return None, "Need at least 4 VGCPs"
    if any(p.r <= 0 for p in vgcps):
        return None, "Each VGCP needs a positive R distance"

    image_pts = []
    world_pts = []
    cx, cy = img_w / 2, img_h / 2

    for p in vgcps:
        image_pts.append([p.x, p.y])
        rel_x_px = p.x - cx
        rel_y_px = p.y - cy
        theta_a = np.arctan2(rel_x_px, focal_px)
        theta_e = np.arctan2(rel_y_px, focal_px)
        r = np.float64(p.r)
        world_x = r * np.cos(theta_e) * np.sin(theta_a)
        world_y = r * np.cos(theta_e) * np.cos(theta_a)
        world_pts.append([world_x, world_y])

    src = np.array(image_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)

    if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
        return None, "VGCP image points are degenerate"
    if np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2:
        return None, "VGCP real-world points are degenerate"

    dst = (dst - np.min(dst, axis=0)) * global_vars["px_scale"] + 100
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or not np.all(np.isfinite(H)):
        return None, "Homography solve failed"
    return H, "Ready"


def update_homo():
    global_vars["f_p"] = calc_f_p(global_vars["cam_fov"], global_vars["video_w"])
    if len(global_vgcps) < 4:
        global_vars["H"] = None
        global_vars["ortho_status"] = "VGCP calibration optional; using raw scale"
        return
    H, status = calc_homo(global_vgcps, global_vars["f_p"], global_vars["video_w"], global_vars["video_h"])
    global_vars["H"] = H
    global_vars["ortho_status"] = status
    if H is not None:
        global_vars["meters_per_pixel"] = 1.0 / max(float(global_vars["px_scale"]), 1e-6)


def is_metric_ortho_ready():
    return global_vars["H"] is not None


def calibration_status_payload():
    ready = global_vars["H"] is not None
    meters_per_pixel = float(global_vars["meters_per_pixel"])
    pixels_per_meter = None if meters_per_pixel <= 0 else 1.0 / meters_per_pixel
    return {
        "ok": True,
        "ortho_ready": ready,
        "ortho_status": global_vars["ortho_status"],
        "meters_per_pixel": meters_per_pixel,
        "pixels_per_meter": pixels_per_meter,
        "vgcp_count": len(global_vgcps),
    }

# ============================================================
# VELOCITY STATISTICS
# ============================================================

def is_valid_velocity(v):
    return np.isfinite(v) and global_vars["min_velocity"] <= v <= global_vars["max_velocity"]


def robust_median(values, mad_multiplier=3.5):
    """Median after MAD outlier rejection."""
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, 0, None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 1e-9:
        filtered = arr
    else:
        robust_z = 0.6745 * np.abs(arr - med) / mad
        filtered = arr[robust_z <= mad_multiplier]
    if filtered.size == 0:
        return None, 0, None
    return float(np.median(filtered)), int(filtered.size), float(np.std(filtered))


def smooth_velocity(state: PipelineState, velocity):
    if velocity is None:
        return None
    maxlen = max(1, int(global_vars["velocity_smoothing_window"]))
    if state.velocity_history.maxlen != maxlen:
        state.velocity_history = deque(state.velocity_history, maxlen=maxlen)
    state.velocity_history.append(float(velocity))
    return float(np.median(np.asarray(state.velocity_history, dtype=np.float32)))

# ============================================================
# MORPHOLOGY AND TRACER HELPERS
# ============================================================

def shrink_polygons(polygons, shape, erode_px):
    if erode_px <= 0 or not polygons:
        return polygons
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [polygon], 255)
    kernel_size = (erode_px * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.erode(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).astype(np.int32) for c in contours if len(c) >= 3]


def extract_morphology_overlay(result, frame_shape):
    polygons, boxes = [], []
    erode_px = max(0, int(global_vars["morphology_mask_erode"]))
    if result.masks is not None:
        for polygon in result.masks.xy:
            polygon = np.asarray(polygon, dtype=np.int32)
            if len(polygon) >= 3:
                polygons.append(polygon)
        polygons = shrink_polygons(polygons, frame_shape, erode_px)
    elif result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return polygons, boxes


def draw_morphology_overlay(frame, polygons, boxes):
    if not polygons and not boxes:
        return
    overlay = frame.copy()
    for polygon in polygons:
        cv2.fillPoly(overlay, [polygon], (255, 160, 40))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    for polygon in polygons:
        cv2.polylines(frame, [polygon], True, (255, 210, 80), 2)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 210, 80), 2)


def extract_tracer_points(result):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        class_id = int(box.cls[0]) if box.cls is not None else -1
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        kind = result.names.get(class_id, f"class_{class_id}")
        detections.append({"point": [cx, cy], "kind": kind, "confidence": confidence})
    return detections


def filter_tracer_detections(detections):
    target = str(global_vars.get("tracer_target_kind", "all") or "all").strip()

    if not target or target.lower() == "all":
        return detections

    allowed = {
        item.strip().lower()
        for item in target.split(",")
        if item.strip()
    }

    if not allowed:
        return detections

    return [
        detection
        for detection in detections
        if str(detection.get("kind", "")).lower() in allowed
    ]


def draw_tracer_points(frame, detections):
    for det in detections:
        cx, cy = det["point"]
        label = f"{det['kind']} {det['confidence']:.2f}"
        cv2.circle(frame, (cx, cy), 9, (0, 80, 255), 2)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
        cv2.putText(frame, label, (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)


def draw_tracer_trails(frame, trails):
    for trail in trails:
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (0, 200, 255), 2)


def build_velocity_mask(shape, polygons, boxes):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], 255)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    if np.count_nonzero(mask) < 16:
        mask.fill(255)
    return mask


def point_in_mask(mask, x, y):
    if mask is None:
        return False
    h, w = mask.shape[:2]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    return 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0


# ============================================================
# RESEARCH-ORIENTED PIV
# ============================================================

def contrast_enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def subpixel_peak_offset(response, loc):
    """Parabolic 3-point peak refinement for template-correlation PIV."""
    peak_x, peak_y = loc
    h, w = response.shape[:2]
    center = float(response[peak_y, peak_x])
    offset_x = 0.0
    offset_y = 0.0

    if 0 < peak_x < w - 1:
        left = float(response[peak_y, peak_x - 1])
        right = float(response[peak_y, peak_x + 1])
        denom = left - (2.0 * center) + right
        if abs(denom) > 1e-9:
            offset_x = 0.5 * (left - right) / denom

    if 0 < peak_y < h - 1:
        top = float(response[peak_y - 1, peak_x])
        bottom = float(response[peak_y + 1, peak_x])
        denom = top - (2.0 * center) + bottom
        if abs(denom) > 1e-9:
            offset_y = 0.5 * (top - bottom) / denom

    return float(np.clip(offset_x, -1.0, 1.0)), float(np.clip(offset_y, -1.0, 1.0))


def match_piv_window(src, dst, x, y, half_w, half_s, window, search, min_corr):
    src_win = src[y - half_w:y + half_w, x - half_w:x + half_w]
    dst_search = dst[y - half_s:y + half_s, x - half_s:x + half_s]

    if src_win.shape != (window, window) or dst_search.shape != (search, search):
        return None

    response = cv2.matchTemplate(dst_search, src_win, cv2.TM_CCOEFF_NORMED)
    _, corr, _, loc = cv2.minMaxLoc(response)
    if corr < min_corr:
        return None

    sub_x, sub_y = subpixel_peak_offset(response, loc)
    match_x = (x - half_s) + loc[0] + sub_x + half_w
    match_y = (y - half_s) + loc[1] + sub_y + half_w
    return float(match_x - x), float(match_y - y), float(corr)


def robust_piv_velocity(vectors, frame_dt):
    if not vectors:
        return None, 0, None

    arr = np.asarray(vectors, dtype=np.float32)
    dx = arr[:, 2]
    dy = arr[:, 3]
    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))
    residuals = np.hypot(dx - med_dx, dy - med_dy)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))

    if mad > 1e-9:
        robust_z = 0.6745 * residuals / mad
        arr = arr[robust_z <= 3.5]

    if arr.size == 0:
        return None, 0, None

    speeds = np.hypot(arr[:, 2], arr[:, 3]) * float(global_vars["meters_per_pixel"]) / max(frame_dt, 1e-6)
    speeds = speeds[np.isfinite(speeds)]
    speeds = speeds[(speeds >= float(global_vars["min_velocity"])) & (speeds <= float(global_vars["max_velocity"]))]
    if speeds.size == 0:
        return None, 0, None

    return float(np.median(speeds)), int(speeds.size), float(np.std(speeds))


def estimate_piv_velocity(prev_gray, gray, mask, frame_dt):
    """
    Practical interrogation-window PIV using normalized cross-correlation.

    Research choices:
    - Uses water mask only.
    - Uses window matching instead of dense full-frame optical flow.
    - Refines the correlation peak to subpixel precision.
    - Rejects poor-correlation, unrealistic, and forward/back inconsistent vectors.
    - Uses vector MAD rejection before converting the robust displacement to speed.
    """
    if frame_dt <= 0:
        return None, np.zeros((*gray.shape, 2), dtype=np.float32), 0, None

    prev = contrast_enhance(prev_gray)
    curr = contrast_enhance(gray)

    h, w = gray.shape[:2]
    window = int(global_vars["piv_window_size"])
    search = int(global_vars["piv_search_size"])
    step = int(global_vars["piv_grid_step"])
    min_corr = float(global_vars["piv_min_corr"])
    max_disp = float(global_vars["piv_max_displacement_px"])
    mpp = float(global_vars["meters_per_pixel"])

    window = max(16, window)
    if window % 2:
        window += 1
    search = max(window + 8, search)
    if search % 2:
        search += 1
    step = max(12, step)

    half_w = window // 2
    half_s = search // 2
    display_flow = np.zeros((h, w, 2), dtype=np.float32)
    vectors = []

    # Avoid borders where the search window would exceed the image.
    for y in range(half_s, h - half_s, step):
        for x in range(half_s, w - half_s, step):
            if mask[y, x] == 0:
                continue

            prev_win = prev[y - half_w:y + half_w, x - half_w:x + half_w]
            curr_search = curr[y - half_s:y + half_s, x - half_s:x + half_s]
            mask_win = mask[y - half_w:y + half_w, x - half_w:x + half_w]

            if prev_win.shape != (window, window) or curr_search.shape != (search, search):
                continue
            if np.count_nonzero(mask_win) < 0.50 * mask_win.size:
                continue
            if float(np.std(prev_win)) < 4.0:
                continue

            forward = match_piv_window(prev, curr, x, y, half_w, half_s, window, search, min_corr)
            if forward is None:
                continue

            dx, dy, corr = forward
            disp_px = mt.hypot(dx, dy)

            if disp_px < 0.25 or disp_px > max_disp:
                continue

            match_x = int(round(x + dx))
            match_y = int(round(y + dy))
            if not (half_s <= match_x < w - half_s and half_s <= match_y < h - half_s):
                continue
            if not point_in_mask(mask, match_x, match_y):
                continue

            backward = match_piv_window(curr, prev, match_x, match_y, half_w, half_s, window, search, min_corr)
            if backward is None:
                continue

            back_dx, back_dy, _ = backward
            if mt.hypot(dx + back_dx, dy + back_dy) > max(1.5, 0.20 * disp_px):
                continue

            velocity = disp_px * mpp / max(frame_dt, 1e-6)
            if not is_valid_velocity(velocity):
                continue

            vectors.append((x, y, dx, dy, corr, velocity))
            display_flow[y, x, 0] = dx
            display_flow[y, x, 1] = dy

    if len(vectors) < int(global_vars["piv_min_vectors"]):
        return None, display_flow, len(vectors), None

    piv_velocity, kept_count, piv_std = robust_piv_velocity(vectors, frame_dt)
    if piv_velocity is None:
        return None, display_flow, len(vectors), None

    return piv_velocity, display_flow, kept_count, piv_std


def draw_piv_vectors(frame, flow, mask):
    step = max(12, int(global_vars["piv_grid_step"]))
    h, w = mask.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if mask[y, x] == 0:
                continue
            dx, dy = flow[y, x]
            mag = mt.hypot(float(dx), float(dy))
            if mag < 0.25:
                continue
            end = (int(x + dx * 2.5), int(y + dy * 2.5))
            cv2.arrowedLine(frame, (x, y), end, (255, 80, 255), 1, tipLength=0.35)

# ============================================================
# STIV FALLBACK
# ============================================================

def estimate_stiv_velocity_from_stack(stiv_stack, frame_dt):
    """
    Estimate STIV line velocity from the slope of texture streaks in a space-time image.

    The brightness-constancy relation is I_t + u * I_x = 0, where u is pixels per
    frame along the STIV sampling line. A robust median of local -I_t/I_x samples is
    less brittle than the previous two-frame phase shift.
    """
    if stiv_stack.shape[0] < 6 or stiv_stack.shape[1] < 16 or frame_dt <= 0:
        return None

    stack = stiv_stack.astype(np.float32)
    if float(np.std(stack)) < 2.0:
        return None

    stack = cv2.GaussianBlur(stack, (5, 3), 0)
    grad_x = cv2.Sobel(stack, cv2.CV_32F, 1, 0, ksize=3)
    grad_t = cv2.Sobel(stack, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.hypot(grad_x, grad_t)

    texture_threshold = max(1e-6, float(np.percentile(texture, 60)))
    x_threshold = max(1e-6, float(np.percentile(np.abs(grad_x), 50)))
    valid = (texture >= texture_threshold) & (np.abs(grad_x) >= x_threshold)
    if np.count_nonzero(valid) < 32:
        return None

    px_per_frame_samples = -grad_t[valid] / grad_x[valid]
    px_per_frame_samples = px_per_frame_samples[np.isfinite(px_per_frame_samples)]
    if px_per_frame_samples.size < 32:
        return None

    mpp = max(float(global_vars["meters_per_pixel"]), 1e-8)
    max_px_per_frame = max(1.0, float(global_vars["max_velocity"]) * frame_dt / mpp)
    px_per_frame_samples = px_per_frame_samples[np.abs(px_per_frame_samples) <= max_px_per_frame]
    if px_per_frame_samples.size < 16:
        return None

    px_per_frame, kept_count, _ = robust_median(px_per_frame_samples)
    if px_per_frame is None or kept_count < 16:
        return None

    velocity = abs(float(px_per_frame)) * mpp / max(frame_dt, 1e-6)
    return velocity if is_valid_velocity(velocity) else None


def update_stiv_velocity(stiv_rows, gray, mask, frame_dt):
    h, w = gray.shape[:2]
    x1 = int(round(global_vars["stiv_start_x"] * (w - 1)))
    y1 = int(round(global_vars["stiv_start_y"] * (h - 1)))
    x2 = int(round(global_vars["stiv_end_x"] * (w - 1)))
    y2 = int(round(global_vars["stiv_end_y"] * (h - 1)))
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    length_px = mt.hypot(int(x2) - int(x1), int(y2) - int(y1))
    if length_px < 16:
        return None, (int(x1), int(y1)), (int(x2), int(y2))

    n = int(round(length_px)) + 1
    xs = np.linspace(x1, x2, n, dtype=np.float32).reshape(1, -1)
    ys = np.linspace(y1, y2, n, dtype=np.float32).reshape(1, -1)
    line = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float32)
    mask_line = cv2.remap(mask, xs, ys, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1)
    valid_line = mask_line > 0
    valid_indexes = np.flatnonzero(valid_line)
    if len(valid_indexes) < 16:
        return None, (int(x1), int(y1)), (int(x2), int(y2))

    segment_breaks = np.where(np.diff(valid_indexes) > 1)[0] + 1
    segments = np.split(valid_indexes, segment_breaks)
    segment = max(segments, key=len)
    if len(segment) < 16:
        return None, (int(x1), int(y1)), (int(x2), int(y2))

    flat_xs = xs.reshape(-1)
    flat_ys = ys.reshape(-1)
    used_start = (int(round(float(flat_xs[segment[0]]))), int(round(float(flat_ys[segment[0]]))))
    used_end = (int(round(float(flat_xs[segment[-1]]))), int(round(float(flat_ys[segment[-1]]))))
    line = line[segment[0]:segment[-1] + 1].reshape(1, -1).astype(np.float32)

    stiv_rows.append(line)
    if len(stiv_rows) < 6:
        return None, used_start, used_end

    widths = [row.shape[1] for row in stiv_rows]
    common_width = min(widths)
    if common_width < 16:
        return None, used_start, used_end

    stiv_stack = np.vstack([row[:, :common_width] for row in stiv_rows])
    velocity = estimate_stiv_velocity_from_stack(stiv_stack, frame_dt)
    return velocity, used_start, used_end

# ============================================================
# DRAWING AND LOGGING
# ============================================================

def fmt_v(v):
    return f"{v:.2f} m/s" if v is not None else "-- m/s"


def draw_hud(frame, surface_velocity, source, tracer_velocity, piv_velocity, stiv_velocity,
             tracer_count, tracked_points, piv_vectors, source_fps, processing_fps, frame_dt, device_label):
    lines = [
        f"Surface: {fmt_v(surface_velocity)} ({source})",
        f"Tracer: {fmt_v(tracer_velocity)} | PIV: {fmt_v(piv_velocity)} | STIV: {fmt_v(stiv_velocity)}",
        f"Tracers: {tracer_count} | Tracked pts: {tracked_points} | PIV vectors: {piv_vectors}",
        f"Source FPS: {source_fps:.2f} | Processing FPS: {processing_fps:.2f} | dt: {frame_dt:.4f}s",
        f"m/px: {global_vars['meters_per_pixel']:.5f} | YOLO: {device_label}",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        y += 20


def append_validation_log(row):
    validation_rows.append(row)

    fieldnames = list(row.keys())
    existing_rows = []
    write_mode = "a"
    write_header = not VALIDATION_CSV.exists()

    if VALIDATION_CSV.exists():
        with VALIDATION_CSV.open(newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = [name for name in (reader.fieldnames or []) if name]
            if existing_fieldnames != fieldnames:
                existing_rows = [
                    {key: value for key, value in old_row.items() if key is not None}
                    for old_row in reader
                ]
                for name in existing_fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
                write_mode = "w"
                write_header = True

    with VALIDATION_CSV.open(write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            if existing_rows:
                writer.writerows(existing_rows)
        writer.writerow(row)


def reset_validation_storage():
    validation_rows.clear()
    if VALIDATION_CSV.exists():
        VALIDATION_CSV.unlink()


def numeric_value(value):
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def metric_error(metric_value, reference_value):
    if metric_value is None or reference_value is None:
        return None, None
    error = float(metric_value) - float(reference_value)
    percent_error = None if abs(reference_value) <= 1e-12 else abs(error / float(reference_value)) * 100.0
    return round(error, 6), None if percent_error is None else round(percent_error, 3)


def summarize_numeric_series(rows, key, reference_value=None):
    values = [numeric_value(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return {
            "count": 0,
            "latest": None,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "error": None,
            "percent_error": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    median = round(float(np.median(arr)), 6)
    error, percent_error = metric_error(median, reference_value)
    return {
        "count": int(arr.size),
        "latest": round(float(arr[-1]), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": median,
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "error": error,
        "percent_error": percent_error,
    }


def load_validation_rows():
    if VALIDATION_CSV.exists():
        with VALIDATION_CSV.open(newline="") as f:
            return list(csv.DictReader(f))
    return list(validation_rows)


def count_values(rows, key):
    counts = {}
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def rows_with_time_seconds(rows):
    timed_rows = []
    elapsed = 0.0

    for row in rows:
        next_row = dict(row)
        row_time = numeric_value(next_row.get("time_seconds"))
        if row_time is None:
            frame_index = numeric_value(next_row.get("frame_index"))
            source_fps = numeric_value(next_row.get("source_fps"))
            source_kind = str(next_row.get("source_kind", ""))
            if source_kind == "video" and frame_index is not None and source_fps and source_fps > 0:
                row_time = frame_index / source_fps
            else:
                row_time = elapsed
        next_row["time_seconds"] = round(float(row_time), 6)
        timed_rows.append(next_row)

        frame_dt = numeric_value(next_row.get("frame_dt")) or 0.0
        elapsed = max(elapsed, float(row_time) + frame_dt)

    return timed_rows


def summarize_results(reference_value=None):
    all_rows = rows_with_time_seconds(load_validation_rows())
    rows = all_rows
    frame_indices = [numeric_value(row.get("frame_index")) for row in rows]
    frame_indices = [int(value) for value in frame_indices if value is not None]
    frame_dt_values = [numeric_value(row.get("frame_dt")) for row in rows]
    frame_dt_values = [value for value in frame_dt_values if value is not None]
    time_values = [numeric_value(row.get("time_seconds")) for row in rows]
    time_values = [value for value in time_values if value is not None]

    return {
        "row_count": len(rows),
        "total_row_count": len(all_rows),
        "csv_path": str(VALIDATION_CSV),
        "frame_start": min(frame_indices) if frame_indices else None,
        "frame_end": max(frame_indices) if frame_indices else None,
        "duration_seconds": round(float(np.sum(frame_dt_values)), 3) if frame_dt_values else 0.0,
        "time_start_seconds": round(min(time_values), 3) if time_values else None,
        "time_end_seconds": round(max(time_values), 3) if time_values else None,
        "float_method_value": None if reference_value is None else round(float(reference_value), 6),
        "source_counts": count_values(rows, "source_kind"),
        "velocity_source_counts": count_values(rows, "velocity_source"),
        "metrics": {
            "surface": summarize_numeric_series(rows, "surface_velocity_mps", reference_value),
            "tracer": summarize_numeric_series(rows, "tracer_velocity_mps", reference_value),
            "piv": summarize_numeric_series(rows, "piv_velocity_mps", reference_value),
            "stiv": summarize_numeric_series(rows, "stiv_velocity_mps", reference_value),
            "processing_fps": summarize_numeric_series(rows, "processing_fps"),
            "tracer_count": summarize_numeric_series(rows, "tracer_count"),
            "tracked_points": summarize_numeric_series(rows, "tracked_points"),
            "piv_vectors": summarize_numeric_series(rows, "piv_vectors"),
            "meters_per_pixel": summarize_numeric_series(rows, "meters_per_pixel"),
        },
    }

# ============================================================
# MAIN PIPELINE
# ============================================================

def generate_frames(stream_token):
    global requested_seek_frame
    with lock:
        src = current_video_src
        kind = current_source_kind

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0 if kind == "video" else 15.0

    metadata = get_capture_metadata(cap, kind)
    metadata["fps"] = source_fps
    update_playback_info(**metadata)

    state = PipelineState(stiv_rows=deque(maxlen=int(global_vars["stiv_history"])))
    frame_counter = 0
    processing_fps = 0.0
    last_yield_time = time.perf_counter()
    elapsed_seconds = 0.0

    try:
        while True:
            frame_started = time.perf_counter()
            profile_times = {"_frame_started_at": frame_started}

            with lock:
                if stream_token != active_stream_token:
                    break
                is_paused = global_vars["is_paused"]
                target_stream_fps = float(global_vars["target_stream_fps"])
                seek_frame = requested_seek_frame
                requested_seek_frame = None

            if seek_frame is not None and kind == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(seek_frame)))
                state.reset_tracking()

            if is_paused:
                if state.last_payload is not None:
                    yield state.last_payload
                time.sleep(0.1)
                continue

            t = time.perf_counter()
            success, frame = cap.read()
            capture_time = time.perf_counter()
            record_stage(profile_times, "read_ms", t)

            if not success or frame is None:
                if kind == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state.reset_tracking()
                    continue
                time.sleep(0.05)
                continue

            current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            frame_index = max(0, current_frame_pos - 1) if kind == "video" else frame_counter

            frame_delta = 1
            if kind == "video" and state.prev_frame_index is not None:
                frame_delta = max(1, frame_index - state.prev_frame_index)
                frame_dt = frame_delta / source_fps
            elif kind == "camera" and state.prev_capture_time is not None:
                frame_dt = max(capture_time - state.prev_capture_time, 1e-6)
            else:
                frame_dt = 1.0 / source_fps

            sample_time_seconds = frame_index / source_fps if kind == "video" and source_fps > 0 else elapsed_seconds

            update_playback_info(current_frame=current_frame_pos)

            t = time.perf_counter()
            frame = resize_for_stream(frame)
            record_stage(profile_times, "resize_ms", t)

            t = time.perf_counter()
            with lock:
                is_ortho = global_vars["is_ortho"]
                H = global_vars["H"]
                target_sz = int(global_vars["target_sz"])
                ortho_ready = is_metric_ortho_ready()
                use_ortho = is_ortho and ortho_ready
                if use_ortho:
                    frame = cv2.warpPerspective(frame, H, (target_sz, target_sz))
            record_stage(profile_times, "orthographic_ms", t)

            state.reset_for_shape(frame.shape[:2])

            t = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display = frame.copy()
            record_stage(profile_times, "grayscale_ms", t)

            # Morphology segmentation, throttled.
            morphology_due = (not state.morphology_polygons and not state.morphology_boxes) or (
                frame_counter % max(1, int(global_vars["morphology_interval"])) == 0
            )
            if morphology_due:
                t = time.perf_counter()
                with lock:
                    morphology_model_path = selected_morphology_model_path
                result = predict_yolo_model(
                    get_morphology_model(),
                    frame,
                    float(global_vars["morphology_threshold"]),
                    int(global_vars["morphology_imgsz"]),
                    morphology_model_path,
                )
                record_stage(profile_times, "morphology_yolo_ms", t)
                state.morphology_polygons, state.morphology_boxes = extract_morphology_overlay(result, gray.shape)

            draw_morphology_overlay(display, state.morphology_polygons, state.morphology_boxes)
            velocity_mask = build_velocity_mask(gray.shape, state.morphology_polygons, state.morphology_boxes)

            # Tracer optical flow is restricted to the morphology mask when available.
            tracer_velocity = None
            tracked_points = 0
            if state.prev_gray is not None and state.track_pts is not None and len(state.track_pts) > 0:
                t = time.perf_counter()
                next_pts, of_status, _ = cv2.calcOpticalFlowPyrLK(
                    state.prev_gray,
                    gray,
                    state.track_pts,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
                record_stage(profile_times, "tracer_optical_flow_ms", t)
                if next_pts is not None and of_status is not None:
                    valid = of_status.ravel() == 1
                    good_new = next_pts[valid]
                    good_old = state.track_pts[valid]
                    valid_indexes = np.flatnonzero(valid)

                    velocities = []
                    kept_points = []
                    next_labels = []
                    next_trails = []

                    for src_idx, new, old in zip(valid_indexes, good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        if not point_in_mask(velocity_mask, a, b) or not point_in_mask(velocity_mask, c, d):
                            continue
                        dx, dy = a - c, b - d
                        disp_px = mt.hypot(float(dx), float(dy))
                        v = disp_px * global_vars["meters_per_pixel"] / max(frame_dt, 1e-6)
                        if not is_valid_velocity(v):
                            continue
                        kind_label = state.track_labels[src_idx] if src_idx < len(state.track_labels) else "tracer"
                        velocities.append(v)
                        kept_points.append([a, b])
                        next_labels.append(kind_label)
                        new_pt = (int(a), int(b))
                        if src_idx < len(state.tracer_trails):
                            trail = state.tracer_trails[src_idx] + [new_pt]
                        else:
                            trail = [(int(c), int(d)), new_pt]
                        next_trails.append(trail[-int(global_vars["tracer_trail_length"]):])
                        cv2.arrowedLine(display, (int(c), int(d)), (int(a), int(b)), (0, 255, 255), 2, tipLength=0.35)

                    tracer_velocity, kept_n, _ = robust_median(velocities)
                    tracked_points = kept_n
                    state.latest_tracer_velocity = tracer_velocity
                    state.track_pts = np.array(kept_points, dtype=np.float32).reshape(-1, 1, 2) if kept_points else None
                    state.track_labels = next_labels
                    state.tracer_trails = next_trails

            draw_tracer_trails(display, state.tracer_trails)

            # PIV is restricted to the morphology mask when available.
            piv_velocity = None
            piv_vector_count = 0
            piv_std = None
            if (
                global_vars["enable_piv"]
                and state.prev_gray is not None
                and frame_counter % max(1, int(global_vars["piv_interval"])) == 0
            ):
                t = time.perf_counter()
                piv_velocity, piv_flow, piv_vector_count, piv_std = estimate_piv_velocity(state.prev_gray, gray, velocity_mask, frame_dt)
                record_stage(profile_times, "piv_ms", t)
                if piv_velocity is not None:
                    state.latest_piv_velocity = piv_velocity
                    draw_piv_vectors(display, piv_flow, velocity_mask)

            # STIV is restricted to the morphology mask when available.
            stiv_velocity = None
            stiv_start = stiv_end = None
            if global_vars["enable_stiv"]:
                t = time.perf_counter()
                stiv_velocity, stiv_start, stiv_end = update_stiv_velocity(state.stiv_rows, gray, velocity_mask, frame_dt)
                record_stage(profile_times, "stiv_ms", t)
                if stiv_velocity is not None:
                    state.latest_stiv_velocity = stiv_velocity
                if stiv_start is not None and stiv_end is not None:
                    cv2.line(display, stiv_start, stiv_end, (255, 0, 255), 1)

            # Tracer detection after velocity update, throttled and restricted to the morphology mask.
            if frame_counter - state.last_tracer_detection_frame >= int(global_vars["detect_interval"]):
                state.last_tracer_detection_frame = frame_counter
                t = time.perf_counter()
                with lock:
                    tracer_model_path = selected_tracer_model_path
                result = predict_yolo_model(
                    get_tracer_model(),
                    frame,
                    float(global_vars["tracer_threshold"]),
                    int(global_vars["tracer_imgsz"]),
                    tracer_model_path,
                )
                record_stage(profile_times, "tracer_yolo_ms", t)
                detections = extract_tracer_points(result)
                detections = filter_tracer_detections(detections)
                detections = [
                    detection
                    for detection in detections
                    if point_in_mask(velocity_mask, detection["point"][0], detection["point"][1])
                ]
                state.latest_tracer_count = len(detections)
                draw_tracer_points(display, detections)
                if detections:
                    points = [d["point"] for d in detections]
                    state.track_pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                    state.track_labels = [d["kind"] for d in detections]
                    state.tracer_trails = [[tuple(p)] for p in points]
                else:
                    state.track_pts = None
                    state.track_labels = []
                    state.tracer_trails = []

            # Source priority for final surface velocity.
            surface_velocity = None
            velocity_source = "unavailable"
            if tracer_velocity is not None and tracked_points >= int(global_vars["min_tracked_points"]):
                surface_velocity = tracer_velocity
                velocity_source = "median tracer"
            elif state.latest_piv_velocity is not None:
                surface_velocity = state.latest_piv_velocity
                velocity_source = "median PIV"
            elif state.latest_stiv_velocity is not None:
                surface_velocity = state.latest_stiv_velocity
                velocity_source = "STIV"

            smoothed_surface = smooth_velocity(state, surface_velocity)

            # Save previous frame state.
            state.prev_gray = gray.copy()
            state.prev_frame_index = frame_index
            state.prev_capture_time = capture_time

            # Keep file videos frame-accurate for velocity estimation. If processing is slower
            # than the source FPS, playback slows down instead of skipping source frames.
            video_skipped_frames = 0
            target_stream_fps = max(1.0, min(30.0, target_stream_fps))
            target_frame_time = 1.0 / target_stream_fps
            processing_time = time.perf_counter() - frame_started
            sleep_time = target_frame_time - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            now = time.perf_counter()
            dt_yield = now - last_yield_time
            last_yield_time = now
            if dt_yield > 0:
                instant = 1.0 / dt_yield
                processing_fps = instant if processing_fps <= 0 else (0.85 * processing_fps + 0.15 * instant)

            draw_hud(
                display,
                smoothed_surface,
                velocity_source,
                tracer_velocity,
                state.latest_piv_velocity,
                state.latest_stiv_velocity,
                state.latest_tracer_count,
                tracked_points,
                piv_vector_count,
                source_fps,
                processing_fps,
                frame_dt,
                get_inference_device_label(),
            )

            # Research validation CSV row.
            append_validation_log({
                "time_seconds": round(float(sample_time_seconds), 6),
                "frame_index": int(frame_index),
                "source_kind": kind,
                "source_fps": round(source_fps, 6),
                "processing_fps": round(processing_fps, 6),
                "frame_dt": round(frame_dt, 6),
                "source_frame_delta": int(frame_delta),
                "video_skipped_frames": int(video_skipped_frames),
                "meters_per_pixel": round(float(global_vars["meters_per_pixel"]), 8),
                "surface_velocity_mps": "" if smoothed_surface is None else round(float(smoothed_surface), 6),
                "velocity_source": velocity_source,
                "tracer_velocity_mps": "" if tracer_velocity is None else round(float(tracer_velocity), 6),
                "tracer_count": int(state.latest_tracer_count),
                "tracked_points": int(tracked_points),
                "piv_velocity_mps": "" if state.latest_piv_velocity is None else round(float(state.latest_piv_velocity), 6),
                "piv_vectors": int(piv_vector_count),
                "piv_std": "" if piv_std is None else round(float(piv_std), 6),
                "stiv_velocity_mps": "" if state.latest_stiv_velocity is None else round(float(state.latest_stiv_velocity), 6),
            })
            elapsed_seconds = max(elapsed_seconds, float(sample_time_seconds) + frame_dt)

            ret, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            state.last_payload = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            record_profile(profile_times, frame_counter, display.shape[:2])
            yield state.last_payload

            frame_counter += 1
    finally:
        cap.release()

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    init_cam_dims()
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    with lock:
        model_paths = [selected_morphology_model_path, selected_tracer_model_path]
    if any(is_onnx_model(path) for path in model_paths) and not can_use_onnx_cuda():
        abort(500, description="ONNX CUDAExecutionProvider is required but unavailable.")
    token = cull_streams(reset_models=False)
    init_cam_dims()
    return Response(generate_frames(token), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
    cull_streams(reset_models=False)
    with lock:
        global_vars["is_paused"] = False
    return jsonify({"ok": True})


@app.route("/playback", methods=["POST"])
def playback():
    data = request.get_json() or {}
    with lock:
        global_vars["is_paused"] = bool(data.get("is_paused", False))
        playback_info["is_paused"] = global_vars["is_paused"]
    return jsonify({"ok": True, "is_paused": global_vars["is_paused"]})


@app.route("/playback_status")
def playback_status():
    with lock:
        info = dict(playback_info)
        info["is_paused"] = global_vars["is_paused"]
    fps = float(info.get("fps") or 0.0)
    current_frame = int(info.get("current_frame") or 0)
    info["current_time"] = current_frame / fps if fps > 0 else 0.0
    return jsonify({"ok": True, **info})


@app.route("/profile_stats")
def profile_stats():
    return jsonify({"ok": True, **summarize_profile()})


@app.route("/validation_log")
def validation_log():
    return jsonify({"ok": True, "csv_path": str(VALIDATION_CSV), "rows_cached": list(validation_rows)[-200:]})


@app.route("/model_options")
def model_options():
    with lock:
        morphology_path = selected_morphology_model_path
        tracer_path = selected_tracer_model_path
    return jsonify({
        "ok": True,
        "models": list_available_models(),
        "morphology_model_path": morphology_path,
        "tracer_model_path": tracer_path,
        "supported_extensions": sorted(ALLOWED_MODEL_EXTENSIONS),
    })


@app.route("/results_summary", methods=["GET", "POST"])
def results_summary():
    data = request.get_json(silent=True) or {}
    reference_value = numeric_value(data.get("float_method_value", request.args.get("float_method_value")))

    if request.method == "POST":
        cull_streams(reset_models=False)
        with lock:
            global_vars["is_paused"] = False
            playback_info["is_paused"] = False
    return jsonify({"ok": True, **summarize_results(reference_value)})


@app.route("/validation_reset", methods=["POST"])
def validation_reset():
    reset_validation_storage()
    return jsonify({"ok": True})


@app.route("/seek", methods=["POST"])
def seek():
    global requested_seek_frame
    data = request.get_json() or {}
    with lock:
        if current_source_kind != "video":
            return jsonify({"ok": False, "error": "Seeking is only available for video."}), 400
        total_frames = int(playback_info.get("total_frames") or 0)
        fps = float(playback_info.get("fps") or 0.0)
        if total_frames <= 0:
            return jsonify({"ok": False, "error": "Video does not expose a seekable timeline."}), 400
        target = get_int(data, "frame", 0) if "frame" in data else int(get_float(data, "time", 0.0) * fps)
        requested_seek_frame = max(0, min(target, total_frames - 1))
        playback_info["current_frame"] = requested_seek_frame
    return jsonify({"ok": True, "current_frame": requested_seek_frame})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    global current_video_src, current_source_kind
    uploaded = request.files.get("video")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"ok": False, "error": "No video file uploaded."}), 400
    if not is_allowed_video(uploaded.filename):
        return jsonify({"ok": False, "error": "Upload avi, mp4, mov, mkv, or webm."}), 400

    filename = secure_filename(uploaded.filename)
    path = UPLOAD_FOLDER / f"{uuid4().hex}_{filename}"
    uploaded.save(path)

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            path.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": "OpenCV could not open uploaded video."}), 400
        success, frame = cap.read()
        if not success or frame is None:
            path.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": "OpenCV could not read uploaded video."}), 400
        frame = resize_for_stream(frame)
        h, w = frame.shape[:2]
        metadata = get_capture_metadata(cap, "video")
    finally:
        cap.release()

    with lock:
        current_video_src = str(path)
        current_source_kind = "video"
        global_vars["video_w"] = w
        global_vars["video_h"] = h
        global_vars["H"] = None
        global_vars["ortho_status"] = "VGCP calibration optional; using raw scale"
        global_vars["is_paused"] = False
        global_vgcps.clear()
        playback_info.update(metadata)
        payload = calibration_status_payload()
    cull_streams(reset_models=False)
    reset_validation_storage()
    return jsonify({"filename": filename, "video_w": w, "video_h": h, **metadata, **payload})


@app.route("/camera_devices")
def camera_devices():
    devices = []
    for i in range(MAX_CAMERA_SCAN_INDEX + 1):
        cap = cv2.VideoCapture(i)
        try:
            if not cap.isOpened():
                continue
            success, frame = cap.read()
            if not success or frame is None:
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            devices.append({
                "index": i,
                "label": f"Camera {i} ({frame.shape[1]}x{frame.shape[0]})",
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "fps": fps if fps and fps > 0 else None,
            })
        finally:
            cap.release()
    return jsonify({"ok": True, "devices": devices})


@app.route("/set_camera_source", methods=["POST"])
def set_camera_source():
    global current_video_src, current_source_kind
    data = request.get_json() or {}
    idx = get_int(data, "camera_index", 0, min_value=0)
    cap = cv2.VideoCapture(idx)
    try:
        if not cap.isOpened():
            return jsonify({"ok": False, "error": f"Could not open camera {idx}."}), 400
        success, frame = cap.read()
        if not success or frame is None:
            return jsonify({"ok": False, "error": f"Could not read camera {idx}."}), 400
        frame = resize_for_stream(frame)
        h, w = frame.shape[:2]
        metadata = get_capture_metadata(cap, "camera")
    finally:
        cap.release()
    with lock:
        current_video_src = idx
        current_source_kind = "camera"
        global_vars["video_w"] = w
        global_vars["video_h"] = h
        global_vars["H"] = None
        global_vars["is_ortho"] = False
        global_vars["ortho_status"] = "Camera source selected; using raw scale"
        global_vgcps.clear()
        playback_info.update(metadata)
        payload = calibration_status_payload()
    cull_streams(reset_models=False)
    reset_validation_storage()
    return jsonify({"source": "camera", "camera_index": idx, "video_w": w, "video_h": h, **payload})


@app.route("/calibration_status")
def calibration_status():
    with lock:
        payload = calibration_status_payload()
    return jsonify(payload)


@app.route("/vgcp", methods=["POST"])
def vgcp():
    data = request.get_json() or {}
    points = data.get("points", [])
    try:
        parsed = [VGCP(p) for p in points]
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "VGCP points must contain numeric x, y, and r."}), 400
    parsed = [p for p in parsed if p.r > 0]
    if len(parsed) < 4:
        return jsonify({"ok": False, "error": "Optional calibration needs at least 4 VGCPs with positive R.", "ortho_ready": False}), 400
    with lock:
        global_vgcps.clear()
        global_vgcps.extend(parsed)
        update_homo()
        if global_vars["H"] is not None:
            global_vars["is_ortho"] = True
        payload = calibration_status_payload()
    cull_streams(reset_models=False)
    return jsonify(payload)


@app.route("/reset_orthorectification", methods=["POST"])
def reset_orthorectification():
    with lock:
        global_vars["H"] = None
        global_vars["ortho_status"] = "VGCP calibration optional; using raw scale"
        global_vars["is_ortho"] = False
        global_vgcps.clear()
        payload = calibration_status_payload()
    cull_streams(reset_models=False)
    return jsonify(payload)


@app.route("/cam_settings", methods=["POST"])
def cam_settings():
    data = request.get_json() or {}
    with lock:
        if "meters_per_pixel" in data and global_vars["H"] is None:
            global_vars["meters_per_pixel"] = get_float(data, "meters_per_pixel", global_vars["meters_per_pixel"], 1e-8, None)
        global_vars["cam_fov"] = get_float(data, "cam_fov", global_vars["cam_fov"], 1.0, 179.0)
        global_vars["is_ortho"] = get_bool(data, "is_ortho", global_vars["is_ortho"])
        update_homo()
    cull_streams(reset_models=False)
    return jsonify({"ok": True, "ortho_status": global_vars["ortho_status"]})


@app.route("/yolo_params", methods=["POST"])
def yolo_params():
    global selected_morphology_model_path, selected_tracer_model_path
    data = request.get_json() or {}
    numeric_float = [
        ("morphology_threshold", 0.0, 1.0), ("tracer_threshold", 0.0, 1.0),
        ("target_stream_fps", 1.0, 30.0), ("meters_per_pixel", 1e-8, None),
        ("max_velocity", 0.01, None), ("min_velocity", 0.0, None),
        ("piv_min_corr", 0.0, 1.0), ("stiv_start_x", 0.0, 1.0),
        ("stiv_start_y", 0.0, 1.0), ("stiv_end_x", 0.0, 1.0), ("stiv_end_y", 0.0, 1.0),
    ]
    numeric_int = [
        ("target_sz", 32, None), ("stream_max_width", 160, None),
        ("morphology_imgsz", 32, None), ("tracer_imgsz", 32, None),
        ("morphology_interval", 1, None), ("detect_interval", 1, None),
        ("morphology_mask_erode", 0, None), ("min_tracked_points", 1, None),
        ("tracer_trail_length", 1, None), ("velocity_smoothing_window", 1, None),
        ("piv_interval", 1, None), ("piv_grid_step", 12, None),
        ("piv_window_size", 16, None), ("piv_search_size", 24, None),
        ("piv_min_vectors", 1, None), ("piv_max_displacement_px", 1, None),
        ("stiv_history", 2, None),
    ]
    bools = ["enable_piv", "enable_stiv"]
    reset_models = False
    with lock:
        for key, lo, hi in numeric_float:
            if key in data:
                global_vars[key] = get_float(data, key, global_vars[key], lo, hi)
        for key, lo, hi in numeric_int:
            if key in data:
                global_vars[key] = get_int(data, key, global_vars[key], lo, hi)
        for key in bools:
            if key in data:
                global_vars[key] = get_bool(data, key, global_vars[key])

        if "tracer_target_kind" in data:
            target_kind = str(data.get("tracer_target_kind", "all") or "all").strip()
            global_vars["tracer_target_kind"] = target_kind if target_kind else "all"

        if "morphology_model_path" in data:
            next_path = validate_model_path(data.get("morphology_model_path"), selected_morphology_model_path)
            if next_path != selected_morphology_model_path:
                selected_morphology_model_path = next_path
                global_vars["morphology_model_path"] = next_path
                reset_models = True

        if "tracer_model_path" in data:
            next_path = validate_model_path(data.get("tracer_model_path"), selected_tracer_model_path)
            if next_path != selected_tracer_model_path:
                selected_tracer_model_path = next_path
                global_vars["tracer_model_path"] = next_path
                reset_models = True

        if global_vars["H"] is not None:
            global_vars["meters_per_pixel"] = 1.0 / max(float(global_vars["px_scale"]), 1e-6)
    cull_streams(reset_models=reset_models)
    return jsonify({"ok": True, "settings": jsonify_safe_settings()})


if __name__ == "__main__":
    print("======================================")
    print("RESEARCH RIVER SURFACE VELOCITY SERVER")
    print("Open http://localhost:5000")
    print("Validation CSV:", VALIDATION_CSV)
    print("======================================")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
