"""
detector.py
-----------
YOLOE with TEXT-PROMPT classes (the approach that actually worked in the
Kaggle notebook): model.set_classes(names, model.get_text_pe(names))
then model.predict(images_batch). No visual/bbox prompting, no refer_image.

Optimisations vs the first version:
  - Batch inference (several images per model.predict call) instead of
    one-by-one -> much less per-image Python/model overhead.
  - Configurable imgsz (smaller = faster, e.g. 416/480 instead of 640).
  - Optional `limit` to run a quick test on the first N images before
    committing to the whole dataset (this is what "tester sur 4 images
    d'abord" needs).

Drawing:
  - We draw ONLY a thin coloured rectangle (+ small label tag) on the
    original image. We do NOT use results.plot() anymore, because it
    fills label backgrounds and (for -seg models) can draw masks over
    the image. The rest of the image stays 100% untouched.
"""

import os
import time
import cv2
import numpy as np
import psutil
import torch

from ultralytics import YOLOE

MODEL_PATH = os.environ.get("YOLOE_MODEL_PATH", "yoloe-v8s-seg.pt")
SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# consistent colour per class index (BGR, since we draw with cv2)
PALETTE = [
    (32, 176, 255),   # amber
    (200, 212, 0),    # cyan
    (95, 90, 255),    # red
    (122, 214, 55),   # green
    (255, 127, 168),  # purple
    (0, 214, 255),    # yellow
    (255, 160, 0),    # blue-ish orange
    (180, 105, 255),  # pink
]

_state = {"model": None, "classes": None, "model_path": None}


def load_model(model_path=None):
    """Loads YOLOE once and keeps it warm in memory between requests."""
    path = model_path or MODEL_PATH
    if _state["model"] is None or _state["model_path"] != path:
        _state["model"] = YOLOE(path)
        _state["model_path"] = path
        _state["classes"] = None  # new model -> classes must be re-set
    return _state["model"]


def set_classes(names, model_path=None):
    """
    names: list[str] e.g. ["forklift", "person", "truck"]
    Mirrors the working notebook exactly:
        text_pe = model.get_text_pe(names)
        model.set_classes(names, text_pe)
    """
    if not names:
        raise ValueError("No class names provided")
    model = load_model(model_path)
    text_pe = model.get_text_pe(names)
    model.set_classes(names, text_pe)
    _state["classes"] = names
    return names


def get_current_classes():
    return _state["classes"]


def _process_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return None


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def draw_boxes_only(image_bgr, xyxy, cls_idx, confs, names):
    """
    Draws a thin coloured rectangle + small label tag per detection.
    Everything else in the image is left byte-for-byte untouched.
    """
    img = image_bgr  # draw in place, caller already has its own copy
    for (x1, y1, x2, y2), ci, cf in zip(xyxy, cls_idx, confs):
        color = PALETTE[int(ci) % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)

        label = f"{names[int(ci)]} {cf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tag_top = max(0, p1[1] - th - 6)
        cv2.rectangle(img, (p1[0], tag_top), (p1[0] + tw + 6, p1[1]), color, -1)
        cv2.putText(img, label, (p1[0] + 3, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def run_on_dataset(dataset_dir, output_dir, conf=0.25, device=None,
                    imgsz=640, batch_size=8, limit=None, skip=0, only_files=None):
    """
    Runs prediction across images in dataset_dir using whatever classes
    were set via set_classes().

    only_files: explicit list of filenames to process, IN THAT ORDER
                (e.g. the 4 images the user hand-picked in the UI for the
                test phase, or "everything except those 4" for phase 2).
                When given, this takes priority over skip/limit.
    skip / limit: fallback slicing (first N / everything after N) used
                  only when only_files is not provided.
    imgsz: inference resolution. Lower (e.g. 416) = faster, less precise.
    batch_size: how many images go into a single model.predict() call.
    """
    if _state["classes"] is None:
        raise RuntimeError("No classes set yet — call set_classes() first.")

    model = _state["model"]
    if device:
        model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    all_images = sorted(
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(SUPPORTED_EXT)
    )

    if only_files:
        on_disk = set(all_images)
        images = [f for f in only_files if f in on_disk]
    else:
        images = all_images[int(skip):]
        if limit:
            images = images[:int(limit)]

    log = []
    t_start_total = time.time()

    for batch_names in _chunks(images, batch_size):
        batch_paths = [os.path.join(dataset_dir, f) for f in batch_names]

        mem_before = _process_mem_mb()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        gpu_before = _gpu_mem_mb()
        t0 = time.time()

        results_list = model.predict(
            batch_paths,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        t1 = time.time()
        mem_after = _process_mem_mb()
        gpu_after = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None

        batch_time = t1 - t0
        per_image_time = batch_time / len(batch_names)
        per_image_ram = (mem_after - mem_before) / len(batch_names)
        per_image_gpu = ((gpu_after - gpu_before) / len(batch_names)) if gpu_before is not None else None

        for fname, res in zip(batch_names, results_list):
            img_bgr = cv2.imread(os.path.join(dataset_dir, fname))
            if img_bgr is None:
                continue  # unreadable file, skip safely

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                cls_idx = res.boxes.cls.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
            else:
                xyxy, cls_idx, confs = [], [], []

            annotated = draw_boxes_only(img_bgr.copy(), xyxy, cls_idx, confs, res.names)
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, annotated)

            log.append({
                "image": fname,
                "output": out_path,
                "time_sec": round(per_image_time, 4),
                "ram_delta_mb": round(per_image_ram, 2),
                "gpu_delta_mb": round(per_image_gpu, 2) if per_image_gpu is not None else None,
                "detections": len(xyxy),
            })

    total_time = round(time.time() - t_start_total, 3)
    summary = {
        "n_images": len(images),
        "n_total_in_folder": len(all_images),
        "skip": int(skip),
        "classes": _state["classes"],
        "imgsz": imgsz,
        "batch_size": batch_size,
        "total_time_sec": total_time,
        "avg_time_sec": round(total_time / len(images), 4) if images else 0,
        "total_detections": sum(r["detections"] for r in log),
        "peak_ram_mb": round(_process_mem_mb(), 2),
        "peak_gpu_mb": round(_gpu_mem_mb(), 2) if _gpu_mem_mb() is not None else None,
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
    }

    return {"summary": summary, "results": log}


if __name__ == "__main__":
    m = load_model()
    set_classes(["forklift", "person", "truck"])
    print("YOLOE ready with classes:", get_current_classes())
