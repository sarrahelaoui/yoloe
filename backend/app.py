"""
app.py - Flask backend for the YOLOE Text-Prompt Detection console.

This mirrors the working Kaggle approach: you type target class names
(e.g. "forklift", "person", "truck") instead of drawing boxes on
reference images. Endpoints:

  POST /api/upload_dataset   -> upload dataset images into backend/dataset/
  GET  /api/dataset_list     -> how many images currently sit in dataset/
  POST /api/set_classes      -> set the target class names on the model
  POST /api/run_detection    -> set classes (if given) + run over dataset_dir
  GET  /api/results          -> last detection run log (json)
  GET  /api/outputs/<file>   -> serve an annotated result image
  GET  /api/health           -> quick status check
"""

import os
import json
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_FILE = os.path.join(BASE_DIR, "last_results.json")

for d in (OUTPUTS_DIR, DATASET_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
CORS(app)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/upload_dataset", methods=["POST"])
def upload_dataset():
    """
    Uploads any number of images straight into backend/dataset/.
    Form field: 'images' (multiple files).
    Optional form field 'clear'='true' wipes the folder first.
    """
    if request.form.get("clear") == "true":
        for f in os.listdir(DATASET_DIR):
            fp = os.path.join(DATASET_DIR, f)
            if os.path.isfile(fp):
                os.remove(fp)

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images received"}), 400

    saved = []
    for f in files:
        if not f.filename:
            continue
        safe_name = os.path.basename(f.filename)
        f.save(os.path.join(DATASET_DIR, safe_name))
        saved.append(safe_name)

    return jsonify({"status": "ok", "count": len(saved), "saved": saved})


@app.route("/api/dataset_list")
def dataset_list():
    imgs = [f for f in os.listdir(DATASET_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    return jsonify({"count": len(imgs), "files": imgs})


@app.route("/api/set_classes", methods=["POST"])
def set_classes_route():
    """Body: {"classes": ["forklift", "person", "truck"]}"""
    try:
        payload = request.get_json(force=True)
        names = [n.strip() for n in payload.get("classes", []) if n.strip()]
        detector.set_classes(names)
        return jsonify({"status": "ok", "classes": names})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/outputs/<path:filename>")
def get_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/run_detection", methods=["POST"])
def run_detection():
    """
    Body:
    {
      "classes": ["forklift", "person", "truck"],   // optional if already set
      "dataset_dir": "dataset",                      // relative to backend/, or absolute
      "conf": 0.25,
      "device": null | "cpu" | "cuda",
      "limit": 4,          // optional: only process the first N images (test phase)
      "skip": 0,            // optional: skip the first N images (rest-of-dataset phase)
      "imgsz": 640,         // optional: inference resolution, lower = faster
      "batch_size": 8       // optional: images per model.predict() call
    }
    """
    try:
        payload = request.get_json(force=True)

        names = [n.strip() for n in payload.get("classes", []) if n.strip()]
        if names:
            detector.set_classes(names)
        elif detector.get_current_classes() is None:
            return jsonify({"error": "No classes set. Send 'classes' at least once."}), 400

        dataset_dir = payload.get("dataset_dir") or "dataset"
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.join(BASE_DIR, dataset_dir)
        if not os.path.isdir(dataset_dir):
            return jsonify({"error": f"Dataset folder not found: {dataset_dir}"}), 400

        conf = float(payload.get("conf", 0.25))
        device = payload.get("device")
        limit = payload.get("limit")
        skip = int(payload.get("skip", 0))
        imgsz = int(payload.get("imgsz", 640))
        batch_size = int(payload.get("batch_size", 8))
        only_files = payload.get("files") or None

        result = detector.run_on_dataset(
            dataset_dir=dataset_dir,
            output_dir=OUTPUTS_DIR,
            conf=conf,
            device=device,
            imgsz=imgsz,
            batch_size=batch_size,
            limit=limit,
            skip=skip,
            only_files=only_files,
        )

        with open(RESULTS_FILE, "w") as fh:
            json.dump(result, fh, indent=2)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/results")
def get_results():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"summary": None, "results": []})
    with open(RESULTS_FILE) as fh:
        return jsonify(json.load(fh))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
