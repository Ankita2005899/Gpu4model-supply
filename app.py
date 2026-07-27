"""
Flask front-end for the face-swap + reenact + lip-sync pipeline.

Flow:
    GET  /                       -> upload form for base_image only
    POST /detect                 -> runs face detection on base_image, shows
                                     a numbered preview + the rest of the form
                                     (driving_video, face_swap_photo, face index)
    POST /submit/<job_id>        -> saves remaining uploads, starts background
                                     thread, redirects to status page
    GET  /status/<job_id>        -> polls job status (queued/running/done/error)
    GET  /download/<job_id>      -> serves the final mp4 once ready

Jobs run in a background thread so the HTTP request that kicks them off
returns immediately -- this matters on Render, where a synchronous request
that runs for minutes/hours would just time out. See README.md for the
important caveats about running this on Render's CPU-only, free-tier-unsafe
plans.
"""

import os
import uuid
import threading
import traceback

import cv2
from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify

import pipeline

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(APP_ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB upload cap

# in-memory job registry -- fine for a single-instance, low-traffic deployment.
# NOTE: this resets if the process restarts. For anything beyond personal use,
# swap this for a real job table (sqlite/Postgres).
JOBS = {}


def job_dir(job_id):
    d = os.path.join(JOBS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "base_image" not in request.files:
        return "Missing base_image", 400

    job_id = uuid.uuid4().hex[:12]
    jdir = job_dir(job_id)
    base_image_path = os.path.join(jdir, "base_image.png")
    request.files["base_image"].save(base_image_path)

    img_bgr = cv2.imread(base_image_path)
    if img_bgr is None:
        return "Could not read uploaded image", 400
    img_h, img_w = img_bgr.shape[:2]

    import mediapipe as mp
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_face = mp.solutions.face_detection
    boxes = []
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as fd:
        results = fd.process(img_rgb)
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * img_w)
                y = int(bbox.ymin * img_h)
                w = int(bbox.width * img_w)
                h = int(bbox.height * img_h)
                boxes.append([x, y, w, h])
    boxes = sorted(boxes, key=lambda b: b[0])

    preview = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(preview, str(i), (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    preview_path = os.path.join(jdir, "preview.png")
    cv2.imwrite(preview_path, preview)

    JOBS[job_id] = {"status": "awaiting_input", "boxes": boxes, "img_w": img_w, "img_h": img_h}

    return render_template(
        "preview.html",
        job_id=job_id,
        boxes=boxes,
        num_faces=len(boxes),
    )


@app.route("/preview_image/<job_id>")
def preview_image(job_id):
    return send_file(os.path.join(job_dir(job_id), "preview.png"))


@app.route("/submit/<job_id>", methods=["POST"])
def submit(job_id):
    if job_id not in JOBS:
        return "Unknown job", 404

    jdir = job_dir(job_id)

    if "driving_video" not in request.files or "face_swap_photo" not in request.files:
        return "Missing driving_video or face_swap_photo", 400

    driving_video_path = os.path.join(jdir, "driving_video.mp4")
    face_swap_photo_path = os.path.join(jdir, "face_swap_photo.png")
    request.files["driving_video"].save(driving_video_path)
    request.files["face_swap_photo"].save(face_swap_photo_path)

    face_index = int(request.form.get("face_index", 0))
    boxes = JOBS[job_id]["boxes"]
    manual_box = request.form.get("manual_box", "").strip()
    if manual_box:
        # expects "x,y,w,h"
        target_box = tuple(int(v.strip()) for v in manual_box.split(","))
    elif boxes:
        target_box = tuple(boxes[face_index])
    else:
        return "No face box available -- provide a manual_box override.", 400

    base_image_path = os.path.join(jdir, "base_image.png")

    JOBS[job_id]["status"] = "queued"
    JOBS[job_id]["error"] = None

    def _run():
        JOBS[job_id]["status"] = "running"
        try:
            final_path = pipeline.run_pipeline(
                job_id=job_id,
                work_dir=jdir,
                base_image_path=base_image_path,
                face_swap_photo_path=face_swap_photo_path,
                driving_video_path=driving_video_path,
                target_box=target_box,
            )
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result_path"] = final_path
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = f"{e}\n{traceback.format_exc()}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return redirect(url_for("status", job_id=job_id))


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in JOBS:
        return "Unknown job", 404
    return render_template("status.html", job_id=job_id, job=JOBS[job_id])


@app.route("/status_json/<job_id>")
def status_json(job_id):
    if job_id not in JOBS:
        return jsonify({"status": "unknown"}), 404
    j = JOBS[job_id]
    return jsonify({"status": j.get("status"), "error": j.get("error")})


@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in JOBS or JOBS[job_id].get("status") != "done":
        return "Not ready", 404
    return send_file(JOBS[job_id]["result_path"], as_attachment=True, download_name="final_output.mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
