"""
Core video pipeline: three inputs -> one talking, motion-driven output.

Inputs:
    base_image_path   - the full scene (e.g. two kids in a market). Everything in
                         this image except the chosen face stays untouched.
    face_swap_photo   - a photo of the face that should replace the chosen
                         character's face in base_image.
    driving_video_path - a video providing head motion + audio for the animation.
    target_box         - (x, y, w, h) of the face in base_image to replace + animate.

Steps:
    1. Face-swap `face_swap_photo`'s identity onto the target face region of
       base_image (insightface inswapper), producing a composited still image.
    2. Crop that region with padding for context.
    3. Animate the crop's head motion with LivePortrait, driven by driving_video.
    4. Tighten lip-sync on the animated crop with Wav2Lip, driven by the audio
       extracted from driving_video.
    5. Composite the processed crop back onto the ORIGINAL untouched base_image,
       frame by frame, with a feathered edge blend.
    6. Mux the driving video's audio into the final result.

NOTE: this file has been written to mirror the working notebook pipeline logic,
but has not been executed end-to-end in this environment (no GPU / no network
here). Expect to debug path/version issues on first real run -- see README.md.
"""

import os
import glob
import subprocess

import cv2
import numpy as np

LIVEPORTRAIT_DIR = os.environ.get("LIVEPORTRAIT_DIR", "/opt/LivePortrait")
WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "/opt/Wav2Lip")
WAV2LIP_CKPT = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
INSWAPPER_MODEL = os.environ.get("INSWAPPER_MODEL", "/opt/models/inswapper_128.onnx")

PADDING_SCALE = 2.6
FEATHER_PX = 20


def log(job, msg):
    print(f"[job {job}] {msg}", flush=True)


def extract_audio(driving_video_path, out_wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", driving_video_path, "-ac", "1", "-ar", "22050", out_wav_path],
        check=True,
    )
    return out_wav_path


def padded_crop_bounds(img_w, img_h, box, scale=PADDING_SCALE):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * scale
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(img_w, cx + side / 2))
    y1 = int(min(img_h, cy + side / 2))
    return x0, y0, x1, y1


def face_swap_onto_image(base_image_path, face_swap_photo_path, target_box, out_path):
    """
    Swaps the identity from face_swap_photo onto the face at target_box inside
    base_image, using insightface's FaceAnalysis + inswapper model. Only the
    targeted face is touched; everything else in base_image is byte-identical.
    """
    import insightface
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 -> CPU
    swapper = insightface.model_zoo.get_model(INSWAPPER_MODEL, download=False)

    base_img = cv2.imread(base_image_path)
    source_img = cv2.imread(face_swap_photo_path)

    base_faces = face_app.get(base_img)
    source_faces = face_app.get(source_img)
    if not base_faces:
        raise RuntimeError("No face detected in base_image for swapping.")
    if not source_faces:
        raise RuntimeError("No face detected in face_swap_photo.")

    # pick the base-image face closest to the requested target_box center
    tx, ty, tw, th = target_box
    target_center = np.array([tx + tw / 2, ty + th / 2])

    def face_center(f):
        x0, y0, x1, y1 = f.bbox
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    chosen = min(base_faces, key=lambda f: np.linalg.norm(face_center(f) - target_center))
    source_face = source_faces[0]

    result = swapper.get(base_img, chosen, source_face, paste_back=True)
    cv2.imwrite(out_path, result)
    return out_path


def animate_crop_with_liveportrait(crop_path, driving_video_path, out_dir):
    cmd = [
        "python", "inference.py",
        "-s", crop_path,
        "-d", driving_video_path,
        "-o", out_dir,
        "--flag_do_crop",
        "--scale", "2.3",
        "--vy_ratio", "-0.05",
    ]
    subprocess.run(cmd, cwd=LIVEPORTRAIT_DIR, check=True)
    result_files = glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True)
    result_files = [f for f in result_files if "concat" not in f.lower()]
    if not result_files:
        raise RuntimeError("LivePortrait produced no output video.")
    return result_files[0]


def lipsync_with_wav2lip(face_video_path, audio_path, out_path):
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", WAV2LIP_CKPT,
        "--face", face_video_path,
        "--audio", audio_path,
        "--outfile", out_path,
        "--pads", "0", "15", "0", "0",
        "--nosmooth",
    ]
    subprocess.run(cmd, cwd=WAV2LIP_DIR, check=True)
    return out_path


def feather_mask(w, h, feather_px=FEATHER_PX):
    mask = np.ones((h, w), dtype=np.float32)
    fp = min(feather_px, w // 2, h // 2)
    if fp > 0:
        mask[:fp, :] *= np.linspace(0, 1, fp)[:, None]
        mask[-fp:, :] *= np.linspace(1, 0, fp)[:, None]
        mask[:, :fp] *= np.linspace(0, 1, fp)[None, :]
        mask[:, -fp:] *= np.linspace(1, 0, fp)[None, :]
    return mask


def composite_crop_over_original(original_image_path, animated_crop_video_path, crop_bounds, out_video_path):
    original = cv2.imread(original_image_path)
    img_h, img_w = original.shape[:2]
    x0, y0, x1, y1 = crop_bounds
    mask = feather_mask(x1 - x0, y1 - y0)

    cap = cv2.VideoCapture(animated_crop_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_w, img_h))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (x1 - x0, y1 - y0)).astype(np.float32)
        canvas = original.copy().astype(np.float32)
        region = canvas[y0:y1, x0:x1]
        m = mask[:, :, None]
        canvas[y0:y1, x0:x1] = frame * m + region * (1 - m)
        writer.write(canvas.astype(np.uint8))
    cap.release()
    writer.release()
    return out_video_path


def mux_audio(video_no_audio_path, audio_path, final_out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_no_audio_path, "-i", audio_path,
         "-c:v", "libx264", "-c:a", "aac", "-shortest", final_out_path],
        check=True,
    )
    return final_out_path


def run_pipeline(job_id, work_dir, base_image_path, face_swap_photo_path, driving_video_path, target_box):
    """
    Orchestrates the full pipeline. Returns the path to the final mp4.
    Raises on any failure -- caller (app.py) is responsible for catching and
    recording the error against the job status.
    """
    os.makedirs(work_dir, exist_ok=True)

    log(job_id, "extracting audio from driving video")
    audio_path = extract_audio(driving_video_path, os.path.join(work_dir, "driving_audio.wav"))

    log(job_id, "face-swapping onto base image")
    composited_image_path = os.path.join(work_dir, "composited.png")
    face_swap_onto_image(base_image_path, face_swap_photo_path, target_box, composited_image_path)

    base_img = cv2.imread(base_image_path)
    img_h, img_w = base_img.shape[:2]
    crop_bounds = padded_crop_bounds(img_w, img_h, target_box)
    x0, y0, x1, y1 = crop_bounds
    composited_img = cv2.imread(composited_image_path)
    crop_path = os.path.join(work_dir, "face_crop.png")
    cv2.imwrite(crop_path, composited_img[y0:y1, x0:x1])

    log(job_id, "running LivePortrait (head motion)")
    lp_out_dir = os.path.join(work_dir, "liveportrait_out")
    lp_video = animate_crop_with_liveportrait(crop_path, driving_video_path, lp_out_dir)

    log(job_id, "running Wav2Lip (lip-sync)")
    lipsynced_path = os.path.join(work_dir, "lipsynced.mp4")
    lipsync_with_wav2lip(lp_video, audio_path, lipsynced_path)

    log(job_id, "compositing back onto original image")
    composited_video_path = os.path.join(work_dir, "composited_noaudio.mp4")
    composite_crop_over_original(base_image_path, lipsynced_path, crop_bounds, composited_video_path)

    log(job_id, "muxing audio")
    final_path = os.path.join(work_dir, "final_output.mp4")
    mux_audio(composited_video_path, audio_path, final_path)

    log(job_id, "done")
    return final_path
