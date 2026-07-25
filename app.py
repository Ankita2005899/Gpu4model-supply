import pyttsx3
import os
import shutil
import subprocess
import uuid
from pathlib import Path
import edge_tts
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from gtts import gTTS

# Background/foreground/light helpers (rembg-based, not mediapipe - mediapipe's
# 'solutions' API proved unreliable across versions during testing)
import cv2
import numpy as np
from rembg import remove

app = FastAPI(title="Talking Face Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Voice cloning (XTTS) is OPTIONAL. It needs a separate Python 3.10 environment
# (torch versions for SadTalker and XTTS conflict). Set XTTS_PYTHON_PATH if you
# have that environment set up; otherwise voice_sample is silently ignored and
# gTTS is used instead.
XTTS_PYTHON_PATH = os.environ.get("XTTS_PYTHON_PATH", "/opt/xtts_env/bin/python")
XTTS_CLONE_SCRIPT = os.environ.get("XTTS_CLONE_SCRIPT", "/opt/xtts_clone.py")
XTTS_AVAILABLE = os.path.exists(XTTS_PYTHON_PATH) and os.path.exists(XTTS_CLONE_SCRIPT)


def replace_background(input_video_path, bg_image_path, output_video_path):
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.imread(str(bg_image_path))
    bg = cv2.resize(bg, (w, h))

    tmp_noaudio = str(output_video_path) + "_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgba = remove(frame)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        person = rgba[:, :, :3].astype(np.float32)
        composited = person * alpha + bg.astype(np.float32) * (1 - alpha)
        writer.write(np.clip(composited, 0, 255).astype(np.uint8))

    cap.release()
    writer.release()

    mux_cmd = [
        "ffmpeg", "-y", "-i", tmp_noaudio, "-i", str(input_video_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(output_video_path)
    ]
    subprocess.run(mux_cmd, capture_output=True, text=True)


def overlay_foreground(input_video_path, fg_image_path, output_video_path):
    cmd = [
        "ffmpeg", "-y", "-i", str(input_video_path), "-i", str(fg_image_path),
        "-filter_complex", "[1:v]scale=iw*0.3:-1[fg];[0:v][fg]overlay=W-w-20:H-h-20",
        "-codec:a", "copy", str(output_video_path)
    ]
    subprocess.run(cmd, capture_output=True, text=True)


def apply_corner_light(input_video_path, color_hex, corner, output_video_path, base_intensity=0.25, motion_boost=0.5):
    color_hex = color_hex.lstrip("#")
    r, g, b = int(color_hex[0:2], 16), int(color_hex[2:4], 16), int(color_hex[4:6], 16)
    color_bgr = (b, g, r)

    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    yy, xx = np.mgrid[0:h, 0:w]
    corner_map = {
        "top-left": (0, 0), "top-right": (w, 0),
        "bottom-left": (0, h), "bottom-right": (w, h),
    }
    cx, cy = corner_map.get(corner, (w, h))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2)
    gradient = np.clip(1 - (dist / max_dist), 0, 1) ** 2

    tmp_noaudio = str(output_video_path) + "_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (w, h))

    prev_gray = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_level = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_level = float(np.mean(diff)) / 255.0
        prev_gray = gray

        intensity = base_intensity + motion_boost * min(motion_level * 5, 1.0)
        light_layer = np.zeros_like(frame, dtype=np.float32)
        for c in range(3):
            light_layer[:, :, c] = gradient * color_bgr[c] * intensity

        blended = np.clip(frame.astype(np.float32) + light_layer, 0, 255).astype(np.uint8)
        writer.write(blended)

    cap.release()
    writer.release()

    mux_cmd = [
        "ffmpeg", "-y", "-i", tmp_noaudio, "-i", str(input_video_path),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(output_video_path)
    ]
    subprocess.run(mux_cmd, capture_output=True, text=True)


FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <title>BolChehra — Talking Face Generator</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 560px; margin: 60px auto; padding: 0 20px; color: #1a1a1a; }
    .brand { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
    .brand svg { width: 44px; height: 44px; flex-shrink: 0; }
    .brand h1 { font-size: 26px; margin: 0; letter-spacing: -0.5px; }
    p.sub { color: #666; margin-top: 4px; margin-bottom: 0; }
    label { display: block; font-weight: 600; margin-top: 20px; margin-bottom: 6px; }
    input[type="file"], textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 8px; font-size: 14px; box-sizing: border-box; }
    textarea { min-height: 90px; resize: vertical; }
    button { margin-top: 24px; width: 100%; padding: 12px; background: #d97757; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; }
    button:disabled { background: #ccc; cursor: not-allowed; }
    #status { margin-top: 16px; font-size: 14px; color: #555; }
    video { margin-top: 20px; width: 100%; border-radius: 8px; }
    .limit-notice { margin-top: 8px; padding: 10px 12px; background: #fff4e5; border: 1px solid #f0b955; border-left: 4px solid #d97757; border-radius: 6px; font-size: 13px; color: #7a4a1a; line-height: 1.5; }
    .char-counter { margin-top: 6px; font-size: 12px; color: #888; text-align: right; }
    .char-counter.over { color: #c0392b; font-weight: 600; }
  </style>
</head>

<body>
  <div class="brand">
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
      <circle cx="50" cy="50" r="48" fill="#d97757" />
      <circle cx="50" cy="46" r="22" fill="#fff" />
      <circle cx="42" cy="42" r="3.2" fill="#d97757" />
      <circle cx="58" cy="42" r="3.2" fill="#d97757" />
      <path d="M40 54 Q50 62 60 54" stroke="#d97757" stroke-width="3" fill="none" stroke-linecap="round" />
      <path d="M14 38 Q8 50 14 62" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round" />
      <path d="M22 32 Q13 50 22 68" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.6" />
      <path d="M86 38 Q92 50 86 62" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round" />
      <path d="M78 32 Q87 50 78 68" stroke="#fff" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.6" />
    </svg>
    <h1>BolChehra</h1>
  </div>
  <p class="sub">Upload a photo, type a script, get a talking video.</p>

  <form id="genForm">
    <label for="image">Photo (face image)</label>
    <input type="file" id="image" accept="image/*" required>

    <label for="voiceSample">Apni awaaz ka sample (optional — agar server pe voice cloning enabled hai)</label>
    <input type="file" id="voiceSample" accept="audio/*">

    <label for="backgroundImage">Background image (optional — video ke peeche fit hoga)</label>
    <input type="file" id="backgroundImage" accept="image/*">

    <label for="foregroundImage">Foreground image (optional — video ke upar overlay hoga, PNG transparent best)</label>
    <input type="file" id="foregroundImage" accept="image/*">

    <label style="display:flex;align-items:center;gap:8px;font-weight:600;margin-top:20px;">
      <input type="checkbox" id="enableLight" style="width:auto;">
      Corner light effect (movement ke saath react karega)
    </label>
    <div id="lightOptions" style="display:none;margin-top:10px;display:flex;gap:16px;">
      <div style="flex:1;">
        <label for="lightColor">Light color</label>
        <input type="color" id="lightColor" value="#00c8ff" style="height:40px;padding:2px;">
      </div>
      <div style="flex:1;">
        <label for="lightCorner">Corner</label>
        <select id="lightCorner" style="width:100%;padding:10px;border:1px solid #ccc;border-radius:8px;">
          <option value="top-left">Top-left</option>
          <option value="top-right">Top-right</option>
          <option value="bottom-left">Bottom-left</option>
          <option value="bottom-right" selected>Bottom-right</option>
        </select>
      </div>
    </div>

    <label for="script">Script</label>
    <textarea id="script" placeholder="Apna script yahan likho..." required></textarea>
    <div class="char-counter" id="charCounter">0 / 5000 characters</div>
    <div id="estimateNote" style="margin-top:4px;font-size:12px;color:#888;text-align:right;"></div>
    <div class="limit-notice">
      ⚠️ <strong>Script 5000 characters se zyada na ho.</strong> Isse zyada lamba text ek saath process nahi ho payega — audio/video generation fail ho sakta hai.
    </div>

    <button type="submit" id="submitBtn">Generate Video</button>
  </form>

  <div id="status"></div>
  <div id="timer" style="margin-top:6px;font-size:13px;color:#888;"></div>
  <div id="progressBarWrap" style="display:none;margin-top:10px;height:6px;background:#f0e5df;border-radius:4px;overflow:hidden;">
    <div id="progressBar" style="height:100%;width:30%;background:#d97757;animation:slide 1.4s ease-in-out infinite;"></div>
  </div>
  <style>
    @keyframes slide {
      0% { transform: translateX(-100%); }
      100% { transform: translateX(400%); }
    }
  </style>
  <video id="resultVideo" controls style="display:none;"></video>

  <script>
    const API_URL = "";

    const form = document.getElementById("genForm");
    const statusEl = document.getElementById("status");
    const btn = document.getElementById("submitBtn");
    const videoEl = document.getElementById("resultVideo");
    const scriptEl = document.getElementById("script");
    const charCounterEl = document.getElementById("charCounter");
    const MAX_CHARS = 5000;

    function estimateMinutes(charCount) {
      // Rough heuristic based on observed processing times (CPU server):
      // ~3 min base overhead (image processing + model load) + ~5 min per 500 characters
      const est = 3 + (charCount / 500) * 5;
      return Math.max(2, Math.round(est));
    }

    scriptEl.addEventListener("input", () => {
      const len = scriptEl.value.length;
      charCounterEl.textContent = `${len} / ${MAX_CHARS} characters`;
      charCounterEl.classList.toggle("over", len > MAX_CHARS);

      const estimateEl = document.getElementById("estimateNote");
      if (len > 0 && len <= MAX_CHARS) {
        const est = estimateMinutes(len);
        estimateEl.textContent = `Estimated wait: ~${est} min`;
      } else {
        estimateEl.textContent = "";
      }
    });

    document.getElementById("enableLight").addEventListener("change", (e) => {
      document.getElementById("lightOptions").style.display = e.target.checked ? "flex" : "none";
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const imageFile = document.getElementById("image").files[0];
      const script = document.getElementById("script").value;

      if (!imageFile || !script) return;

      if (script.length > MAX_CHARS) {
        statusEl.textContent = `Script bahut lamba hai (${script.length} characters). Kripya ${MAX_CHARS} characters se kam karo.`;
        return;
      }

      const formData = new FormData();
      formData.append("image", imageFile);
      formData.append("script", script);

      const voiceFile = document.getElementById("voiceSample").files[0];
      if (voiceFile) formData.append("voice_sample", voiceFile);

      const bgFile = document.getElementById("backgroundImage").files[0];
      if (bgFile) formData.append("background_image", bgFile);

      const fgFile = document.getElementById("foregroundImage").files[0];
      if (fgFile) formData.append("foreground_image", fgFile);

      if (document.getElementById("enableLight").checked) {
        formData.append("light_color", document.getElementById("lightColor").value);
        formData.append("light_corner", document.getElementById("lightCorner").value);
      }

      btn.disabled = true;
      btn.textContent = "Generating... (this can take a few minutes)";
      statusEl.textContent = "Processing your video, please wait...";
      videoEl.style.display = "none";
      document.getElementById("progressBarWrap").style.display = "block";
      const estimatedMin = estimateMinutes(script.length);
      let elapsedSec = 0;
      const timerEl = document.getElementById("timer");
      timerEl.textContent = `Time elapsed: 0:00 (estimated ~${estimatedMin} min for this script)`;
      const timerInterval = setInterval(() => {
        elapsedSec++;
        const m = Math.floor(elapsedSec / 60);
        const s = (elapsedSec % 60).toString().padStart(2, "0");
        timerEl.textContent = `Time elapsed: ${m}:${s} (estimated ~${estimatedMin} min for this script)`;
      }, 1000);

      try {
        const res = await fetch(`${API_URL}/generate`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error || "Something went wrong");
        }

        const blob = await res.blob();
        const videoUrl = URL.createObjectURL(blob);
        videoEl.src = videoUrl;
        videoEl.style.display = "block";
        statusEl.textContent = "Done!";
      } catch (err) {
        statusEl.textContent = "Error: " + err.message;
      } finally {
        clearInterval(timerInterval);
        document.getElementById("progressBarWrap").style.display = "none";
        btn.disabled = false;
        btn.textContent = "Generate Video";
      }
    });
  </script>
</body>

</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FRONTEND_HTML


BASE_DIR = Path(__file__).resolve().parent
SADTALKER_DIR = BASE_DIR / "SadTalker"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

MAX_SCRIPT_CHARS = 5000


@app.get("/health")
def health():
    return {"status": "ok", "voice_cloning_available": XTTS_AVAILABLE}

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    script: str = Form(...),
    voice_sample: UploadFile = File(None),
    background_image: UploadFile = File(None),
    foreground_image: UploadFile = File(None),
    light_color: str = Form(None),
    light_corner: str = Form(None),
):
    if len(script) > MAX_SCRIPT_CHARS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Script too long ({len(script)} chars). Max allowed is {MAX_SCRIPT_CHARS} characters."},
        )

    job_id = str(uuid.uuid4())[:8]

    try:
        image_path = UPLOAD_DIR / f"{job_id}_{image.filename}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Voice: use XTTS cloning only if that environment is actually configured;
        # otherwise fall back to gTTS automatically (no crash either way).
        if voice_sample is not None and XTTS_AVAILABLE:
            audio_path = UPLOAD_DIR / f"{job_id}.wav"
            sample_path = UPLOAD_DIR / f"{job_id}_sample.wav"
            with open(sample_path, "wb") as f:
                shutil.copyfileobj(voice_sample.file, f)
            clone_cmd = [XTTS_PYTHON_PATH, XTTS_CLONE_SCRIPT, script, str(sample_path.resolve()), str(audio_path.resolve())]
            clone_env = os.environ.copy()
            clone_env["COQUI_TOS_AGREED"] = "1"
            clone_result = subprocess.run(clone_cmd, capture_output=True, text=True, env=clone_env)
            if clone_result.returncode != 0:
                return JSONResponse(status_code=500, content={"error": "Voice cloning failed: " + clone_result.stderr[-500:]})
        else:
            audio_path = UPLOAD_DIR / f"{job_id}.wav"
            engine = pyttsx3.init()
            engine.save_to_file(script, str(audio_path))
            engine.runAndWait()

        job_result_dir = RESULT_DIR / job_id
        job_result_dir.mkdir(exist_ok=True)

        cmd = [
            "python3.8", "inference.py",
            "--driven_audio", str(audio_path.resolve()),
            "--source_image", str(image_path.resolve()),
            "--result_dir", str(job_result_dir.resolve()),
            "--preprocess", "full",
            "--enhancer", "gfpgan",
            "--size", "512",
            "--batch_size", "32",
        ]

        try:
            subprocess.run(cmd, cwd=str(SADTALKER_DIR), check=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            return JSONResponse(status_code=500, content={"error": "Inference failed", "detail": str(e)})
        except subprocess.TimeoutExpired:
            return JSONResponse(status_code=504, content={"error": "Inference timed out (video generation takes too long on this server)"})

        mp4_files = list(job_result_dir.glob("**/*.mp4"))
        if not mp4_files:
            return JSONResponse(status_code=500, content={"error": "No output video found"})

        current_video = mp4_files[0]

        if background_image is not None:
            bg_path = UPLOAD_DIR / f"{job_id}_bg.png"
            with open(bg_path, "wb") as f:
                shutil.copyfileobj(background_image.file, f)
            bg_output = job_result_dir / f"{job_id}_with_bg.mp4"
            replace_background(current_video, bg_path, bg_output)
            if bg_output.exists():
                current_video = bg_output

        if foreground_image is not None:
            fg_path = UPLOAD_DIR / f"{job_id}_fg.png"
            with open(fg_path, "wb") as f:
                shutil.copyfileobj(foreground_image.file, f)
            fg_output = job_result_dir / f"{job_id}_with_fg.mp4"
            overlay_foreground(current_video, fg_path, fg_output)
            if fg_output.exists():
                current_video = fg_output

        if light_color and light_corner:
            light_output = job_result_dir / f"{job_id}_with_light.mp4"
            apply_corner_light(current_video, light_color, light_corner, light_output)
            if light_output.exists():
                current_video = light_output

        return FileResponse(current_video, media_type="video/mp4", filename=f"{job_id}.mp4")

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {str(e)}", "trace": traceback.format_exc()[-1500:]},
        )