# CPU-only image. This will be large (several GB) because it bakes in
# LivePortrait + Wav2Lip + insightface model weights so the app doesn't
# need internet access at request time. Build time will be long (expect
# 15-40+ minutes) and you need a Render plan whose build machine has enough
# RAM/disk -- see README.md.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
COPY requirements.txt .
# CPU-only torch wheel (much smaller than the CUDA build, and CUDA is
# useless on Render anyway since it has no GPUs).
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
      --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- LivePortrait ---
RUN git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /opt/LivePortrait
RUN pip install --no-cache-dir -r /opt/LivePortrait/requirements.txt || true
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='KwaiVGI/LivePortrait', local_dir='/opt/LivePortrait/pretrained_weights')"

# --- Wav2Lip ---
RUN git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /opt/Wav2Lip
RUN mkdir -p /opt/Wav2Lip/checkpoints && \
    wget -q "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth" \
      -O /opt/Wav2Lip/checkpoints/wav2lip_gan.pth
RUN sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,/' \
    /opt/Wav2Lip/audio.py

# --- insightface face-swap model ---
# NOTE: inswapper_128.onnx is community-distributed (not officially hosted by
# insightface). You must source it yourself and host it somewhere you control
# (e.g. your own S3/GCS bucket, or bake it in via COPY) -- put a reachable URL
# here, or use the COPY line below if you place the file next to the Dockerfile.
RUN mkdir -p /opt/models
# Option A: download from a URL you control:
# RUN wget -q "<YOUR_HOSTED_URL>/inswapper_128.onnx" -O /opt/models/inswapper_128.onnx
# Option B: copy a local file you've placed next to this Dockerfile:
# COPY inswapper_128.onnx /opt/models/inswapper_128.onnx

ENV LIVEPORTRAIT_DIR=/opt/LivePortrait
ENV WAV2LIP_DIR=/opt/Wav2Lip
ENV INSWAPPER_MODEL=/opt/models/inswapper_128.onnx

COPY . .

ENV PORT=5000
EXPOSE 5000

# Single worker: this pipeline is heavy (CPU + RAM per job), so don't run
# multiple concurrent jobs on a small instance. Long timeout because the
# HTTP handlers themselves are fast (background thread does the real work) --
# but gunicorn's own defaults are fine here since we never block a request.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
