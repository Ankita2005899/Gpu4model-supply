FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    git wget ffmpeg build-essential libgl1 libglib2.0-0 espeak libespeak1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/lib/x86_64-linux-gnu/libespeak.so.1 /usr/lib/libespeak.so.1 || true

WORKDIR /app/backend

# 1. Clone SadTalker
RUN git clone https://github.com/Winfredy/SadTalker.git SadTalker

# 2. Install SadTalker's own requirements (CPU-only torch, since Render has no GPU)
WORKDIR /app/backend/SadTalker
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN bash scripts/download_models.sh

# 3. Install API dependencies (FastAPI, gTTS, rembg for background/foreground/light effects)
WORKDIR /app/backend
COPY requirements.txt .
RUN pip install -r requirements.txt

# NOTE: Voice cloning (XTTS) is NOT included in this image. It needs a separate
# Python 3.10 environment (its torch version conflicts with SadTalker's torch 1.12).
# If you want voice cloning in production, set up a separate microservice for it
# and point XTTS_PYTHON_PATH / XTTS_CLONE_SCRIPT env vars at it. Without that,
# the app automatically falls back to gTTS - no crash either way.

COPY app.py .

EXPOSE 10000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
