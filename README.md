# Talking Character Studio

Flask web app version of the notebook pipeline: upload a **base image**, a
**driving video**, and a **face-swap photo**; pick which face in the base
image to replace; get back a video where that character has the face-swap
photo's identity, moving and speaking according to the driving video.

## How it works

1. `/` — upload the base image.
2. `/detect` — server runs face detection (MediaPipe), shows a numbered
   preview, and asks for the driving video + face-swap photo + which face
   to target.
3. `/submit/<job_id>` — kicks off a **background thread** that:
   - face-swaps the photo onto the chosen face (insightface `inswapper`)
   - crops that region with padding
   - animates head motion with **LivePortrait**, driven by the video
   - tightens lip-sync with **Wav2Lip**, driven by the video's audio
   - composites the result back onto the original untouched image
   - muxes the audio back in
4. `/status/<job_id>` — poll until done, then download.

Processing runs in a background thread (not inside the request/response
cycle) specifically so the HTTP request returns instantly and doesn't hit
Render's request timeout — the actual work can take minutes to hours.

## Before you deploy: things that WILL bite you

I want to be direct about this rather than let you find out the hard way:

1. **Render has no GPU support.** Everything here runs on CPU. A clip that
   takes ~1-2 minutes on a Colab GPU could take tens of minutes to a few
   hours here, depending on video length and your instance's CPU. You said
   you're OK with that — just confirming it's not a bug, it's the tradeoff.

2. **Do not use Render's Free or Starter plan.** LivePortrait + Wav2Lip +
   insightface loaded in the same process commonly need 4-8GB+ RAM. Small
   instances will OOM-kill the job partway through, usually with no useful
   error. `render.yaml` here specifies `plan: standard` as a floor, not a
   ceiling — go up if jobs keep dying.

3. **The Docker image is large** (multiple GB — PyTorch + LivePortrait
   weights + Wav2Lip checkpoint + insightface models). Build time on Render
   will likely be 15-40+ minutes. That's normal, not a hang.

4. **`inswapper_128.onnx` is not included.** It's a community-distributed
   model (not officially hosted by insightface for licensing/misuse
   reasons), so you need to source it yourself and either:
   - host it somewhere you control and uncomment the `wget` line in the
     `Dockerfile`, or
   - place the `.onnx` file next to the `Dockerfile` and uncomment the
     `COPY` line instead.
   Do this before your first build — the app will fail at runtime without it.

5. **Jobs live in memory (`JOBS` dict in `app.py`).** If the Render service
   restarts or redeploys mid-job, that job's status is lost (the background
   thread also dies with the process). Fine for personal/occasional use;
   for anything more serious, swap this for a real job queue (Redis/Celery,
   or even just a SQLite table) so jobs survive restarts.

6. **One job at a time per instance, realistically.** The Dockerfile runs a
   single gunicorn worker on purpose — running two heavy CPU jobs
   concurrently on one instance will make both crawl or OOM. If you need
   concurrency, scale by adding instances, not workers.

7. **Untested end-to-end in this environment.** I don't have GPU or general
   internet access where I built this, so I could not actually run the full
   pipeline here to confirm it works byte-for-byte. The logic mirrors the
   notebook version we already discussed, but budget time to debug path or
   version issues on your first real deploy — treat this as a strong
   starting point, not a guaranteed-working drop-in.

8. **Face-swap technology and consent.** This pipeline can put a real
   person's face onto a video of someone else's body/motion. Only use
   photos of people who've consented to this, and be thoughtful about
   sharing outputs that could be mistaken for real footage of someone.

## Local test (before touching Render at all)

Running this locally first — even on CPU — will surface most bugs far
faster than a 30-minute Render build/deploy cycle each time:

```bash
pip install -r requirements.txt
# also need LivePortrait + Wav2Lip cloned locally, weights downloaded,
# and LIVEPORTRAIT_DIR / WAV2LIP_DIR / INSWAPPER_MODEL env vars set
# to point at them -- same as the Dockerfile does.
python app.py
# visit http://localhost:5000
```

## Deploying to Render

1. Push this folder to a GitHub repo.
2. In the Render dashboard: New -> Blueprint -> point at the repo (it'll
   read `render.yaml`), or New -> Web Service -> Docker if you'd rather
   configure the plan/disk by hand.
3. Make sure the plan is at least `standard` (see caveat #2).
4. First build will take a while (caveat #3) — that's expected.
5. Once live, visit the service URL and walk through the 3-step flow.

## File map

- `app.py` — Flask routes, background job runner, in-memory job status
- `pipeline.py` — the actual processing steps (face swap, LivePortrait,
  Wav2Lip, compositing, audio mux)
- `templates/` — the 3-step UI (upload -> pick face -> status/download)
- `Dockerfile` — CPU-only image, bakes in LivePortrait + Wav2Lip + weights
- `render.yaml` — Render Blueprint config
- `requirements.txt` — Python deps
