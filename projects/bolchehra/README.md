# Talking Face Generator (SadTalker Web App) — BolChehra

Photo + script upload karke talking-face video generate karta hai (SadTalker use karke), plus background replace, foreground overlay, aur motion-reactive corner-light effects.

## ⚠️ Important reality check before deploying

SadTalker GPU pe fast chalta hai, CPU pe **bahut slow** (kai minutes per video).

- **Render ke free/paid plans mein GPU nahi milta** — sirf CPU. Video generation Render pe 3-15+ minute le sakta hai per request, aur free tier timeout/sleep bhi hoga.
- Agar production-quality speed chahiye, GPU-backed host use karo: **RunPod** (~$0.20-0.40/hr for T4), **Vast.ai** (~$0.15-0.30/hr), **Lambda Labs**, ya AWS EC2 g4dn. Ye sab **paid** hain — GPU hosting free permanently kahin nahi milta.
- Docker image bhi bada hoga (SadTalker models ~2-3 GB).

**Voice cloning (XTTS) is NOT included in this Docker image** by default — XTTS needs its own Python 3.10 environment with a different torch version than SadTalker's, which is fragile to set up reliably inside one container. Without it configured, uploading a voice sample is silently ignored and gTTS is used instead (no crash). If you want voice cloning in production, run XTTS as a separate microservice and set the `XTTS_PYTHON_PATH` / `XTTS_CLONE_SCRIPT` environment variables to point at it.

## Project structure
```
talking-face-app/
├── backend/
│   ├── app.py            # FastAPI server (also serves the frontend at "/")
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html        # Standalone copy, for hosting frontend separately if needed
└── README.md
```

Note: `app.py` already serves the full frontend UI at `/` — you don't strictly need a separate `frontend/index.html` unless you want to host the UI on a different domain (e.g., GitHub Pages) from the API.

## 1. Local test (optional, before deploying)

```bash
cd backend
pip install -r requirements.txt
git clone https://github.com/Winfredy/SadTalker.git
cd SadTalker && pip install -r requirements.txt && bash scripts/download_models.sh
cd ..
uvicorn app:app --reload
```
Then open `http://localhost:8000` in a browser.

## 2. Push to GitHub

```bash
cd talking-face-app
git init
git add .
git commit -m "Initial commit: talking face generator app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
(Create the empty repo on GitHub first at github.com/new — don't initialize it with a README there, to avoid merge conflicts.)

**Before pushing, double check `api_key.txt` and any real API keys/tokens are NOT committed** — `.gitignore` already excludes `api_key.txt`, but always verify with `git status` before your first commit.

## 3. Deploy backend on Render (CPU-only, slow)

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Root Directory:** `backend`
   - **Environment:** Docker (Render will detect the `Dockerfile`)
   - **Instance type:** at least 2GB RAM
4. Click **Create Web Service** — first build will take a while (cloning SadTalker + downloading models happens at build time)
5. Once deployed, you'll get a URL like `https://your-app.onrender.com`

Since `app.py` serves the frontend itself at `/`, this URL alone gives you a working app — no separate frontend hosting needed.

## 4. (Optional) Host frontend separately

If you'd rather host the UI on a different domain (e.g., GitHub Pages), use `frontend/index.html` and update:
```js
const API_URL = "https://your-app.onrender.com";
```
to your actual Render URL, then host that file anywhere static files are served.

## Notes
- Backend generates TTS audio in Indian-accent English (`gTTS`, `tld='co.in'`) by default.
- Background removal uses `rembg` (not mediapipe — mediapipe's `solutions` API proved unreliable across versions during testing).
- Uploaded images/videos are stored temporarily in `backend/uploads/` and `backend/results/` — not committed to git (see `.gitignore`).
