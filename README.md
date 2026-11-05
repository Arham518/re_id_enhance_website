# RE Touch ID — Website & Enhancement App

Web front end and Flask API for **RE Touch ID – Aged & Labour Fingerprint Enhancement**. The site explains the product, shows a gallery, and includes a scanner flow that uploads a fingerprint image to a local Flask backend for classification and enhancement-oriented processing.

Built with **React, Vite, Tailwind CSS, React Three Fiber, Motion, React Router and Axios** on the front end, and **Flask, OpenCV, Keras, scikit-learn, Pillow, NumPy and joblib** on the backend. Optional CycleGAN training / inference scripts live under `CycleGAN/`.

> **Status:** research / portfolio demo. The scanner talks to `http://127.0.0.1:5000/upload`.
> Stage-2 Keras weights are loaded from a **local absolute path** in `backend/app.py` — update `MODEL_PATH` for your machine. Without the `.keras` file, enhancement falls back according to what is available (classifier / classical enhancement).

## What works now vs. simulated vs. planned

| Area | Status |
| --- | --- |
| Marketing pages (Home, About, Services, Gallery) | ✅ Front-end works |
| Scanner UI (upload, preview, result panels) | ✅ Works |
| Flask `/upload` + `/health` + static upload/processed routes | ✅ Works when backend is running |
| Stage-1 scikit-learn classifier (`manual_fingerprint_classifier.pkl`) | ✅ Loads if the pickle is present |
| Stage-2 Keras enhancer (`G_L2N.keras`) + `aged_enhancement.py` | ✅ When `MODEL_PATH` points at a real file |
| CycleGAN train / infer scripts | ✅ Scripts in `CycleGAN/` — separate from the live API |
| Hosted production API with auth | 🗓 Planned |
| Gemini / Google AI Studio features | 🧪 Dependency present from earlier scaffolding — not required for the scanner path |

## Setup

### Frontend

```bash
git clone https://github.com/Arham518/re_id_enhance_website.git
cd re_id_enhance_website
npm install
npm run dev       # Vite on port 3000
npm run build
```

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Also install opencv-python and tensorflow/keras if you need stage-2 enhancement
python app.py                      # http://127.0.0.1:5000
```

Edit `MODEL_PATH` in `backend/app.py` so it points at your `G_L2N.keras` (or place weights where you prefer). Keep real API keys out of git — use `.env.local` only if you enable optional Gemini tooling.

## Project structure

```
src/
  pages/          Home, About, Services, Gallery, Scanner
  components/     shared UI + 3D accents
  lib/            helpers
backend/
  app.py                      Flask API
  aged_enhancement.py         classical / pipeline enhancement helpers
  manual_fingerprint_classifier.pkl
  requirements.txt
CycleGAN/                     train_cyclegan_fingerprint.py, infer_cyclegan.py, outputs/
public/
```

## Known limitations

- CORS is open (`*`) for local demo use — tighten before any public deploy.
- Absolute Windows `MODEL_PATH` will not work on other machines until changed.
- Uploaded images are written under `backend/uploads/` and `backend/processed/` at runtime (gitignored in a proper setup).
- Package name in `package.json` is still the generic `react-example`.

## Roadmap

1. Configurable model path via environment variable
2. Authenticated, rate-limited hosted API
3. Clearer offline / model-missing UI states in the Scanner

## Security

Never commit API keys, `.env` files, or private model weights you are not allowed to share. The demo Flask app has no auth — do not expose it on the public internet as-is.

## Author

Muhammad Arham · [GitHub Arham518](https://github.com/Arham518)
