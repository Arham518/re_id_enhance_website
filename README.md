# RE Touch ID — Website & Enhancement App

Frontend and supporting backend for **RE Touch ID – Aged & Labour Fingerprint Enhancement Software**. The site explains the product, shows a gallery, and includes a scanner flow that sends fingerprint images to a Flask API for classification and enhancement-oriented processing.

This repository is the web-facing project (React UI plus Python backend). Related training / CycleGAN code may also live here under `CycleGAN/`.

## What it does

- Marketing and product pages (Home, About, Services, Gallery, Scanner)
- Upload flow for degraded, aged, or labour-worn fingerprints
- Flask API with scikit-learn classifier support
- Optional CycleGAN-related scripts for fingerprint enhancement research

## Tech stack

**Frontend:** React, Vite, Tailwind CSS, React Three Fiber, Motion, React Router  
**Backend:** Flask, Flask-CORS, scikit-learn, Pillow, NumPy, joblib

## Run the frontend

```bash
git clone https://github.com/Arham518/re_id_enhance_website.git
cd re_id_enhance_website
npm install
npm run dev
```

Optional: create `.env.local` if you use Gemini or other keys locally (do not commit real secrets).

## Run the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Place model files (for example `manual_fingerprint_classifier.pkl`) where the backend expects them.

## Developer

**Muhammad Arham** — [GitHub](https://github.com/Arham518)
