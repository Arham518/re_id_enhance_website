import os
import cv2
import numpy as np
import joblib
import keras
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid

# In-line enhancement logic
def enhance_fingerprint(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    background = cv2.GaussianBlur(img, (31,31), 0)
    corrected = cv2.subtract(img, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    contrast = clahe.apply(corrected)
    blur = cv2.GaussianBlur(contrast, (5,5), 0)
    sharpened = cv2.addWeighted(contrast, 1.7, blur, -0.7, 0)
    return cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ========================= CONFIG =========================
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"
UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['PROCESSED_FOLDER'] = str(PROCESSED_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ========================= MODELS =========================
CLASSIFIER_PATH = BASE_DIR / "manual_fingerprint_classifier.pkl"
MODEL_PATH = BASE_DIR / "backend" / "G_L2N.keras"
MODEL_PATH_H5 = BASE_DIR / "backend" / "G_L2N.h5"

print("[INFO] Initializing Biometric Models...")
try:
    classifier = joblib.load(str(CLASSIFIER_PATH))
    print("[SUCCESS] Stage 1 Classifier loaded.")
except Exception as e:
    print(f"[ERROR] Stage 1 Classifier load failed: {e}")
    classifier = None

model = None
try:
    if MODEL_PATH.exists():
        model = keras.models.load_model(str(MODEL_PATH), safe_mode=False)
        print("[SUCCESS] Stage 2 Model (G_L2N) loaded.")
    elif MODEL_PATH_H5.exists():
        model = keras.models.load_model(str(MODEL_PATH_H5), safe_mode=False)
        print("[SUCCESS] Stage 2 Model (G_L2N .h5) loaded.")
    else:
        print("[ERROR] Stage 2 Model (G_L2N) not found.")
except Exception as e:
    print(f"[ERROR] Stage 2 Model load failed: {e}")

# ========================= UTILS =========================
def is_fingerprint(img):
    """Robust structural check to detect non-fingerprints."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    mid_tones = np.sum((norm > 64) & (norm < 192)) / norm.size
    # Thresholds tuned to reject faces/objects while keeping fingerprints
    if lap_var < 8000 or mid_tones > 0.18:
        return False
    return True

def extract_features(img):
    """Extract Hu Moments for the Stage 1 Classifier."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i])) if hu[i] != 0 else 0
    return hu

# Labels for Stage 2
STAGE2_MAP = { 0: "Aged", 1: "Labour", 2: "Normal" }

# ========================= ROUTES =========================
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        
        # 1. Preprocessing (MANDATORY)
        save_name = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
            
        img_resized = cv2.resize(img, (224, 224))
        processed_filename = f"{unique_id}_processed.png"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, img_resized)

        # 2. Stage 1: Classification & Confidence Check
        max_conf = 0
        s1_class = "Unknown"
        if classifier:
            if is_fingerprint(img_resized):
                features = extract_features(img_resized)
                probs = classifier.predict_proba([features])[0]
                max_conf = np.max(probs) * 100
                s1_class = STAGE2_MAP.get(np.argmax(probs), "Normal")
            else:
                max_conf = 0 # Force rejection for non-fingerprints

        # STRICT RULE: If confidence < 40, REJECT
        if max_conf < 40:
            return jsonify({
                "success": True,
                "result": {
                    "status": "unknown",
                    "message": "Please upload a valid fingerprint image.",
                    "original_image_url": f"/uploads/{save_name}",
                    "processed_image_url": f"/uploads/{save_name}",
                    "metrics": {
                        "confidence": f"{max_conf:.1f}%",
                        "classification": "Invalid Image"
                    }
                }
            }), 200

        # 3. Stage 2: Model Routing
        final_classification = s1_class
        if model:
            try:
                k_img = img_resized.astype('float32') / 255.0
                if len(k_img.shape) == 2:
                    k_img = cv2.cvtColor(k_img, cv2.COLOR_GRAY2RGB)
                k_img = np.expand_dims(k_img, axis=0)
                
                preds = model.predict(k_img)
                # Classifier mode
                if len(preds.shape) == 2 and preds.shape[1] <= 10:
                    final_classification = STAGE2_MAP.get(np.argmax(preds[0]), "Normal")
                # Generator mode
                elif len(preds.shape) == 4:
                    gen_img = (preds[0] * 255).astype(np.uint8)
                    cv2.imwrite(processed_path, gen_img)
            except Exception as e:
                print(f"[ERROR] Stage 2 Inference failed: {e}")

        # Always apply enhancement for Aged fingerprints if not already done by model
        if final_classification == "Aged":
            enhanced_img = enhance_fingerprint(img_resized)
            cv2.imwrite(processed_path, enhanced_img)
        
        return jsonify({
            "success": True,
            "result": {
                "status": "success",
                "message": f"{final_classification} processed successfully.",
                "original_image_url": f"/uploads/{save_name}",
                "processed_image_url": f"/processed/{processed_filename}",
                "metrics": {
                    "confidence": f"{max_conf:.1f}%",
                    "classification": f"{final_classification} Fingerprint"
                }
            }
        }), 200

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def get_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    from waitress import serve
    print("[INFO] BioScan Production Server starting on port 5000...")
    serve(app, host='0.0.0.0', port=5000)
