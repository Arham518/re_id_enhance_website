import os
import cv2
import uuid
import joblib
import keras
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path

from aged_enhancement import enhance_fingerprint

# ========================= APP CONFIG =========================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = Path(__file__).parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"

UPLOAD_FOLDER.mkdir(exist_ok=True)
PROCESSED_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['PROCESSED_FOLDER'] = str(PROCESSED_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ========================= MODEL PATHS =========================

CLASSIFIER_PATH = BASE_DIR / "manual_fingerprint_classifier.pkl"

MODEL_PATH = Path(
    r"C:\Users\Arham Bhatti\Documents\re_id_enhance\CycleGAN\outputs\G_L2N.keras"
)

# ========================= LOAD MODELS =========================

print("Loading models...")

# -------- Stage 1 Classifier --------

try:
    classifier = joblib.load(str(CLASSIFIER_PATH))
    print("[SUCCESS] Stage 1 Classifier loaded.")
except Exception as e:
    classifier = None
    print(f"[ERROR] Failed to load Stage 1 Classifier: {e}")

# -------- Stage 2 Keras Model --------

try:
    model = keras.models.load_model(str(MODEL_PATH))
    print("[SUCCESS] Stage 2 Model (G_L2N) loaded.")
except Exception as e:
    model = None
    print(f"[ERROR] Failed to load Stage 2 Model: {e}")

# ========================= UTILS =========================

def is_fingerprint(img):
    """
    Detect whether image looks like fingerprint
    """

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    edges = cv2.Canny(gray, 30, 100)

    edge_density = np.count_nonzero(edges) / edges.size
    variance = np.var(gray)

    if edge_density < 0.05 or variance < 50:
        return False

    return True


def extract_features(img):
    """
    Extract Hu Moments features
    """

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, thresh = cv2.threshold(
        gray,
        128,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    moments = cv2.moments(thresh)

    hu = cv2.HuMoments(moments).flatten()

    for i in range(7):

        if hu[i] != 0:
            hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))
        else:
            hu[i] = 0

    return hu

# ========================= CLASS MAP =========================

CLASS_MAP = {
    0: "AGE",
    1: "LABOUR",
    2: "WHORL"
}

# ========================= ROUTES =========================

@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({
            'error': 'No file part'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'No selected file'
        }), 400

    filename = secure_filename(file.filename)

    unique_id = str(uuid.uuid4())[:8]

    save_name = f"{unique_id}_{filename}"

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        save_name
    )

    file.save(filepath)

    # ================= IMAGE READ =================

    img = cv2.imread(filepath)

    if img is None:
        return jsonify({
            'error': 'Invalid image format'
        }), 400

    img_resized = cv2.resize(img, (224, 224))

    # ================= SAVE INITIAL PROCESSED =================

    processed_filename = f"{unique_id}_processed.png"

    processed_path = os.path.join(
        app.config['PROCESSED_FOLDER'],
        processed_filename
    )

    cv2.imwrite(processed_path, img_resized)

    # ================= FINGERPRINT CHECK =================

    if not is_fingerprint(img_resized):

        return jsonify({
            "success": True,
            "result": {
                "status": "success",
                "message": "Image is not a fingerprint.",
                "original_image_url": f"/uploads/{save_name}",
                "processed_image_url": f"/processed/{processed_filename}",
                "metrics": {
                    "confidence": "100%",
                    "classification": "Unknown"
                }
            }
        }), 200

    # ================= STAGE 1 CLASSIFICATION =================

    if classifier is not None:

        try:
            features = extract_features(img_resized)

            probs = classifier.predict_proba([features])[0]

            max_conf = np.max(probs) * 100

            class_idx = np.argmax(probs)

            classification = CLASS_MAP.get(
                class_idx,
                "Unknown"
            )

        except Exception as e:

            print(f"Classifier prediction error: {e}")

            max_conf = 0
            classification = "Unknown"

    else:

        max_conf = 0
        classification = "Unknown"

    if max_conf < 40:
        classification = "Unknown"

    # ================= PROCESSING =================

    result_text = classification

    final_img = img_resized

    if classification == "AGE":

        try:
            final_img = enhance_fingerprint(img_resized)

            cv2.imwrite(processed_path, final_img)

        except Exception as e:
            print(f"Enhancement error: {e}")

    # ================= STAGE 2 MODEL =================

    if model is not None and classification in ["AGE", "LABOUR"]:

        try:

            k_img = cv2.resize(final_img, (224, 224))

            if len(k_img.shape) == 2:
                k_img = cv2.cvtColor(
                    k_img,
                    cv2.COLOR_GRAY2RGB
                )

            k_img = k_img.astype('float32') / 255.0

            k_img = np.expand_dims(k_img, axis=0)

            preds = model.predict(k_img)

            print(
                f"Keras prediction for {classification}: {preds}"
            )

        except Exception as e:

            print(f"Keras prediction error: {e}")

    # ================= RESPONSE =================

    return jsonify({
        "success": True,
        "result": {
            "status": "success",
            "message": f"{result_text} processed successfully.",
            "original_image_url": f"/uploads/{save_name}",
            "processed_image_url": f"/processed/{processed_filename}",
            "metrics": {
                "confidence": f"{max_conf:.1f}%",
                "classification": result_text
            }
        }
    }), 200

# ========================= FILE ROUTES =========================

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


@app.route('/processed/<filename>')
def get_processed_file(filename):

    return send_from_directory(
        app.config['PROCESSED_FOLDER'],
        filename
    )

# ========================= HEALTH CHECK =========================

@app.route('/health', methods=['GET'])
def health():

    return jsonify({
        "status": "active",
        "classifier_loaded": classifier is not None,
        "keras_model_loaded": model is not None
    })

# ========================= MAIN =========================

if __name__ == '__main__':

    from waitress import serve

    print("[INFO] BioScan Backend starting on port 5000...")

    serve(
        app,
        host='0.0.0.0',
        port=5000
    )