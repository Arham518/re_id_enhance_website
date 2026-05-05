import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# =========================
# Utils (same as training)
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_img(img):
    return (img / 127.5) - 1.0


def denormalize_img(img):
    return (img + 1.0) / 2.0


def apply_clahe(img):
    arr = img.astype(np.uint8)
    if len(arr.shape) == 3:
        arr = arr[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(arr)
    return out[:, :, None]


# =========================
# Load image
# =========================
def load_image(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))

    img = apply_clahe(img)  # IMPORTANT (same as training)

    img = img.astype(np.float32)
    img = normalize_img(img)

    return img


# =========================
# Save comparison
# =========================
def save_comparison(inp, out, save_path):
    inp = denormalize_img(inp)
    out = denormalize_img(out)

    inp = (inp * 255).astype(np.uint8)
    out = (out * 255).astype(np.uint8)

    combined = np.concatenate([inp, out], axis=1)

    cv2.imwrite(save_path, combined)


# =========================
# Inference
# =========================
def run_inference(model, input_path, output_dir, img_size):
    ensure_dir(output_dir)

    if os.path.isdir(input_path):
        paths = sorted(list(Path(input_path).glob("*.png")))
    else:
        paths = [Path(input_path)]

    print(f"\nProcessing {len(paths)} images...")

    for i, p in enumerate(paths):
        img = load_image(str(p), img_size)

        inp = np.expand_dims(img, axis=0)  # [1, H, W, 1]

        pred = model(inp, training=False)[0].numpy()

        save_path = os.path.join(output_dir, f"{p.stem}_enhanced.png")
        comp_path = os.path.join(output_dir, f"{p.stem}_compare.png")

        # Save output only
        out_img = denormalize_img(pred)
        out_img = (out_img * 255).astype(np.uint8)
        cv2.imwrite(save_path, out_img)

        # Save comparison
        save_comparison(img, pred, comp_path)

        if i % 10 == 0:
            print(f"Processed {i}/{len(paths)}")

    print("\n✅ Inference complete!")


# =========================
# Main
# =========================
def main(args):
    print("\nLoading model...")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    run_inference(
        model,
        args.input_path,
        args.output_dir,
        args.img_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="infer_results")
    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()
    main(args)