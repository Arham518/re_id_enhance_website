"""
CycleGAN Fingerprint Enhancement (Single-File Training Script)

Executive summary
- Unpaired CycleGAN (ResNet generator + PatchGAN discriminator) for fingerprint enhancement.
- Mixed precision enabled (mixed_float16) with LossScaleOptimizer and correct scaling/unscaling in custom loops.
- All losses/metrics cast to float32 for numerical stability; generator/discriminator outputs forced to float32.
- Leakage-free test split by fingerprint key; test uses only matched pairs on held-out keys.
- Balanced fingerprint-aware losses: adversarial + cycle (L1 + SSIM) + identity + frequency + edge + orientation + ridge + perceptual.

Notes
- CLAHE uses tf.numpy_function (convenient but not differentiable, not XLA compatible). For best performance,
  preprocess CLAHE offline and remove apply_clahe_tf() from the tf.data pipeline.
- MobileNetV2 perceptual features are optional; set --perc_weight 0 to disable.

Loss weights table (defaults)
| Name           | Default | Role |
|----------------|---------|------|
| lambda_cycle   | 10.0    | cycle L1 weight |
| lambda_ssim    | 2.0     | cycle SSIM weight |
| lambda_identity| 5.0     | identity L1 weight |
| freq_weight    | 0.0005  | FFT magnitude alignment |
| edge_weight    | 1.0     | Sobel edge tensor alignment |
| orient_weight  | 1.0     | ridge flow/orientation alignment |
| ridge_weight   | 1.5     | ridge magnitude alignment |
| perc_weight    | 0.2     | perceptual (MobileNetV2) feature alignment |

Mermaid timeline
timeline
  title CycleGAN fingerprint training loop
  section Setup
    Split keys : leakage-free test
    Input pipeline : map/shuffle/batch/prefetch
    Build models : G/D + perceptual
    Resume : restore checkpoint
  section Epoch
    LR schedule : constant then linear decay
    Train G : scale->grads->unscale->clip->apply
    Train D : buffer fakes, scale->unscale->clip->apply
    Eval : PSNR/SSIM
    Save : history/plots/visuals/ckpt
"""

import os
import re
import csv
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# =========================
# Perceptual backbone (created once)
# Build it BEFORE enabling mixed precision so it stays float32.
# =========================
PERCEPTUAL_SIZE = 224
_MOBILENETV2_PREPROCESS = tf.keras.applications.mobilenet_v2.preprocess_input
_PERCEPTUAL_MODEL = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(PERCEPTUAL_SIZE, PERCEPTUAL_SIZE, 3),
)
_PERCEPTUAL_MODEL.trainable = False

# =========================
# Mixed precision policy
# Using mixed_float16 requires LossScaleOptimizer in custom loops.
# =========================
mixed_precision.set_global_policy("mixed_float16")

# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def normalize_img(img: tf.Tensor) -> tf.Tensor:
    # uint8 [0,255] -> float32 [-1,1]
    return (img / 127.5) - 1.0

def denormalize_img(img: tf.Tensor) -> tf.Tensor:
    # [-1,1] -> [0,1]
    return (img + 1.0) / 2.0

def strip_labour_suffix(stem: str) -> str:
    return re.sub(r"_(CR|Obl|Zcut)$", "", stem, flags=re.IGNORECASE)

def key_from_path(path: str, is_labour: bool) -> str:
    stem = Path(path).stem
    return strip_labour_suffix(stem) if is_labour else stem

def list_pngs(folder: str):
    folder = Path(folder)
    return sorted([str(p) for p in folder.glob("*.png")])

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def save_history_csv(path: str, history: dict):
    keys = list(history.keys())
    n = max(len(v) for v in history.values()) if history else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch"] + keys)
        for i in range(n):
            row = [i + 1] + [(history[k][i] if i < len(history[k]) else "") for k in keys]
            w.writerow(row)

def append_epoch_metrics_csv(path: str, row_dict: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def get_initial_epoch_from_checkpoint(ckpt_manager, ckpt_every: int) -> int:
    if not ckpt_manager.latest_checkpoint:
        return 0
    ckpt_name = os.path.basename(ckpt_manager.latest_checkpoint)
    m = re.search(r"ckpt-(\d+)$", ckpt_name)
    if not m:
        return 0
    ckpt_number = int(m.group(1))
    return ckpt_number * ckpt_every

def make_splits(labour_files, real_files, test_ratio=0.2):
    """
    Split by key so evaluation is leakage-free.
    Training is unpaired (domain datasets).
    Testing uses matched (labour, real) pairs on held-out keys.
    """
    labour_by_key = {}
    real_by_key = {}

    for p in labour_files:
        k = key_from_path(p, is_labour=True)
        labour_by_key.setdefault(k, []).append(p)

    for p in real_files:
        k = key_from_path(p, is_labour=False)
        real_by_key[k] = p

    matched_keys = sorted(set(labour_by_key.keys()) & set(real_by_key.keys()))
    random.shuffle(matched_keys)

    n_test = max(1, int(len(matched_keys) * test_ratio))
    test_keys = set(matched_keys[:n_test])
    train_keys = set(matched_keys[n_test:])

    labour_train = [p for p in labour_files if key_from_path(p, True) not in test_keys]
    real_train = [p for p in real_files if key_from_path(p, False) not in test_keys]

    test_pairs = []
    for k in sorted(test_keys):
        lab_path = labour_by_key[k][0]
        real_path = real_by_key[k]
        test_pairs.append((lab_path, real_path, k))

    stats = {
        "labour_total": len(labour_files),
        "real_total": len(real_files),
        "matched_keys_total": len(matched_keys),
        "train_matched_keys": len(train_keys),
        "test_matched_keys": len(test_keys),
        "labour_train_files": len(labour_train),
        "real_train_files": len(real_train),
        "test_pairs": len(test_pairs),
    }
    return labour_train, real_train, test_pairs, stats

# =========================
# Layers (serializable)
# =========================
@tf.keras.utils.register_keras_serializable(package="FingerprintEnhancement")
class ReflectionPadding2D(layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, x):
        p_h, p_w = self.padding
        return tf.pad(x, [[0, 0], [p_h, p_h], [p_w, p_w], [0, 0]], mode="REFLECT")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"padding": self.padding})
        return cfg

@tf.keras.utils.register_keras_serializable(package="FingerprintEnhancement")
class InstanceNorm(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, x):
        # Stable moments in float32, then cast back.
        x_dtype = x.dtype
        x32 = tf.cast(x, tf.float32)
        mean, var = tf.nn.moments(x32, axes=[1, 2], keepdims=True)
        y32 = (x32 - mean) / tf.sqrt(var + self.epsilon)
        return tf.cast(y32, x_dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"epsilon": self.epsilon})
        return cfg

# =========================
# Model blocks
# =========================
def resnet_block(x, filters, name_prefix="res"):
    residual = x

    y = ReflectionPadding2D((1, 1), name=f"{name_prefix}_pad1")(x)
    y = layers.Conv2D(filters, 3, padding="valid", use_bias=False, name=f"{name_prefix}_conv1")(y)
    y = InstanceNorm(name=f"{name_prefix}_in1")(y)
    y = layers.ReLU(name=f"{name_prefix}_relu1")(y)

    y = ReflectionPadding2D((1, 1), name=f"{name_prefix}_pad2")(y)
    y = layers.Conv2D(filters, 3, padding="valid", use_bias=False, name=f"{name_prefix}_conv2")(y)
    y = InstanceNorm(name=f"{name_prefix}_in2")(y)

    return layers.Add(name=f"{name_prefix}_add")([residual, y])

def build_generator(img_size=224, channels=1, n_res_blocks=9, name="generator"):
    inp = layers.Input(shape=(img_size, img_size, channels), name=f"{name}_input")

    x = ReflectionPadding2D((3, 3), name=f"{name}_pad_in")(inp)
    x = layers.Conv2D(64, 7, padding="valid", use_bias=False, name=f"{name}_c7s1_64")(x)
    x = InstanceNorm(name=f"{name}_in_in")(x)
    x = layers.ReLU(name=f"{name}_relu_in")(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same", use_bias=False, name=f"{name}_d128")(x)
    x = InstanceNorm(name=f"{name}_in_d128")(x)
    x = layers.ReLU(name=f"{name}_relu_d128")(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same", use_bias=False, name=f"{name}_d256")(x)
    x = InstanceNorm(name=f"{name}_in_d256")(x)
    x = layers.ReLU(name=f"{name}_relu_d256")(x)

    for i in range(n_res_blocks):
        x = resnet_block(x, 256, name_prefix=f"{name}_res{i+1}")

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False, name=f"{name}_u128")(x)
    x = InstanceNorm(name=f"{name}_in_u128")(x)
    x = layers.ReLU(name=f"{name}_relu_u128")(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", use_bias=False, name=f"{name}_u64")(x)
    x = InstanceNorm(name=f"{name}_in_u64")(x)
    x = layers.ReLU(name=f"{name}_relu_u64")(x)

    x = ReflectionPadding2D((3, 3), name=f"{name}_pad_out")(x)

    # Force float32 output under mixed precision (recommended by TF mixed precision guide).
    out = layers.Conv2D(
        channels, 7, padding="valid", activation="tanh",
        dtype="float32", name=f"{name}_out"
    )(x)

    return tf.keras.Model(inp, out, name=name)

def build_discriminator(img_size=224, channels=1, name="discriminator"):
    inp = layers.Input(shape=(img_size, img_size, channels), name=f"{name}_input")

    x = layers.Conv2D(64, 4, strides=2, padding="same", name=f"{name}_c64")(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same", use_bias=False, name=f"{name}_c128")(x)
    x = InstanceNorm(name=f"{name}_in128")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same", use_bias=False, name=f"{name}_c256")(x)
    x = InstanceNorm(name=f"{name}_in256")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=1, padding="same", use_bias=False, name=f"{name}_c512")(x)
    x = InstanceNorm(name=f"{name}_in512")(x)
    x = layers.LeakyReLU(0.2)(x)

    # Force float32 logits under mixed precision.
    out = layers.Conv2D(
        1, 4, strides=1, padding="same",
        dtype="float32", name=f"{name}_out"
    )(x)
    return tf.keras.Model(inp, out, name=name)

# =========================
# Image buffer
# =========================
class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = int(max_size)
        self.buffer = []

    def query(self, images):
        output = []
        for img in tf.unstack(images, axis=0):
            img = tf.expand_dims(img, axis=0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                output.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, len(self.buffer) - 1)
                    old = self.buffer[idx]
                    self.buffer[idx] = img
                    output.append(old)
                else:
                    output.append(img)
        return tf.concat(output, axis=0)

# =========================
# Losses (float32-safe)
# =========================
_bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):
    real_logits = tf.cast(real_logits, tf.float32)
    fake_logits = tf.cast(fake_logits, tf.float32)
    real_loss = _bce(tf.ones_like(real_logits), real_logits)
    fake_loss = _bce(tf.zeros_like(fake_logits), fake_logits)
    return 0.5 * (real_loss + fake_loss)

def generator_adv_loss(fake_logits):
    fake_logits = tf.cast(fake_logits, tf.float32)
    return _bce(tf.ones_like(fake_logits), fake_logits)

def cycle_loss(real, cycled, lambda_cycle=10.0, lambda_ssim=2.0):
    real = tf.cast(real, tf.float32)
    cycled = tf.cast(cycled, tf.float32)

    l1 = tf.reduce_mean(tf.abs(real - cycled))

    # SSIM on [0,1] for stability.
    real_01 = denormalize_img(real)
    cycled_01 = denormalize_img(cycled)
    ssim = 1.0 - tf.reduce_mean(tf.image.ssim(real_01, cycled_01, max_val=1.0))
    return (lambda_cycle * l1) + (lambda_ssim * ssim)

def identity_loss(real, same, lambda_identity=5.0):
    real = tf.cast(real, tf.float32)
    same = tf.cast(same, tf.float32)
    return lambda_identity * tf.reduce_mean(tf.abs(real - same))

def frequency_loss(real, fake, weight=0.0005):
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)
    real_fft = tf.signal.fft2d(tf.cast(real[:, :, :, 0], tf.complex64))
    fake_fft = tf.signal.fft2d(tf.cast(fake[:, :, :, 0], tf.complex64))
    return tf.cast(weight, tf.float32) * tf.reduce_mean(tf.abs(tf.abs(real_fft) - tf.abs(fake_fft)))

def edge_loss(real, fake, weight=1.0):
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)
    real_edges = tf.image.sobel_edges(real)  # shape [B,H,W,1,2] (dy, dx)
    fake_edges = tf.image.sobel_edges(fake)
    return tf.cast(weight, tf.float32) * tf.reduce_mean(tf.abs(real_edges - fake_edges))

def orientation_loss(real, fake, weight=1.0, eps=1e-6):
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)

    real_edges = tf.image.sobel_edges(real)
    fake_edges = tf.image.sobel_edges(fake)

    # sobel_edges last dim order is [dy, dx]
    real_dy = real_edges[..., 0]
    real_dx = real_edges[..., 1]
    fake_dy = fake_edges[..., 0]
    fake_dx = fake_edges[..., 1]

    real_theta = tf.atan2(real_dy + eps, real_dx + eps)
    fake_theta = tf.atan2(fake_dy + eps, fake_dx + eps)

    return tf.cast(weight, tf.float32) * tf.reduce_mean(tf.abs(real_theta - fake_theta))

def ridge_loss(real, fake, weight=1.5):
    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)

    real_edges = tf.image.sobel_edges(real)
    fake_edges = tf.image.sobel_edges(fake)

    # magnitude over last dimension (dy, dx)
    real_mag = tf.sqrt(tf.reduce_sum(tf.square(real_edges), axis=-1))  # [B,H,W,1]
    fake_mag = tf.sqrt(tf.reduce_sum(tf.square(fake_edges), axis=-1))
    return tf.cast(weight, tf.float32) * tf.reduce_mean(tf.abs(real_mag - fake_mag))

def perceptual_loss(real, fake, weight=0.2, perceptual_size=224):
    """
    MobileNetV2 expects inputs preprocessed with mobilenet_v2.preprocess_input (scales to [-1,1]).
    Our tensors are in [-1,1], but we convert to [0,255] then preprocess to follow the documented API.
    """
    if weight == 0.0:
        return tf.constant(0.0, tf.float32)

    real = tf.cast(real, tf.float32)
    fake = tf.cast(fake, tf.float32)

    # Resize to perceptual backbone size.
    if perceptual_size != int(real.shape[1]) or perceptual_size != int(real.shape[2]):
        real = tf.image.resize(real, [perceptual_size, perceptual_size], method="bilinear")
        fake = tf.image.resize(fake, [perceptual_size, perceptual_size], method="bilinear")

    real3 = tf.image.grayscale_to_rgb((real + 1.0) * 127.5)  # [0,255]
    fake3 = tf.image.grayscale_to_rgb((fake + 1.0) * 127.5)

    real3 = _MOBILENETV2_PREPROCESS(real3)  # [-1,1]
    fake3 = _MOBILENETV2_PREPROCESS(fake3)

    real_feat = tf.cast(_PERCEPTUAL_MODEL(real3, training=False), tf.float32)
    fake_feat = tf.cast(_PERCEPTUAL_MODEL(fake3, training=False), tf.float32)

    return tf.cast(weight, tf.float32) * tf.reduce_mean(tf.abs(real_feat - fake_feat))

# =========================
# Preprocessing (CLAHE via numpy_function)
# =========================
def apply_clahe_tf(img_tensor):
    def clahe_np(x):
        arr = ((x + 1.0) * 127.5).astype(np.uint8)  # [0,255]
        arr = arr[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = clahe.apply(arr)
        out = out[:, :, None].astype(np.float32)
        out = (out / 127.5) - 1.0
        return out

    out = tf.numpy_function(clahe_np, [img_tensor], tf.float32)
    out.set_shape(img_tensor.shape)
    return out

def decode_png(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.resize(img, [img_size, img_size], method="bilinear")
    img = tf.cast(img, tf.float32)
    return normalize_img(img)

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    return img

def make_domain_dataset(paths, img_size, batch_size, training=True, is_labour=False):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        img = decode_png(path, img_size)
        if is_labour:
            img = apply_clahe_tf(img)
        if training:
            img = augment(img)
        return img

    if training:
        ds = ds.shuffle(min(2048, max(128, len(paths))), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def make_test_dataset(test_pairs, img_size, batch_size):
    labour_paths = [p[0] for p in test_pairs]
    real_paths = [p[1] for p in test_pairs]
    keys = [p[2] for p in test_pairs]

    ds = tf.data.Dataset.from_tensor_slices((labour_paths, real_paths, keys))

    def _load(lab_path, real_path, key):
        lab = decode_png(lab_path, img_size)
        lab = apply_clahe_tf(lab)
        real = decode_png(real_path, img_size)
        return lab, real, key

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# =========================
# LR schedule
# =========================
def get_lr(epoch, total_epochs, initial_lr=2e-4):
    if epoch < total_epochs // 2:
        return float(initial_lr)
    decay_epochs = total_epochs - (total_epochs // 2)
    progress = (epoch - total_epochs // 2) / max(1, decay_epochs)
    return float(initial_lr * max(0.0, 1.0 - progress))

# =========================
# Trainer
# =========================
class CycleGANTrainer:
    def __init__(
        self,
        img_size=224,
        batch_size=4,
        initial_lr=2e-4,
        lambda_cycle=10.0,
        lambda_ssim=2.0,
        lambda_identity=5.0,
        freq_weight=0.0005,
        edge_weight=1.0,
        orient_weight=1.0,
        ridge_weight=1.5,
        perc_weight=0.2,
        perceptual_size=224,
        clip_norm=5.0,
    ):
        self.img_size = int(img_size)
        self.perceptual_size = int(perceptual_size)
        self.clip_norm = float(clip_norm)

        self.G_L2N = build_generator(img_size=self.img_size, name="G_L2N")
        self.G_N2L = build_generator(img_size=self.img_size, name="G_N2L")
        self.D_Normal = build_discriminator(img_size=self.img_size, name="D_Normal")
        self.D_Labour = build_discriminator(img_size=self.img_size, name="D_Labour")

        # Shared LR variable so schedule updates all optimisers cleanly.
        self.lr_var = tf.Variable(initial_lr, dtype=tf.float32, trainable=False, name="lr")

        base_g1 = tf.keras.optimizers.Adam(learning_rate=self.lr_var, beta_1=0.5)
        base_g2 = tf.keras.optimizers.Adam(learning_rate=self.lr_var, beta_1=0.5)
        base_dn = tf.keras.optimizers.Adam(learning_rate=self.lr_var, beta_1=0.5)
        base_dl = tf.keras.optimizers.Adam(learning_rate=self.lr_var, beta_1=0.5)

        self.g1_opt = mixed_precision.LossScaleOptimizer(base_g1)
        self.g2_opt = mixed_precision.LossScaleOptimizer(base_g2)
        self.dn_opt = mixed_precision.LossScaleOptimizer(base_dn)
        self.dl_opt = mixed_precision.LossScaleOptimizer(base_dl)

        self.fake_normal_buffer = ImageBuffer(50)
        self.fake_labour_buffer = ImageBuffer(50)

        self.lambda_cycle = float(lambda_cycle)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_identity = float(lambda_identity)
        self.freq_weight = float(freq_weight)

        self.edge_weight = float(edge_weight)
        self.orient_weight = float(orient_weight)
        self.ridge_weight = float(ridge_weight)
        self.perc_weight = float(perc_weight)

    def set_lr(self, lr: float):
        self.lr_var.assign(float(lr))

    @tf.function
    def train_generators(self, real_labour, real_normal):
        with tf.GradientTape(persistent=True) as tape:
            fake_normal = self.G_L2N(real_labour, training=True)
            cycled_labour = self.G_N2L(fake_normal, training=True)

            fake_labour = self.G_N2L(real_normal, training=True)
            cycled_normal = self.G_L2N(fake_labour, training=True)

            same_normal = self.G_L2N(real_normal, training=True)
            same_labour = self.G_N2L(real_labour, training=True)

            d_fake_normal = self.D_Normal(fake_normal, training=True)
            d_fake_labour = self.D_Labour(fake_labour, training=True)

            adv_l2n = generator_adv_loss(d_fake_normal)
            adv_n2l = generator_adv_loss(d_fake_labour)

            # Split cycle by domain
            cyc_lab = cycle_loss(real_labour, cycled_labour, self.lambda_cycle, self.lambda_ssim)
            cyc_norm = cycle_loss(real_normal, cycled_normal, self.lambda_cycle, self.lambda_ssim)

            id_l2n = identity_loss(real_normal, same_normal, self.lambda_identity)
            id_n2l = identity_loss(real_labour, same_labour, self.lambda_identity)

            # Structural preservation losses are input ↔ output (unpaired-safe).
            freq_l2n = frequency_loss(real_labour, fake_normal, self.freq_weight)
            freq_n2l = frequency_loss(real_normal, fake_labour, self.freq_weight)

            edge_l2n = edge_loss(real_labour, fake_normal, self.edge_weight)
            edge_n2l = edge_loss(real_normal, fake_labour, self.edge_weight)

            orient_l2n = orientation_loss(real_labour, fake_normal, self.orient_weight)
            orient_n2l = orientation_loss(real_normal, fake_labour, self.orient_weight)

            ridge_l2n = ridge_loss(real_labour, fake_normal, self.ridge_weight)
            ridge_n2l = ridge_loss(real_normal, fake_labour, self.ridge_weight)

            perc_l2n = perceptual_loss(real_labour, fake_normal, self.perc_weight, self.perceptual_size)
            perc_n2l = perceptual_loss(real_normal, fake_labour, self.perc_weight, self.perceptual_size)

            total_g1 = adv_l2n + cyc_lab + id_l2n + 0.5 * freq_l2n + edge_l2n + orient_l2n + ridge_l2n + perc_l2n
            total_g2 = adv_n2l + cyc_norm + id_n2l + 0.5 * freq_n2l + edge_n2l + orient_n2l + ridge_n2l + perc_n2l

            scaled_g1 = self.g1_opt.get_scaled_loss(total_g1)
            scaled_g2 = self.g2_opt.get_scaled_loss(total_g2)

        g1_grads = tape.gradient(scaled_g1, self.G_L2N.trainable_variables)
        g2_grads = tape.gradient(scaled_g2, self.G_N2L.trainable_variables)

        g1_grads = self.g1_opt.get_unscaled_gradients(g1_grads)
        g2_grads = self.g2_opt.get_unscaled_gradients(g2_grads)

        g1_grads = [tf.clip_by_norm(g, self.clip_norm) if g is not None else None for g in g1_grads]
        g2_grads = [tf.clip_by_norm(g, self.clip_norm) if g is not None else None for g in g2_grads]

        self.g1_opt.apply_gradients(zip(g1_grads, self.G_L2N.trainable_variables))
        self.g2_opt.apply_gradients(zip(g2_grads, self.G_N2L.trainable_variables))

        del tape
        return fake_normal, fake_labour, total_g1, total_g2

    @tf.function
    def train_discriminator_normal(self, real_normal, fake_normal):
        with tf.GradientTape() as tape:
            d_real = self.D_Normal(real_normal, training=True)
            d_fake = self.D_Normal(fake_normal, training=True)
            loss = discriminator_loss(d_real, d_fake)
            scaled = self.dn_opt.get_scaled_loss(loss)

        grads = tape.gradient(scaled, self.D_Normal.trainable_variables)
        grads = self.dn_opt.get_unscaled_gradients(grads)
        grads = [tf.clip_by_norm(g, self.clip_norm) if g is not None else None for g in grads]
        self.dn_opt.apply_gradients(zip(grads, self.D_Normal.trainable_variables))
        return loss

    @tf.function
    def train_discriminator_labour(self, real_labour, fake_labour):
        with tf.GradientTape() as tape:
            d_real = self.D_Labour(real_labour, training=True)
            d_fake = self.D_Labour(fake_labour, training=True)
            loss = discriminator_loss(d_real, d_fake)
            scaled = self.dl_opt.get_scaled_loss(loss)

        grads = tape.gradient(scaled, self.D_Labour.trainable_variables)
        grads = self.dl_opt.get_unscaled_gradients(grads)
        grads = [tf.clip_by_norm(g, self.clip_norm) if g is not None else None for g in grads]
        self.dl_opt.apply_gradients(zip(grads, self.D_Labour.trainable_variables))
        return loss

# =========================
# Evaluation
# =========================
def evaluate_model(model, test_ds):
    psnr_vals, ssim_vals = [], []
    for lab, real, _keys in test_ds:
        pred = model(lab, training=False)
        pred_01 = tf.cast(denormalize_img(pred), tf.float32)
        real_01 = tf.cast(denormalize_img(real), tf.float32)
        psnr = tf.image.psnr(real_01, pred_01, max_val=1.0)
        ssim = tf.image.ssim(real_01, pred_01, max_val=1.0)
        psnr_vals.extend(psnr.numpy().tolist())
        ssim_vals.extend(ssim.numpy().tolist())
    return {
        "avg_psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "avg_ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "n_eval_images": len(psnr_vals),
    }

def save_epoch_visuals(model, test_ds, out_dir, epoch, max_batches=2):
    epoch_dir = os.path.join(out_dir, "epoch_visuals", f"epoch_{epoch:03d}")
    ensure_dir(epoch_dir)

    saved = 0
    for batch_idx, (lab, real, keys) in enumerate(test_ds):
        if batch_idx >= max_batches:
            break

        pred = model(lab, training=False)

        pred_01 = tf.cast(denormalize_img(pred), tf.float32)
        real_01 = tf.cast(denormalize_img(real), tf.float32)
        lab_01 = tf.cast(denormalize_img(lab), tf.float32)

        psnr = tf.image.psnr(real_01, pred_01, max_val=1.0)
        ssim = tf.image.ssim(real_01, pred_01, max_val=1.0)

        for i in range(lab.shape[0]):
            key = keys[i].numpy().decode("utf-8")

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(lab_01[i, :, :, 0], cmap="gray")
            axes[0].set_title("Labour")
            axes[0].axis("off")

            axes[1].imshow(pred_01[i, :, :, 0], cmap="gray")
            axes[1].set_title("Enhanced")
            axes[1].axis("off")

            axes[2].imshow(real_01[i, :, :, 0], cmap="gray")
            axes[2].set_title("Real")
            axes[2].axis("off")

            fig.suptitle(
                f"Epoch {epoch:03d} | {key} | PSNR={float(psnr[i]):.2f} dB | SSIM={float(ssim[i]):.4f}"
            )
            plt.tight_layout()
            save_path = os.path.join(epoch_dir, f"{saved:04d}_{key}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            saved += 1

def evaluate_and_save_final(model, test_ds, out_dir, max_batches=10):
    psnr_vals, ssim_vals = [], []
    vis_dir = os.path.join(out_dir, "test_results_final")
    ensure_dir(vis_dir)

    saved = 0
    for batch_idx, (lab, real, keys) in enumerate(test_ds):
        pred = model(lab, training=False)

        pred_01 = tf.cast(denormalize_img(pred), tf.float32)
        real_01 = tf.cast(denormalize_img(real), tf.float32)
        lab_01 = tf.cast(denormalize_img(lab), tf.float32)

        psnr = tf.image.psnr(real_01, pred_01, max_val=1.0)
        ssim = tf.image.ssim(real_01, pred_01, max_val=1.0)

        psnr_vals.extend(psnr.numpy().tolist())
        ssim_vals.extend(ssim.numpy().tolist())

        if batch_idx < max_batches:
            for i in range(lab.shape[0]):
                key = keys[i].numpy().decode("utf-8")

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(lab_01[i, :, :, 0], cmap="gray")
                axes[0].set_title("Labour")
                axes[0].axis("off")

                axes[1].imshow(pred_01[i, :, :, 0], cmap="gray")
                axes[1].set_title("Enhanced")
                axes[1].axis("off")

                axes[2].imshow(real_01[i, :, :, 0], cmap="gray")
                axes[2].set_title("Real")
                axes[2].axis("off")

                fig.suptitle(f"{key} | PSNR={float(psnr[i]):.2f} dB | SSIM={float(ssim[i]):.4f}")
                plt.tight_layout()
                save_path = os.path.join(vis_dir, f"{saved:04d}_{key}.png")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                saved += 1

    metrics = {
        "avg_psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "avg_ssim": float(np.mean(ssim_vals)) if ssim_vals else None,
        "n_eval_images": len(psnr_vals),
    }
    save_json(os.path.join(out_dir, "eval_metrics_final.json"), metrics)
    return metrics

# =========================
# Plotting
# =========================
def save_training_curves(history, out_dir):
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history["g_l2n"], label="G_L2N")
    plt.plot(history["g_n2l"], label="G_N2L")
    plt.title("Generator Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history["d_normal"], label="D_Normal")
    plt.plot(history["d_labour"], label="D_Labour")
    plt.title("Discriminator Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history["psnr"], label="PSNR")
    plt.title("PSNR")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history["ssim"], label="SSIM")
    plt.title("SSIM")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()

# =========================
# Main
# =========================
def main(args):
    ensure_dir(args.output_dir)
    ensure_dir(args.checkpoint_dir)

    labour_files = list_pngs(args.labour_dir)
    real_files = list_pngs(args.real_dir)

    if not labour_files:
        raise FileNotFoundError(f"No PNGs found in labour dir: {args.labour_dir}")
    if not real_files:
        raise FileNotFoundError(f"No PNGs found in real dir: {args.real_dir}")

    labour_train, real_train, test_pairs, stats = make_splits(
        labour_files, real_files, test_ratio=args.test_ratio
    )

    print("\nDataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    save_json(os.path.join(args.output_dir, "dataset_stats.json"), stats)

    labour_train_ds = make_domain_dataset(
        labour_train, args.img_size, args.batch_size, training=True, is_labour=True
    )
    real_train_ds = make_domain_dataset(
        real_train, args.img_size, args.batch_size, training=True, is_labour=False
    )
    test_ds = make_test_dataset(test_pairs, args.img_size, args.batch_size)

    trainer = CycleGANTrainer(
        img_size=args.img_size,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        lambda_cycle=args.lambda_cycle,
        lambda_ssim=args.lambda_ssim,
        lambda_identity=args.lambda_identity,
        freq_weight=args.freq_weight,
        edge_weight=args.edge_weight,
        orient_weight=args.orient_weight,
        ridge_weight=args.ridge_weight,
        perc_weight=args.perc_weight,
        perceptual_size=args.perceptual_size,
        clip_norm=args.clip_norm,
    )

    ckpt = tf.train.Checkpoint(
        G_L2N=trainer.G_L2N,
        G_N2L=trainer.G_N2L,
        D_Normal=trainer.D_Normal,
        D_Labour=trainer.D_Labour,
        g1_opt=trainer.g1_opt,
        g2_opt=trainer.g2_opt,
        dn_opt=trainer.dn_opt,
        dl_opt=trainer.dl_opt,
        lr=trainer.lr_var,
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.checkpoint_dir, max_to_keep=3)

    initial_epoch = 0
    if ckpt_manager.latest_checkpoint:
        status = ckpt.restore(ckpt_manager.latest_checkpoint)
        status.expect_partial()
        initial_epoch = get_initial_epoch_from_checkpoint(ckpt_manager, args.ckpt_every)
        print(f"\nResumed from checkpoint: {ckpt_manager.latest_checkpoint}")
        print(f"Continuing from epoch: {initial_epoch + 1}")
    else:
        print("\nStarting fresh training")

    history = {
        "g_l2n": [],
        "g_n2l": [],
        "d_normal": [],
        "d_labour": [],
        "psnr": [],
        "ssim": [],
    }

    history_json_path = os.path.join(args.output_dir, "history.json")
    history_csv_path = os.path.join(args.output_dir, "history.csv")
    epoch_metrics_csv = os.path.join(args.output_dir, "epoch_metrics.csv")

    if initial_epoch > 0 and os.path.exists(history_json_path):
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for k in history.keys():
                if k in loaded and isinstance(loaded[k], list):
                    history[k] = loaded[k]
        except Exception as e:
            print(f"Warning: could not reload previous history.json: {e}")

    print(f"\nTraining until epoch {args.epochs}...")
    for epoch in range(initial_epoch, args.epochs):
        epoch_num = epoch + 1

        new_lr = get_lr(epoch, args.epochs, args.lr)
        trainer.set_lr(new_lr)

        g1_losses, g2_losses, dn_losses, dl_losses = [], [], [], []

        for real_labour, real_normal in tf.data.Dataset.zip((labour_train_ds, real_train_ds)):
            fake_normal, fake_labour, g1, g2 = trainer.train_generators(real_labour, real_normal)

            fake_normal_buf = trainer.fake_normal_buffer.query(fake_normal)
            fake_labour_buf = trainer.fake_labour_buffer.query(fake_labour)

            dn = trainer.train_discriminator_normal(real_normal, fake_normal_buf)
            dl = trainer.train_discriminator_labour(real_labour, fake_labour_buf)

            g1_losses.append(float(g1.numpy()))
            g2_losses.append(float(g2.numpy()))
            dn_losses.append(float(dn.numpy()))
            dl_losses.append(float(dl.numpy()))

        avg_g1 = float(np.mean(g1_losses))
        avg_g2 = float(np.mean(g2_losses))
        avg_dn = float(np.mean(dn_losses))
        avg_dl = float(np.mean(dl_losses))

        metrics = evaluate_model(trainer.G_L2N, test_ds)
        avg_psnr = metrics["avg_psnr"]
        avg_ssim = metrics["avg_ssim"]

        history["g_l2n"].append(avg_g1)
        history["g_n2l"].append(avg_g2)
        history["d_normal"].append(avg_dn)
        history["d_labour"].append(avg_dl)
        history["psnr"].append(avg_psnr)
        history["ssim"].append(avg_ssim)

        print(
            f"Epoch {epoch_num:03d}/{args.epochs} | "
            f"LR {new_lr:.2e} | "
            f"G_L2N {avg_g1:.4f} | G_N2L {avg_g2:.4f} | "
            f"D_Normal {avg_dn:.4f} | D_Labour {avg_dl:.4f} | "
            f"PSNR {avg_psnr:.4f} | SSIM {avg_ssim:.4f}"
        )

        epoch_row = {
            "epoch": epoch_num,
            "lr": new_lr,
            "g_l2n": avg_g1,
            "g_n2l": avg_g2,
            "d_normal": avg_dn,
            "d_labour": avg_dl,
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "n_eval_images": metrics["n_eval_images"],
        }
        append_epoch_metrics_csv(epoch_metrics_csv, epoch_row)

        save_json(history_json_path, history)
        save_history_csv(history_csv_path, history)
        save_training_curves(history, args.output_dir)

        if (epoch_num % args.visual_every) == 0:
            save_epoch_visuals(
                trainer.G_L2N,
                test_ds,
                args.output_dir,
                epoch=epoch_num,
                max_batches=args.max_visual_batches,
            )

        if (epoch_num % args.ckpt_every) == 0:
            save_path = ckpt_manager.save()
            print(f"  Saved checkpoint: {save_path}")

    trainer.G_L2N.save(os.path.join(args.output_dir, "G_L2N.keras"))
    trainer.G_N2L.save(os.path.join(args.output_dir, "G_N2L.keras"))

    final_metrics = evaluate_and_save_final(
        trainer.G_L2N,
        test_ds,
        args.output_dir,
        max_batches=args.max_final_visual_batches,
    )

    print("\nFinal Evaluation:")
    print(json.dumps(final_metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labour_dir", type=str, required=True)
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=r"./outputs")
    parser.add_argument("--checkpoint_dir", type=str, default=r"./ckpts")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_ssim", type=float, default=2.0)
    parser.add_argument("--lambda_identity", type=float, default=5.0)

    parser.add_argument("--freq_weight", type=float, default=0.0005)
    parser.add_argument("--edge_weight", type=float, default=1.0)
    parser.add_argument("--orient_weight", type=float, default=1.0)
    parser.add_argument("--ridge_weight", type=float, default=1.5)
    parser.add_argument("--perc_weight", type=float, default=0.2)

    parser.add_argument("--perceptual_size", type=int, default=224)
    parser.add_argument("--clip_norm", type=float, default=5.0)

    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--visual_every", type=int, default=5)
    parser.add_argument("--max_visual_batches", type=int, default=2)
    parser.add_argument("--max_final_visual_batches", type=int, default=10)

    args = parser.parse_args()
    main(args)