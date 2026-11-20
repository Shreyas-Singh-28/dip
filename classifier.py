"""MelaNet melanoma classifier (VGG16, trained on ISIC 2016 Task 3).

Code: https://github.com/hasibzunair/adversarial-lesions (MIT)  |  weights: MelaNet.h5, Apache-2.0 on Hugging Face.
Research use only. Preprocessing follows the authors' reference demo (Hugging Face Space 'melanoma-detection-demo').

Weights, in order of preference:
  weights/MelaNet.keras  slim copy (59 MB, no training state, identical predictions)  <- use this if you have it
  weights/MelaNet.h5     the authors' original (177 MB); downloaded automatically if neither file exists.
                         The download shows progress and RESUMES if interrupted: just start the app again.
"""
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

URL = "https://github.com/hasibzunair/adversarial-lesions/releases/latest/download/MelaNet.h5"
SLIM, FULL = Path("weights/MelaNet.keras"), Path("weights/MelaNet.h5")
FINETUNED = Path("weights/MelaNet_crop.keras")   # made by finetune_classifier.py (MelaNet fine-tuned on U-Net lesion crops)
SIZE = 256
RED_ONLY = True   # the reference demo feeds the red channel replicated 3x; set False to feed true RGB


def download(url, dest, tries=20):
    """Download with a progress display; continues from the .part file after an interruption."""
    dest.parent.mkdir(exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        done = part.stat().st_size if part.exists() else 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
            with urllib.request.urlopen(req, timeout=30) as r:
                resumed = r.status == 206                    # server honoured the Range request
                total = int(r.headers.get("Content-Length", 0)) + (done if resumed else 0)
                done = done if resumed else 0
                with open(part, "ab" if resumed else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r[classifier] downloading {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
            print()
            part.rename(dest)
            return
        except OSError as e:                                 # timeout, dropped connection ...
            print(f"\n[classifier] interrupted ({e}); retrying {attempt + 1}/{tries} ...")
    raise RuntimeError(f"download failed; fetch {url} in a browser and save it as {dest}")


def base_path():
    """Path of the original MelaNet weights (downloaded if missing)."""
    if not SLIM.exists() and not FULL.exists():
        download(URL, FULL)
    return SLIM if SLIM.exists() else FULL


def preprocess(pil_img):
    """PIL image -> (1, 256, 256, 3) float32 array, exactly as in the authors' demo."""
    x = np.asarray(pil_img.convert("RGB"), dtype="float32") / 255.0
    if RED_ONLY:
        x = np.repeat(x[..., :1], 3, axis=-1)
    return tf.image.resize(x, (SIZE, SIZE)).numpy()[None]


def crop_lesion(pil_img, mask, margin=0.25):
    """Square crop of the ORIGINAL image around the U-Net lesion mask (bool, any size), with `margin` x lesion size on each side."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return pil_img                                        # no lesion found: use the whole image
    W, H = pil_img.size
    x0, x1 = xs.min() * W / mask.shape[1], (xs.max() + 1) * W / mask.shape[1]
    y0, y1 = ys.min() * H / mask.shape[0], (ys.max() + 1) * H / mask.shape[0]
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2 * (1 + 2 * margin)
    return pil_img.crop((int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, W)), int(min(cy + half, H))))


def mask_lesion(pil_img, mask):
    """Original image with everything outside the U-Net lesion mask set to black."""
    big = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(pil_img.size, Image.NEAREST)) > 0
    return Image.fromarray(np.asarray(pil_img.convert("RGB")) * big[..., None])


class Classifier:
    """Callable: PIL image -> probability of malignancy. Default: the original MelaNet; or pass a model path."""

    def __init__(self, path=None):
        self.net = tf.keras.models.load_model(path or base_path(), compile=False)

    def __call__(self, pil_img):
        return float(self.net([preprocess(pil_img)], training=False).numpy()[0, 1])   # softmax output: [benign, malignant]
