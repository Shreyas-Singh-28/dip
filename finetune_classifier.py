"""Fine-tune MelaNet on U-Net lesion crops (ISIC 2016 training set) -> weights/MelaNet_crop.keras

  python finetune_classifier.py --images data_cls/train/images --labels data_cls/train/labels.csv

Why it is fast: the U-Net crops and the frozen VGG16 layers (blocks 1-4) are computed ONCE and cached in cache/;
training only touches the last conv block and the classifier head, on small cached feature maps.
  stage 1: retrain the head on the crop features (seconds)
  stage 2: also fine-tune the last conv block, low learning rate (minutes)   [skip with --head-only]
Then: python evaluate_pipeline.py --images data_cls/test/images --labels data_cls/test/labels.csv   (the app picks up the new model by itself)

Use only the TRAINING images here. Never tune anything on the test set.
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

import classifier
from app import analyse
from evaluate_pipeline import find_image, read_labels

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--head-only", action="store_true", help="skip stage 2 (fine-tuning the last conv block)")
ap.add_argument("--epochs", type=int, default=8, help="max epochs of stage 2")
a = ap.parse_args()
tf.keras.utils.set_random_seed(0)

# ---- split MelaNet at block4_pool: frozen bottom (run once) + trainable top -------------------------------------------
base = tf.keras.models.load_model(classifier.base_path(), compile=False)
cut = [l.name for l in base.layers].index("block4_pool")
bottom = tf.keras.Model(base.input, base.layers[cut].output)
top_in = tf.keras.Input(base.layers[cut].output.shape[1:])
x = top_in
for layer in base.layers[cut + 1:-1]:           # block5 convs, pooling, global average pooling
    x = layer(x)
head = base.layers[-1]                          # Dense(2, softmax), warm-started from MelaNet
features = tf.keras.Model(top_in, x)            # -> 512 numbers per image
top = tf.keras.Model(top_in, head(x))           # -> [benign, malignant]  (shares weights with `base`)

# ---- 1. U-Net crops -> block4 feature maps, cached ------------------------------------------------------------------
df = read_labels(a.labels)
cache = Path("cache") / "block4_train.npz"
if cache.exists():
    d = np.load(cache)
    X, y, group = d["X"], d["y"], d["group"]
    print(f"loaded cached features: {cache}")
else:
    X, y, group, buf = [], [], [], []

    def flush():
        if buf:
            X.extend(bottom([np.concatenate(buf)], training=False).numpy().astype("float16"))
            buf.clear()

    for i, r in df.iterrows():
        img = Image.fromarray(np.array(Image.open(find_image(a.images, r.image)).convert("RGB")))
        mask = analyse(np.array(img), classify=False)["mask"]          # U-Net only
        xin = classifier.preprocess(classifier.crop_lesion(img, mask))
        for v in (xin, np.ascontiguousarray(xin[:, :, ::-1])):          # original + horizontal flip
            buf.append(v)
            y.append(int(r.label)); group.append(i)
        if len(buf) >= 32:
            flush()
        if (i + 1) % 100 == 0:
            print(f"  crops/features: {i + 1}/{len(df)}")
    flush()
    X, y, group = np.stack(X), np.array(y), np.array(group)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, X=X, y=y, group=group)

# ---- 2. split by IMAGE (a flip never ends up on the other side), class weights --------------------------------------
tr_img, va_img = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df.label.values, random_state=0)
tr, va = np.isin(group, tr_img), np.isin(group, va_img)
cw = {c: tr.sum() / (2 * (y[tr] == c).sum()) for c in (0, 1)}
stop = lambda p: tf.keras.callbacks.EarlyStopping("val_loss", patience=p, restore_best_weights=True)


def batches(model_x, mask, shuffle):
    d = tf.data.Dataset.from_tensor_slices((model_x[mask], y[mask]))
    d = d.shuffle(int(mask.sum()), seed=0) if shuffle else d
    return d.batch(32).map(lambda f, t: (tf.cast(f, "float32"), t)).prefetch(2)


def train(model, inputs, lr, epochs, patience):
    model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(batches(inputs, tr, True), validation_data=batches(inputs, va, False), epochs=epochs,
              class_weight=cw, callbacks=[stop(patience)], verbose=2)


# ---- 3. stage 1: head only, on cached 512-d features -----------------------------------------------------------------
for layer in top.layers[1:-1]:
    layer.trainable = False
F = np.concatenate([features(X[i:i + 64].astype("float32"), training=False).numpy() for i in range(0, len(X), 64)])
print("\nstage 1: retraining the classifier head")
train(tf.keras.Sequential([tf.keras.Input((F.shape[1],)), head]), F, 1e-3, 40, 6)

# ---- 4. stage 2: also the last conv block ---------------------------------------------------------------------------
if not a.head_only:
    for layer in top.layers[1:-1]:
        layer.trainable = True
    print("\nstage 2: fine-tuning the last conv block (low learning rate)")
    train(top, X, 1e-5, a.epochs, 2)

classifier.FINETUNED.parent.mkdir(exist_ok=True)
base.save(classifier.FINETUNED)                  # `base` shares its weights with `top`/`head`
print(f"\nsaved {classifier.FINETUNED}  (the app and evaluate_pipeline.py use it automatically)")
