"""Does the U-Net help the classifier?  Compares MelaNet on the original image with versions that use the U-Net mask:
  original | blackout (background set to black) | crop (square crop around the lesion) | fine-tuned crop (if weights/MelaNet_crop.keras exists)

  python evaluate_pipeline.py --images data_cls/images --labels data_cls/labels.csv [--threshold 0.5]

labels.csv: one row per image, columns  image,label   (a header row is optional, so ISIC's own ground-truth CSV works).
  image = file name, extension optional (ISIC_0000003 or ISIC_0000003.jpg)
  label = benign / malignant  (also accepted: 0 / 1, 0.0 / 1.0)
Writes metrics/cls_metrics.json (shown on the app's Evaluation tab) and results/cls_per_image.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata

import classifier
from app import analyse, clf, clf_ft

EXT = ["", ".jpg", ".jpeg", ".png"]
LABELS = {"1": 1, "1.0": 1, "malignant": 1, "melanoma": 1, "mel": 1,
          "0": 0, "0.0": 0, "benign": 0, "nevus": 0, "nv": 0}


def read_labels(path):
    df = pd.read_csv(path)
    if not {"image", "label"} <= set(df.columns):     # e.g. ISIC ground-truth files have no header row
        df = pd.read_csv(path, header=None, names=["image", "label"], usecols=[0, 1])
    df["image"] = df.image.astype(str).str.strip()
    df["label"] = df.label.astype(str).str.strip().str.lower().map(LABELS)
    assert df.label.notna().all(), f"unknown label values; accepted: {sorted(LABELS)}"
    return df


def find_image(folder, name):
    path = next((p for e in EXT if (p := Path(folder) / f"{name}{e}").is_file()), None)
    assert path, f"image not found in {folder}: {name}"
    return path


def auroc(y, p):
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)   # exact, via the rank statistic


def metrics(y, p, t):
    pred = p >= t
    tp, fn = (pred & (y == 1)).sum(), (~pred & (y == 1)).sum()
    tn, fp = (~pred & (y == 0)).sum(), (pred & (y == 0)).sum()
    return {"AUROC": auroc(y, p), "Sensitivity": tp / max(tp + fn, 1), "Specificity": tn / max(tn + fp, 1),
            "Accuracy": (tp + tn) / len(y)}


def bootstrap(y, pa, pb, n=1000, seed=0):
    """95% intervals for each AUROC and for the paired difference (b - a): resample images, keep pairs together."""
    rng, A, B = np.random.default_rng(seed), [], []
    while len(A) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            A.append(auroc(y[i], pa[i])); B.append(auroc(y[i], pb[i]))
    ci = lambda v: f"{np.percentile(v, 2.5):.3f} to {np.percentile(v, 97.5):.3f}"
    return ci(A), ci(B), np.mean(np.array(B) - np.array(A)), ci(np.array(B) - np.array(A))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    if clf is None:
        raise SystemExit("Classifier not loaded (see the message printed above).")

    df = read_labels(a.labels)
    rows = []
    for i, r in df.iterrows():
        img = np.array(Image.open(find_image(a.images, r.image)).convert("RGB"))
        out = analyse(img, use_mask=False)                   # U-Net mask + MelaNet on the original image
        pil, m = Image.fromarray(img), out["mask"]
        crop = classifier.crop_lesion(pil, m)
        row = {"image": r.image, "label": int(r.label), "p_original": out["p_orig"],
               "p_masked": clf(classifier.mask_lesion(pil, m)), "p_crop": clf(crop)}
        if clf_ft:
            row["p_crop_finetuned"] = clf_ft(crop)
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(df)}")

    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    Path("metrics").mkdir(exist_ok=True)
    res.to_csv("results/cls_per_image.csv", index=False)
    y, po = res.label.values, res.p_original.values
    variants = [("original", "p_original"), ("U-Net blackout", "p_masked"), ("U-Net crop", "p_crop")]
    if clf_ft:
        variants.append(("U-Net crop, fine-tuned", "p_crop_finetuned"))

    out = {"Images": len(res), "Malignant": int(y.sum()), "Threshold": a.threshold}
    for name, col in variants:
        out.update({f"{k} ({name})": round(float(v), 4) for k, v in metrics(y, res[col].values, a.threshold).items()})
        out[f"AUROC 95% CI ({name})"] = bootstrap(y, po, res[col].values)[1 if name != "original" else 0]
    summary = []
    for name, col in variants[1:]:                           # paired comparison with the original image
        d, ci = bootstrap(y, po, res[col].values)[2:]
        out[f"AUROC change vs original: {name}"] = f"{d:+.3f} (95% CI {ci})"
        summary.append((name, d, ci))
    if clf_ft:                                               # what fine-tuning adds on top of cropping
        d, ci = bootstrap(y, res.p_crop.values, res.p_crop_finetuned.values)[2:]
        out["AUROC change from fine-tuning (crop -> fine-tuned crop)"] = f"{d:+.3f} (95% CI {ci})"
    Path("metrics/cls_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nA variant helps only if its 'AUROC change' interval lies entirely above 0:")
    for name, d, ci in summary:
        print(f"  {name:24s} {d:+.3f}  (95% CI {ci})")
