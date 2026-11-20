# Akriti

### Empirical Study of Compound Loss Optimization for ISIC 2016 Skin Lesion Segmentation & Real-Time Inference Dashboard

[![GitHub Repository](https://img.shields.io/badge/GitHub-Shreyas--Singh--28%2Fdip-blue?logo=github)](https://github.com/Shreyas-Singh-28/dip)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC%202016%20Challenge-green)](https://challenge.isic-archive.com/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Gradio-orange)](https://www.gradio.app/)

Welcome to **Akriti**, a research project and interactive deployment pipeline investigating **compound pixel-mask loss optimization** for automated skin lesion segmentation on the **ISIC 2016** dataset. 

This work systematically benchmarks standalone loss functions against weighted composite losses (combining **Lovász-Hinge**, **Boundary-Weighted Binary Cross-Entropy (BW-BCE)**, and **Focal Loss**) using a lightweight U-Net architecture.

---

## Key Research Findings

1. **Two-Loss Compound Objective Outperforms Standalone & Three-Loss Setups:**
   Across controlled 30-epoch, multi-seed experiments (seeds `42`, `123`, `2024`), a two-loss mixture of **Lovász-Hinge + Boundary-Weighted BCE** achieved the highest overall validation performance (**0.8851 Dice**, **0.8081 IoU**, **0.5862 Boundary F1**, and **13.49 px HD95**) and the strongest unseen test performance (**0.8813 Dice**, **0.8054 IoU**, **12.85 px HD95** across 379 test pairs).
2. **Adding Focal Loss Did Not Yield Further Gains:**
   A one-step central finite-difference search in softmax-logit space selected optimal three-loss weights of **45.2% Lovász-Hinge, 27.4% BW-BCE, and 27.4% Focal**. While this mixture outperformed every individual single-loss baseline (BCE, Focal, Lovász-Hinge, BW-BCE), it scored **0.8781 test Dice**—trailing the simpler two-loss mixture by **0.32 percentage points**.

---

## Experimental Methodology & Architecture

### Dataset & Reproducible Splits
* **Training / Validation Set:** 900 matched image-mask pairs from the ISIC 2016 training partition, deterministically shuffled (`split seed = 42`) into **720 training** and **180 validation** samples.
* **Official Test Set:** 379 held-out image-mask pairs (`official_test_data_used=true`, verified zero ID overlap with training/validation manifests, never used for hyperparameter or threshold tuning).
* **Preprocessing:** RGB images resized to $128 \times 128$ via bilinear interpolation and normalized to $[0, 1]$; binary ground-truth masks resized via nearest-neighbor interpolation and thresholded at $>127$.

### Model & Optimization Setup
* **Architecture:** Lightweight U-Net featuring two $3\times3$ ReLU convolutions per block, encoder channel widths `[16, 32, 64]`, a `128`-channel bottleneck, symmetric decoder `[64, 32, 16]` with skip connections, and a single sigmoid output channel.
* **Inner Optimizer:** Adam (`lr = 1e-4`, `batch_size = 4`, gradient clipping `clipnorm = 1.0`) trained for 30 epochs ($5,400$ gradient updates per seed).
* **Outer Loss-Weight Search:** Central finite-difference evaluation over 6 candidate weight configurations in logit space minimizing $J = 1 - \text{Mean Validation Dice}$.

---

## 📊 Experimental Results

### 1. Controlled 30-Epoch Validation Comparison (Mean across Seeds `42, 123, 2024`)
*Higher is better for Dice, IoU, Precision, Recall, and Boundary F1 (2 px tolerance); lower is better for 95th-percentile Hausdorff Distance (HD95).*

| Loss Objective | Validation Dice (Mean ± SD) | IoU (Jaccard) | Precision | Recall | Boundary F1 | HD95 (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Binary Cross-Entropy (BCE)** | $0.8683 \pm 0.0120$ | 0.7836 | **0.9493** | 0.8292 | 0.5246 | 14.21 |
| **Focal Loss** | $0.8756 \pm 0.0108$ | 0.7921 | 0.9463 | 0.8411 | 0.5318 | 14.24 |
| **Lovász-Hinge** | $0.8773 \pm 0.0117$ | 0.7969 | 0.9438 | 0.8485 | 0.5725 | 14.37 |
| **Boundary-Weighted BCE (BW-BCE)** | $0.8706 \pm 0.0036$ | 0.7897 | 0.9413 | 0.8391 | 0.5596 | 15.92 |
| **Three-Loss (Equal: 33.3% each)** | $0.8825 \pm 0.0073$ | 0.8042 | 0.9473 | 0.8528 | 0.5810 | 13.92 |
| **Three-Loss (Best: 45.2 / 27.4 / 27.4%)** | $0.8830 \pm 0.0107$ | 0.8050 | 0.9389 | 0.8612 | 0.5826 | 14.05 |
| **Two-Loss (Lovász + BW-BCE)** | **$0.8851 \pm 0.0060$** | **0.8081** | 0.9419 | **0.8620** | **0.5862** | **13.49** |

### 2. Final Hold-Out Test Evaluation (379 ISIC 2016 Test Pairs)

| Model Group | Test Dice (Mean ± SD) | IoU (Jaccard) | Precision | Recall | Boundary F1 | HD95 (px)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Selected Three-Loss Mixture** | $0.8781 \pm 0.0062$ | 0.8015 | 0.9164 | **0.8761** | 0.5649 | 13.19 |
| **Two-Loss Reference (Lovász + BW-BCE)** | **$0.8813 \pm 0.0037$** | **0.8054** | **0.9213** | 0.8758 | **0.5707** | **12.85** |

*\*Note: HD95 averages exclude 2 edge-case test images where seed-42 checkpoints produced empty masks (undefined HD95). Median per-image HD95 was 5.9 px for Two-Loss and 6.5 px for Three-Loss.*

#### Per-Seed Test Dice Breakdown
| Seed | Three-Loss Dice | Two-Loss Dice | Advantage |
| :---: | :---: | :---: | :--- |
| **42** | 0.8752 | **0.8790** | Two-loss higher by `+0.0038` |
| **123** | 0.8852 | **0.8856** | Two-loss higher by `+0.0004` |
| **2024** | 0.8739 | **0.8793** | Two-loss higher by `+0.0053` |

---

## ⚠️ Study Scope & Limitations

To maintain rigorous scientific framing, the following constraints apply to our current findings:
* **Validation-Tuned Search:** The six three-loss weight candidates were evaluated on the validation split; one finite-difference update explores a local neighborhood and does not prove global optimality.
* **Sample Size & Significance:** Three random seeds capture descriptive run-to-run variance rather than formal statistical significance intervals.
* **Patient Independence:** ISIC 2016 metadata lacks patient-level identifiers, so strict patient-disjoint splitting cannot be verified.
* **Task Scope:** The current models perform pixel-level lesion segmentation ($128 \times 128$ / $256 \times 256$ masks) and do not perform clinical malignancy diagnosis.

---

## Akriti Web Application & Telemetry Dashboard

In addition to the offline research notebooks, this repository includes **`app.py`**, a Gradio-based web application for interactive inference and server monitoring:

* **Analyse Tab:** Upload dermoscopic images or use a webcam to generate binary lesion masks, red-tinted boundary overlays, lesion area percentage, and per-request inference latency.
* **Network Telemetry Tab:** Tracks cumulative server uptime, active/peak concurrency, total data transferred, per-client request breakdown, and latency distributions (Mean, Median, P95, P99, Jitter).
* **Evaluation Tab:** Displays static benchmark metrics loaded from `metrics/seg_metrics.json`.

### Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Shreyas-Singh-28/dip.git
cd dip

# 2. Install dependencies
pip install gradio numpy pandas tensorflow pillow scikit-learn opencv-python

# 3. Place trained weights at weights/unet.keras and launch the server
python app.py
```
Access the interface locally at `http://127.0.0.1:7860` or from any device on your local network via `http://<server-ip>:7860`.

---

## 🙏 Acknowledgements & References

* **ISIC 2016 Challenge:** Gutman et al., *Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016*.
* **Reference Implementation Inspiration:** [yuanqing811/ISIC2018](https://github.com/yuanqing811/ISIC2018) for medical image segmentation workflow design.
