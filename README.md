# Bi-Temporal Siamese Deep Learning for Aerial Image Change Detection (LEVIR-CD+)
### Digital Egypt Pioneers Initiative (DEPI) — Artificial Intelligence Track Graduation Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-cu126-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official GitHub Repository:** [https://github.com/nhahub/NHA-5-130](https://github.com/nhahub/NHA-5-130)

---

## 👥 Project Team & Engineering Roles

| # | Team Member | Engineering Role & Core Ownership |
|:---:|---|---|
| 1 | **Ahmed Refaat Abd-Elmotelb** | **Lead ML Engineer & Pipeline Architect**<br>End-to-End Siamese Pipeline, Tiled Sliding-Window Inference, Production Web Studio, Master Thesis |
| 2 | **Ahmed Reda Mohamed** | **Deep Learning Specialist & Loss Design**<br>Differential Fusion Block (`\|A - B\|` + `[A, B]`), Weighted BCE (`w=10`) + Soft Dice Loss Formulation, Ablation Studies |
| 3 | **Raeed Youssef Mahfouz** | **Data Systems & Augmentation Specialist**<br>LEVIR-CD+ Discovery, Dihedral D4 & Photometric Augmentation, Bundled Demo Curation, Morphological Post-Processing |
| 4 | **Ahmed Mohamed Qotb** | **Evaluation Metrics & Quality Assurance Lead**<br>128-Scene Test Split Benchmarks, Confusion Matrix Analysis, Pixel Error Maps (TP/FP/FN), QA & Deliverables |

---

## 📽️ Final Graduation Defense Presentation (30-Minute Keynote)

Tailored specifically for the **30-minute academic committee defense** (20-22 min presentation + 3 min live demo + 5 min jury Q&A):

1. **Master PowerPoint Presentation Deck (`.pptx`)**:
   * File path: [`documentation/Graduation_Project_Final_Defense.pptx`](file:///d:/DEPI%20Project/documentation/Graduation_Project_Final_Defense.pptx) *(and root copy: [`Graduation_Project_Final_Defense.pptx`](file:///d:/DEPI%20Project/Graduation_Project_Final_Defense.pptx))*
   * **16:9 Widescreen Master Deck** with embedded high-resolution figures, training curves, confusion matrices, and equations.
   * **Presenter View Notes (Speaker Notes)** embedded directly into every slide: exact timing allocations, speaker script, and anticipated jury questions with model answers.
2. **Interactive Web Presentation Deck (Keynote Mode)**:
   * Launch in any browser: **`http://localhost:5000/static/presentation.html`**
   * Supports keyboard navigation (`→`, `←`, `Space`) and Fullscreen mode (`F`).
3. **Defense Speaker Guide & Jury Cheat Sheet**:
   * Word Docx: [`documentation/Defense_Presentation_30_Minutes_Guide.docx`](file:///d:/DEPI%20Project/documentation/Defense_Presentation_30_Minutes_Guide.docx)
   * Markdown: [`documentation/Defense_Presentation_30_Minutes_Guide.md`](file:///d:/DEPI%20Project/documentation/Defense_Presentation_30_Minutes_Guide.md)
   * Contains a **minute-by-minute timeline**, slide-by-slide script, and the **Top 10 Anticipated Jury Questions** with technical answers.

---

## 🚀 Quick Start & Production Web Studio

The repository includes a turnkey web studio with pre-loaded weights and 8 bundled benchmark scenes (100% offline, zero download required).

### Option 1: One-Click Launcher (Windows)
Double-click [`run_production.bat`](file:///d:/DEPI%20Project/run_production.bat) or run:
```cmd
run_production.bat
```
The browser will automatically open at: **`http://localhost:5000`**

### Option 2: Python Command Line
```bash
# 1. Activate your virtual environment
python app.py
```
Then navigate to: **`http://localhost:5000`**

### Option 3: Standalone CLI Inference (`predict.py`)
Run inference on arbitrary satellite image pairs directly from the terminal without launching the server:
```bash
python predict.py --before demo_samples/scene_01_train_638/A.png \
                  --after demo_samples/scene_01_train_638/B.png \
                  --output change_mask.png \
                  --overlay change_overlay.png
```

### Option 4: Cloud Docker Container
```bash
# Build and run with GPU acceleration via Docker Compose
docker compose up --build
```

---

## 📊 Quantitative SOTA Benchmarks (LEVIR-CD+ Test Split)

Rigorous zero-leakage evaluation across the official **128 benchmark test scenes** (1024×1024 resolution = 134.2 million evaluated pixels):

| Model Architecture | Fusion Mechanism | Loss Objective | Test Precision | Test Recall | Test IoU | Test F1-Score |
|---|---|---|:---:|:---:|:---:|:---:|
| **Siamese Baseline** | Channel Concatenation | BCE + Soft Dice | 0.793 | 0.720 | 0.605 | 0.754 |
| **Diff-Enhanced U-Net** | Diff-Cat (`\|A - B\|` + `[A, B]`) | Weighted BCE (`w=10`) + Dice | 0.824 | 0.749 | 0.647 | 0.785 |
| **🏆 SOTA Dual Ensemble** | Weighted Voting (0.45 + 0.55) | Hybrid Multi-Loss + 4-Fold TTA | **0.841** | **0.765** | **0.669** | **0.801** |

### Key Architectural Findings:
1. **Diff-Cat Fusion Impact**: Providing the decoder with explicit differential features `|F_A - F_B|` alongside concatenated representations yields a **+4.2% IoU leap** over naive concatenation.
2. **Class Imbalance Resolution**: Real building changes occupy less than 5% of pixels. Pos-weighted BCE (`w=10.0`) forces gradient updates on rare structural alterations, preventing trivial all-zero convergence.
3. **Ensemble & TTA Synergy**: Combining complementary models pushes the final F1 past the **0.80 benchmark threshold** with **11.47 Million true positive change pixels** accurately identified.
4. **Sub-Second Latency**: Optimized sliding-window inference with GPU batching processes an entire 1024×1024 scene in **~794 ms**.

---

## ✨ Web Studio Features & Capabilities

* **Interactive Curtain Swipe View**: Real-time before/after draggable divider allowing millimeter-precision inspection.
* **Geospatial Footprint Engine**: Automated pixel-to-ground conversion (0.5m GSD = 0.25 m²/pixel). Extracts structure count via `scipy.ndimage.label`, altered ground area in m² and hectares, and density indexing.
* **Morphological Noise Filtering**: 3×3 opening eliminates isolated sensor speckle noise; closing fills internal building courtyard voids.
* **Error Analysis Mapping**: Color-coded diagnostic maps (Neon Green = TP, Crimson Red = FP, Electric Blue = FN).
* **Export Hub**: One-click download of binary change masks, semi-transparent cyan overlays, and clipboard statistical reports.
* **Direct Documentation Downloads**: Download IEEE milestone reports (Phases 1-5), Master Thesis, and PowerPoint slides directly from the UI.

---

## 📁 Repository Directory Structure

```
NHA-5-130/
├── app.py                                   # Multi-threaded production backend (ThreadingHTTPServer)
├── predict.py                               # Standalone CLI predictor for terminal inference
├── run_production.bat                       # One-click Windows production launcher
├── Dockerfile                               # Multi-stage lightweight container
├── docker-compose.yml                       # GPU-accelerated compose service
├── requirements.txt                         # Python dependencies
├── README.md                                # Bilingual documentation (English & Arabic)
├── .gitignore                               # Git exclusion rules (<50MB compliance)
├── Graduation_Project_Final_Defense.pptx   # 30-Minute Master Keynote Slide Deck
├── Defense_Presentation_30_Minutes_Guide.docx# Defense speech script & jury Q&A guide
│
├── static/                                  # Modular clean frontend assets
│   ├── index.html                           # Glassmorphic responsive studio UI
│   ├── style.css                            # Modern dark theme styles
│   ├── app.js                               # Robust JavaScript controller
│   └── presentation.html                    # Interactive web presentation deck
│
├── models/                                  # Trained production weights (<50MB, no Git LFS needed)
│   ├── best_concat_model.pt                 # Baseline Concat weights (34.6 MB)
│   └── best_diffcat_model.pt                # Diff-Cat weights (39.6 MB)
│
├── demo_samples/                            # 8 bundled benchmark scenes for instant offline demo
│   ├── scene_01_train_638/                  # (A.png, B.png, label.png)
│   └── ...                                  # scene_02 through scene_08
│
├── documentation/                           # Formal IEEE milestones & thesis deliverables
│   ├── Graduation_Project_Master_Report.docx# Comprehensive Graduation Thesis
│   ├── Phase_01_Project_Plan_and_Scope.docx # Phase 1 Scope & Timeline
│   ├── Phase_02_Data_Pipeline_and_EDA.docx  # Phase 2 Data Pipeline & EDA
│   ├── Phase_03_Model_Architecture_and_Training.docx # Phase 3 Architecture & Loss
│   ├── Phase_04_Evaluation_and_Testing.docx # Phase 4 Benchmarks & Ablation
│   ├── Phase_05_Deployment_and_Optimization.docx # Phase 5 Production Manual
│   ├── Graduation_Project_Final_Defense.pptx # Master Slide Deck
│   ├── Defense_Presentation_30_Minutes_Guide.docx # Speaker Guide & Q&A
│   └── figures/                             # High-res diagrams, curves, and confusion matrices
│
├── notebooks/                               # Jupyter research notebooks
│   ├── 01_data_exploration.ipynb           # EDA and class imbalance analysis
│   ├── 02_model_training.ipynb              # Siamese training and loss tracking
│   └── 03_evaluation_and_metrics.ipynb      # Test split evaluation and PR curves
│
└── src/                                     # Core deep learning package
    ├── model.py                             # SiameseUNet with Diff-Cat fusion
    ├── dataset.py                           # Bi-temporal dataset loader & augmentations
    ├── loss.py                              # Weighted BCE + Soft Dice formulation
    ├── metrics.py                           # IoU, F1, Precision, Recall meters
    └── config.py                            # Hyperparameters and paths
```

---

<br>

# النسخة العربية الشاملة لمشروع التخرج 🇪🇬

مشروع تخرج مسار الذكاء الاصطناعي بمبادرة رواد مصر الرقمية (**Digital Egypt Pioneers Initiative - DEPI**).

### ملخص المشروع الهندسي
يهدف المشروع إلى حل أزمة المراقبة اليدوية البطيئة لصور الأقمار الصناعية والطائرات عبر بناء نظام تعلم عميق متكامل ثنائي التوقيت (Bi-temporal Siamese Deep Learning) قادر على رصد التوسعات العمرانية، والبناء المخالف، وتغيرات المنشآت بدقة فائقة وفي زمن يقل عن ثانية واحدة.

### فريق العمل وأدوارهم الهندسية:
1. **أحمد رفعت عبد المطلب** (Lead ML Engineer): مهندس خط الأنابيب، وتصميم التدريب والاستدلال المتسلسل، وبناء خادم الويب الإنتاجي، والتقرير الأكاديمي الشامل.
2. **أحمد رضا محمد** (Deep Learning Specialist): مهندس التعلم العميق، وتصميم بلوك الدمج التفاضلي `Diff-Cat`، وصياغة دالة الفقد الهجينة `Weighted BCE + Soft Dice`.
3. **رائد يوسف محفوظ** (Data Systems Specialist): مهندس نظم البيانات والمعالجة المورفولوجية، وتصميم الـ Augmentations المتزامنة، وإعداد العينات التجريبية.
4. **أحمد محمد قطب** (QA & Metrics Lead): مسؤول التقييم وضمان الجودة، وتحليل مصفوفات الارتباك بكسل-بكسل، وتدقيق مقاييس الـ SOTA Benchmark.

---

### عرض المناقشة النهائية (30 دقيقة) 📽️
* **ملف البوربوينت (.pptx)**: متواجد باسم [`Graduation_Project_Final_Defense.pptx`](file:///d:/DEPI%20Project/Graduation_Project_Final_Defense.pptx) (16:9 بنظام الشرائح العريضة ومدمج داخل كل شريحة ملاحظات المتحدث والأسئلة المتوقعة).
* **دليل الإلقاء والأسئلة العشرة الذهبية المتوقعة**: متواجد في [`Defense_Presentation_30_Minutes_Guide.docx`](file:///d:/DEPI%20Project/Defense_Presentation_30_Minutes_Guide.docx) (جدول دقيقة بدقيقة، نص الإلقاء، وإجابات الأسئلة المتوقعة بالأرقام).
* **العرض التفاعلي عبر المتصفح**: افتح الرابط `http://localhost:5000/static/presentation.html` واضغط `F` لوضع ملء الشاشة.

---

### خطوات التشغيل السريع:
* **عبر ويندوز بنقرة واحدة**: دبل كليك على [`run_production.bat`](file:///d:/DEPI%20Project/run_production.bat).
* **عبر سطر الأوامر**: `python app.py` ثم توجه إلى `http://localhost:5000`.
* **عبر أداة الـ CLI**: `python predict.py --before A.png --after B.png --output mask.png`.

---

## 📜 Citation & Academic Attribution

If you utilize this pipeline, dataset configuration, or architectural implementation, please cite:

```bibtex
@misc{depi_aerial_change_detection_2026,
  author = {Abd-Elmotelb, Ahmed Refaat and Mohamed, Ahmed Reda and Mahfouz, Raeed Youssef and Qotb, Ahmed Mohamed},
  title = {Bi-Temporal Siamese Deep Learning for High-Resolution Aerial Image Change Detection on LEVIR-CD+},
  year = {2026},
  publisher = {GitHub},
  journal = {Digital Egypt Pioneers Initiative (DEPI) Graduation Projects},
  howpublished = {\url{https://github.com/nhahub/NHA-5-130}}
}
```
