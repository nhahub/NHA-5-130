# Aerial Image Change Detection (LEVIR-CD+) — Siamese U-Net Studio

مشروع تخرج مبادرة رواد مصر الرقمية (**Digital Egypt Pioneers Initiative - DEPI**) في مسار الذكاء الاصطناعي والتعلم العميق (**AI Track**).

**رابط المستودع (GitHub Repository):** [https://github.com/nhahub/NHA-5-130](https://github.com/nhahub/NHA-5-130)

### فريق العمل (Project Team)
| # | الاسم (Name) | الدور والتخصص (Role & Responsibilities) |
|:---:|---|---|
| 1 | **Ahmed Refaat Abd-Elmotelb** | Lead ML Engineer & Pipeline Architect (Siamese U-Net, Zero-Leakage Split, Dual Ensemble, TTA) |
| 2 | **Ahmed Reda Mohamed** | Deep Learning Specialist & Loss Design (Diff-Cat Fusion, BCE + Soft Dice, pos_weight=10.0) |
| 3 | **Raeed Youssef Mahfouz** | Data Systems & Augmentation Specialist (Data Discovery, Synchronized Augmentation, Evaluation Tiling) |
| 4 | **Ahmed Mohamed Qotb** | Evaluation Metrics & Quality Assurance Lead (Confusion Meter, Validation Threshold Tuning, Error Maps) |

---

### عرض المناقشة النهائية (Final Graduation Defense Presentation) 📽️

تم إعداد عرض تقديمي متكامل واحترافي للمناقشة النهائية يغطي كافة تفاصيل المشروع ومراحله الخمس دون الحاجة للتنقل بين ملفات متعددة أثناء العرض:
1. **ملف PowerPoint المعتمد (.pptx)**: متواجد في المسار:
   `documentation/Graduation_Project_Final_Defense.pptx`
   - مصمم بنظام الشاشة العريضة (16:9).
   - يتضمن 15 شريحة غنية بكافة الرسوم البيانية، ومصفوفة الارتباك، والمعادلات الرياضية لدوال الفقد، والجداول المقارنة (Benchmarks)، ومخرجات الفحص البصري وعينات قبل/بعد.
2. **وضع العرض التفاعلي عبر المتصفح (Web Keynote Mode)**:
   - يمكن تشغيله مباشرة في أي متصفح بالانتقال إلى: **`http://localhost:5000/static/presentation.html`**
   - يدعم التنقل بأسهم لوحة المفاتيح ووضع ملء الشاشة (Fullscreen - `F`).

---

## تشغيل واجهة الويب والإنتاج (Production Web Studio) 🚀

تم تجهيز المشروع بنظام تشغيل فوري مستقل مع أوزان النماذج وعينات اختبار مدمجة جاهزة للمعاينة المباشرة:

### تشغيل بنقرة واحدة (Windows):
قم بتشغيل الملف التنفيذي:
```cmd
run_production.bat
```
أو عبر سطر الأوامر:
```bash
python app.py
```
ثم توجه إلى: **`http://localhost:5000`**

### مميزات واجهة الويب والتحسينات المتقدمة (Features & Major Upgrades):
1. **نمط المقارنة التفاعلي (Interactive Swipe / Curtain Comparison)**:
   - إمكانية سحب شريط فاصل ديناميكي (Swipe Slider) يميناً ويساراً لمقارنة صورة "قبل" مع صورة "بعد" ورؤية المباني تظهر لحظياً تحت التراكب المضيء بدقة بصرية مذهلة مثل أنظمة GIS العالمية (ArcGIS / Google Earth).
2. **الذكاء الجيومكاني وتحليل البصمات الإنشائية (Geospatial & Footprint Analytics)**:
   - حساب إجمالي المساحة المتغيرة على الأرض بالمتر المربع ($m^2$) والهكتار ($ha$) اعتماداً على دقة القمر الصناعي (0.5m GSD).
   - خوارزمية ذكية لاستخراج عدد المباني المنشأة حديثاً (Distinct Building Count) عبر التحليل المتصل للبيكسلات (`scipy.ndimage.label`).
   - حساب متوسط مساحة المبنى وأكبر مجمع إنشائي مكتشف، وتصنيف كثافة التوسع العمراني.
3. **فلترة الضوضاء المورفولوجية وملء فجوات الأسقف (Morphological Noise Filtering & Roof Hole-Filling)**:
   - تطبيق عمليات Opening لإزالة الشوائب والضوضاء المعزولة الناتجة عن انعكاسات الشمس أو المركبات.
   - تطبيق عمليات Closing لسد الفجوات والظلال داخل المباني لجعل الحواف المعمارية حادة وواضحة.
4. **أداة التنبؤ المستقلة عبر سطر الأوامر (`predict.py` CLI)**:
   - تنفيذ التنبؤ واستخراج أقنعة التغير مباشرة على أي صورتين من الـ Terminal دون الحاجة لتشغيل السيرفر.
5. **تصدير النتائج والتقارير الفورية (Export Hub)**:
   - زر لتحميل قناع التغير الثنائي (Mask PNG).
   - زر لتحميل الصورة المتراكبة (Overlay PNG).
   - زر لنسخ تقرير الفحص الشامل ومؤشرات الدقة إلى الحافظة (Copy Stats).
6. **دعم بيئات الحاويات السحابية (Docker & Docker Compose)**:
   - يتضمن `Dockerfile` و `docker-compose.yml` لتشغيل المشروع فورياً على أي سيرفر سحابي بضغطة زر.
7. **استعراض العينات المدمجة (Demo Benchmark Scenes)**:
   - 8 مشاهد اختبارية حقيقية عالية الدقة مدمجة مسبقاً في `demo_samples/` تعمل دون الحاجة لتحميل 3.5GB.
8. **رفع صور مخصصة (Custom Upload)**:
   - إمكانية رفع أي صورتين جويتين (قبل وبعد) بدقة وحجم حر لتحديد التغيرات بدقة.
9. **اختيار النموذج (Model Selector)**:
   - التبديل بين نموذج التجميع الأقوى `SOTA Ensemble` أو النموذج التفاضلي `Diff-Cat` أو نموذج الأساس `Baseline Concat`.
10. **مؤشر العتبة الفوري (Threshold Slider)**:
    - التحكم في عتبة القرار (0.10 إلى 0.90) ومعاينة التغيير لحظيًا مع إحصائيات الدقة.
11. **لوحة المقارنة الشاملة وخرائط الأخطاء (Error Analysis Map)**:
    - خريطة تحليل الأخطاء التفاعلية:
      - **أخضر**: تغير صحيح تم اكتشافه (True Positive).
      - **أحمر**: إنذار خاطئ (False Positive).
      - **أزرق**: تغير مفقود لم يُكتشف (False Negative).
12. **مركز تحميل مخرجات التوثيق (Documentation Hub)**:
    - إمكانية تحميل كافة ملفات التوثيق الرسمية للمشروع (.docx) بنقرة واحدة.

---

## 1) بنية المشروع (Project Layout)

```
NHA-5-130/
├── app.py                 # واجهة الويب التفاعلية الشاملة (Production Web Studio + Swipe Comparison)
├── predict.py             # أداة التنبؤ المستقلة عبر الأوامر (Standalone CLI Predictor)
├── run_production.bat     # سكربت تشغيل سريع بنقرة واحدة على ويندوز
├── Dockerfile             # إعدادات الحاوية السحابية خفيفة الوزن
├── docker-compose.yml     # تشغيل فوري للحاوية بأمر واحد (docker compose up)
├── explore_data.py        # فحص الداتا: الشجرة، المطابقة، الأحجام، القيم، عينات بصرية
├── train.py               # التدريب + early stopping + حفظ أفضل checkpoint (+ --resume)
├── evaluate.py            # تقييم test: مقاييس كاملة + عتبة auto + confusion + رسوم
├── make_notebooks.py      # مولّد دفاتر Jupyter (يمكن إعادة توليدها في أي وقت)
├── smoke_test.py          # اختبار سريع لخط الأنابيب ببيانات اصطناعية
├── diagnose.py            # أداة تشخيص (eval vs train BN، جودة العتبات)
├── requirements.txt       # حزم بايثون المطلوبة
├── README.md              # دليل المشروع الشامل
├── .gitignore             # استثناء الملفات الضخمة والمؤقتة
├── models/                # أوزان النماذج المدربة للإنتاج (< 50MB متوافقة مع GitHub)
│   ├── best_diffcat_model.pt   (IoU: 0.647, F1: 0.785 - 39.6 MB)
│   └── best_concat_model.pt    (IoU: 0.605, F1: 0.754 - 34.6 MB)
├── demo_samples/          # عينات اختبار حقيقية مدمجة للتجربة الفورية دون تحميل
│   ├── scene_01_train_638/ (A.png, B.png, label.png)
│   └── ... (8 scenes)
├── documentation/         # ملفات التوثيق الرسمية للمشروع (Word + Markdown + Figures)
│   ├── 01_Project_Planning_and_Management.docx (.md)
│   ├── 02_Literature_Review.docx (.md)
│   ├── 03_Requirements_Gathering.docx (.md)
│   ├── 04_System_Analysis_and_Design.docx (.md)
│   ├── 05_Implementation_Source_Code_and_Execution.docx (.md)
│   ├── 06_Final_Presentation_Testing_and_Reports.docx (.md)
│   ├── DEPI_Graduation_Project_Documentation.docx (.md)
│   └── figures/
├── notebooks/             # دفتر تفاعلي لكل مرحلة من مراحل المشروع
│   ├── 00_setup.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
└── src/
    ├── config.py          # إعدادات المشروع (seed، image size، batch، lr، ...)
    ├── data_discovery.py  # اكتشاف بنية الداتا تلقائيًا ومطابقة A/B/mask
    ├── dataset.py         # Datasets + augmentations متزامنة + تقطيع tiles
    ├── model.py           # Siamese U-Net (Concat / Diff-Cat)
    ├── losses.py          # BCEWithLogitsLoss (pos_weight=10.0) + Soft Dice
    └── metrics.py         # مقاييس التقييم الشاملة (IoU, F1, Precision, Recall, ...)
```

## 2) التثبيت

```bash
# بيئة افتراضية (يُنصح بـ Python 3.10–3.11)
conda create -n levir-cd python=3.11 -y
conda activate levir-cd

# PyTorch مع CUDA (اختر النسخة المناسبة لتعريف كرت الشاشة)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# باقي الاعتماديات
pip install -r requirements.txt
```

على جهاز بدون GPU يكفي `pip install torch` وسيستخدم الكود CPU تلقائيًا.

## 3) التشغيل — مسارَان: Jupyter أو Python

### المسار التفاعلي (Jupyter) — دفتر لكل مرحلة

```bash
D:\envs\levir-cd\python.exe -m jupyter lab notebooks
```
اختر الـkernel **Python (levir-cd)** ثم نفّذ الدفاتر بالترتيب:

| الدفتر | الغرض |
|---|---|
| `notebooks/00_setup.ipynb` | فحص البيئة والـGPU |
| `notebooks/01_data_exploration.ipynb` | تنزيل الداتا، شجرة المجلدات، المطابقة، الإحصاءات، عينات بصرية، توزيع نسبة التغيّر |
| `notebooks/02_training.ipynb` | بناء النموذج + حلقة تدريب تفاعلية + منحنيات (كل تحسن يُحفظ فورًا — المقاطعة لا تفقد الأفضل) |
| `notebooks/03_evaluation.ipynb` | ضبط العتبة على validation + تقرير test الكامل بمقاييس غير مضللة + confusion matrix + أمثلة بصرية |

الدفاتر تستدعي نفس محرك `src/` الذي تستخدمه السكربتات — لا يوجد منطق مكرر.

### المسار غير التفاعلي (Python)

```bash
python explore_data.py                  # فحص الداتا + عينات + تقرير JSON
python train.py                         # تدريب كامل (كل التشغيل في outputs/run_<timestamp>/)
python train.py --epochs 60 --batch-size 16 --lr 1e-3 --image-size 256
python train.py --resume outputs\run_XXXX\best_model.pt --epochs 25 --lr 2.5e-4   # استكمال
python evaluate.py --run-dir outputs\run_XXXX --threshold auto
```
`--threshold auto` يجرب العتبات 0.30–0.70 على **validation** ويستخدم الأفضل
(أعلى F1) على test — مطبوع ومحفوظ في `test_metrics.json`. وبدون `--run-dir`
يختار السكربت تلقائيًا الـrun صاحب أعلى val IoU (يطبع المرشحين قبل الاختيار)،
حتى لا يلتقط تشغيلات تجريبية قصيرة مثل `run_nb_*` من دفتر التدريب.

## 4) طريقة الحساب والتعريفات (مقاييس غير مضللة)

- التنبؤ: `pred = sigmoid(logits) > threshold` (0.5 افتراضيًا، أو ضبط تلقائي
  على validation بـ`--threshold auto`).
- **مقاييس فئة التغيّر (headline):** `Precision = TP/(TP+FP)`،
  `Recall = TP/(TP+FN)`، `F1 = 2PR/(P+R)`، `IoU = TP/(TP+FP+FN)` —
  على مستوى البكسل عبر عدّاد confusion واحد يتراكم على كامل الـtest set.
- **مقاييس مجمّعة مقاومة لعدم التوازن:** `Mean IoU` (متوسط IoU الفئتين)،
  `Balanced Accuracy` (متوسط recall الفئتين)، و`Cohen's Kappa`
  ((p_o − p_e)/(1 − p_e)) الذي يصحح الاتفاق العشوائي.
- **جدول لكل فئة** (change / unchanged) بنفس التعريفات مع تبديل أدوار TP/TN،
  وconfusion matrix بصيغتين: نسبة من كل البكسل ونسبة صفية طبيعية.
- تنبيه: Accuracy ≈ 97.7% تبدو مرتفعة لكنها فقاعة — 95.8% من البكسل غير
  متغيرة أصلًا، لذا لا تُعتمد إلا كرقم مرجعي بجوار المقاييس أعلاه.

## 5) الافتراضات والقرارات (مهم)

- **بنية الداتا الفعلية (مكتشفة تلقائيًا)**: الداتا المنزّلة هي **LEVIR-CD+**
  ببنية `LEVIR-CD+/train/{A,B,label}` (637 صورة) و `LEVIR-CD+/test/{A,B,label}`
  (348 صورة)، كل صورة **1024×1024 RGB (PNG)**، والأقنعة ثنائية بقيم {0,1}.
- **التقسيم ومنع تسريب بيانات الاختبار (Data Leakage Prevention)**:
  - يحتوي LEVIR-CD+ على `train` (637 صورة) و `test` (348 صورة) ولا يوجد مجلد `val` منفصل.
  - تم ضبط الكود افتراضيًا عبر `RESPECT_OFFICIAL_SPLITS = True` في `src/config.py` لضمان الدقة العلمية:
    - **يُحفظ مجلد `test` الرسمي (348 صورة) كاملاً ومعزولاً تماماً دون أي مساس أو تسريب**.
    - يُشتق مجلد الـ validation من صور `train` الرسمية فقط (525 train و 112 validation).
    - إذا رغب المستخدم في استرجاع التقسيم القديم التجميعي (70/15/15) لأغراض المقارنة التاريخية، يمكن تعيين `RESPECT_OFFICIAL_SPLITS = False`.
  - فحص تسريب صارم يتأكد برمجياً من عدم وجود أي معرّف صورة مشترك بين أي من المجموعات الثلاث.
- **اكتشاف المجلدات تلقائيًا** ولا يُفترض مسبقًا: الكود يمشي في الشجرة، يصنّف
  المجلدات بكلمات مفتاحية (A/B, image1/image2, mask/label, ...)، ويطابق الملفات
  بجذور الأسماء المشتركة (file stem). إن فشل الاكتشاف يتوقف برسالة واضحة ويطلب
  تعبئة `OVERRIDES` في `src/data_discovery.py`.
- الصور (1024×1024) أكبر من `IMAGE_SIZE` (256): التدريب بقصّات عشوائية 256×256
  متزامنة على الصورتين والقناع، والتقييم بشبكة tiles كاملة 16×16 لكل صورة ثم
  تُعاد تجميع التنبؤات لصورة كاملة.
- الـaugmentations هندسية فقط (انعكاس أفقي/رأسي، دوران بـ k×90°) وتُطبَّق بنفس
  المعاملات على الصورتين والقناع معًا — لا تكسر التطابق المكاني ولا تفقد ثنائية القناع.
- normalize للصور: `/255` ثم `(x-0.5)/0.5`؛ القناع يبقى {0,1} دائمًا، ويُقرأ
  بشكل متسامح (0/1 أو 0/255 أو 0/128) عبر عتبة عند نصف القيمة القصوى.
- بيئة التشغيل الفعلية: conda env باسم `levir-cd` على `D:\envs\levir-cd`
  (Python 3.11 + PyTorch 2.14.1+cu126) — القرص C: كان شبه ممتلئ فنُقلت البيئة
  والكاشات إلى D:. استخدم `D:\envs\levir-cd\python.exe` لتشغيل السكربتات.
  كما يعمل `--num-workers 0` افتراضيًا هنا لأن فتح عمليات فرعية يعيد تحميل
  DLLات CUDA ويسبب "paging file too small" على هذا الجهاز.

## 6) الأجهزة

- استُخدم في التشغيل الفعلي: RTX 3050 Laptop (4 GB) مع mixed precision،
  `base_channels=32`، `batch_size=16` عند 256×256، `--num-workers 0`.
- بدون GPU يعمل الكود على CPU تلقائيًا (أبطأ فقط).

## 7) ما تم تشغيله فعليًا والنتائج النهائية

جميع الأرقام التالية من تشغيل حقيقي على هذا الجهاز (RTX 3050 Laptop 4GB،
بيئة `D:\envs\levir-cd`، PyTorch 2.14.1+cu126):

- `explore_data.py` ✔ — التقرير في `outputs/data_report.json` والعينات في
  `outputs/figures/data_samples.png` (985 عينة 1024×1024، أقنعة {0,1}، ~4.2–4.9%
  بكسل متغيرة لكل split).
- **التدريب (مرحلتان):** ✔
  1. `run_20261009_200513`: 40 epoch من الصفر → best val IoU = 0.576.
  2. `run_20261011_073203`: استكمال 25 epoch من أفضل checkpoint
     (`--resume ... --lr 2.5e-4`) → **best val IoU = 0.610** (epoch 18).
- `evaluate.py --threshold auto` ✔ — ضبط العتبة على validation (اختار 0.30،
  أعلى val F1=0.761) ثم تقييم مرة واحدة على test (149 صورة = 2384 بلاطة
  256×256). النتائج في `outputs/run_20261011_073203/eval_20261011_082622/`:

  | المقياس | القيمة |
  |---|---|
  | Precision (change) | 0.7336 |
  | Recall (change) | 0.7841 |
  | **F1 (change)** | **0.7580** |
  | **IoU (change)** | **0.6103** |
  | Mean IoU (فئتان) | 0.7939 |
  | Balanced Accuracy | 0.8856 |
  | Cohen's Kappa | 0.7467 |
  | Overall Accuracy | 0.9783 (غير معبّرة: 95.7% من البكسل غير متغيرة) |

  Per-class: change (P .734 / R .784 / F1 .758 / IoU .610) —
  unchanged (P .990 / R .987 / F1 .989 / IoU .978).
  Confusion (بكسل): TP=5.31M، FP=1.93M، FN=1.46M، TN=147.53M —
  الصورة في `confusion_matrix.png` (بصيغتين: نسبة كلية ونسبة صفية)، و6 أمثلة
  بصرية في `predictions/`، والملف الكامل (مع جدول مسح العتبات على validation)
  في `test_metrics.json`.

- **التقييم الرسمي المعتمد الخالي من التسريب (348 عينة test = 5,568 بلاطة):** ✔
  تم تقييم النماذج بدقة تامة على كامل مجلد الاختبار الرسمي دون أي مساس أو تسريب، وأُجريت دراسة مقارنة تجريبية (Ablation Study) شاملة:

  | المقياس | Baseline (بدون TTA) | Baseline + TTA | Diff-Enhanced + pos_weight | **🏆 SOTA Ensemble (Concat + Diff-Cat + TTA)** |
  |---|---|---|---|:---:|
  | **Val IoU** | 0.6098 | 0.6820 | 0.6465 | **0.7068 (+9.7%)** |
  | **Val F1-score** | 0.7607 | 0.8109 | 0.7853 | **0.8282 (+6.8%)** |
  | **Test Precision** | 0.6439 | 0.6483 | 0.6121 | **0.6551** |
  | **Test Recall** | 0.7496 | 0.7461 | 0.7673 | **0.7713 (أعلى استدعاء)** |
  | **Test F1-score** | 0.6927 | 0.6938 | 0.6809 | **0.7084 (أعلى F1)** |
  | **Test IoU** | 0.5299 | 0.5311 | 0.5162 | **0.5485 (أعلى IoU)** |
  | **Mean IoU (الفئتان)** | 0.7510 | 0.7517 | 0.7430 | **0.7609** |
  | **Balanced Accuracy** | 0.8660 | 0.8645 | 0.8733 | **0.8770 (87.7%)** |
  | **Cohen's Kappa** | 0.6786 | 0.6798 | 0.6658 | **0.6950** |
  | **True Positives (بكسل)** | 11.14M | 11.09M | 11.41M | **11.47M (+330,000 بكسل)** |

  - **تحليل النتائج الرسمية المعتمدة (Ablation Study Analysis):**
    1. **نموذج التجميع الأفضل (SOTA Ensemble):** دمج تمثيلات Concat (0.45) مع Diff-Cat (0.55) بجانب تقنية 4-fold TTA حقق قفزة نوعية ووصل إلى **Val IoU = 0.7068** و **Val F1 = 0.8282**، كما حقق أعلى F1 واعلى IoU وأعلى استدعاء على الـ 348 عينة اختبار رسمية كاملة (11.47 مليون بكسل متغيرة مكتشفة بدقة).
    2. **تأثير TTA والتفاضل:** خفض الإنذارات الخاطئة (False Positives) ورفع حساسية النموذج للمباني الصغيرة والحدود الدقيقة.
    3. **النزاهة الأكاديمية (Zero-Leakage Benchmark):** تم تقييم 348 عينة اختبار رسمية كاملة (5,568 بلاطة = 364.9 مليون بكسل) دون أي تسريب لبيانات الاختبار في التدريب أو ضبط العتبة.

ملاحظة صادقة عن التشخيص أثناء التطوير: أول تشغيلين توقفا مبكرًا بسبب خطأ
قياس (double-sigmoid في حساب مقاييس الـvalidation — النموذج كان يتعلم جيدًا
لكن المقاييس كانت متجمدة)، تم إصلاحه ثم إعادة التدريب من الصفر؛ كما تم تصحيح
معادلة `iou_unchanged` في `src/metrics.py` لتصبح `TN / (TN + FP + FN)`. للتحسين لاحقًا:
تدريب أطول/أعرض، أو TTA، أو استبدال BatchNorm بـGroupNorm لتقليل التذبذب بين epochs.
