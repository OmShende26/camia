# CAMIA Execution Flow - Complete File Dependency Map

## Entry Point
```bash
bash run_camia_custom.sh
```

---

## File Execution Order & Dependencies

### **STEP 1: Feature Construction**
```bash
python run_ours_construct_mia_data_custom.py base_model=common-pile/comma-v0.1-2t
```

#### **Direct Imports:**
```
run_ours_construct_mia_data_custom.py (MAIN)
├── utils.py                           [utility functions: chunks, random_sampling, etc.]
├── mimir/config.py                    [ExperimentConfig dataclass]
├── omegaconf (external)               [Hydra config framework]
├── hydra (external)                   [Config management]
├── mimir/utils.py                     [fix_seed function]
└── mimir/models_without_debugging.py  [LanguageModel class]
    ├── mimir/config.py                [ExperimentConfig]
    ├── mimir/custom_datasets.py       [SEPARATOR constant]
    └── mimir/data_utils.py            [drop_last_word function]
```

#### **Configuration Files:**
```
conf/config.yaml                       [Hydra config - loaded via @hydra.main decorator]
```

#### **Output Generated:**
```
results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
├── all_features_custom.pkl           [Member & non-member features]
└── run logs
```

---

### **STEP 2: Train Logistic Regression Classifier**
```bash
python run_ours_train_lr.py base_model=common-pile/comma-v0.1-2t
```

#### **Direct Imports:**
```
run_ours_train_lr.py (MAIN)
├── util_features.py                   [Feature processing & model training]
│   ├── sklearn (external)             [LogisticRegression, PCA, metrics]
│   ├── pandas (external)              [DataFrame operations]
│   └── scipy (external)               [Statistical functions]
├── pickle (external)                  [Load .pkl files from step 1]
├── numpy (external)                   [Numerical operations]
└── omegaconf (external)               [Config loading]
```

#### **Input Files (from Step 1):**
```
results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
└── all_features_custom.pkl
```

#### **Output Generated:**
```
results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
├── trained_model_*.pkl               [Trained LogisticRegression classifier]
├── mia_scores_*.json                 [MIA predictions]
└── performance_metrics.json           [AUC, accuracy, precision, recall]
```

---

### **STEP 3: Generate ROC Curves**
```bash
python run_ours_get_roc.py base_model=common-pile/comma-v0.1-2t
```

#### **Direct Imports:**
```
run_ours_get_roc.py (MAIN)
├── util_features.py                   [Feature loading & processing]
├── pickle (external)                  [Load model & predictions]
├── numpy (external)                   [Numerical operations]
├── scipy (external)                   [ROC curve calculations]
├── sklearn.metrics (external)         [roc_curve, roc_auc_score]
└── matplotlib (external)              [Plot generation]
```

#### **Input Files (from Step 2):**
```
results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
├── trained_model_*.pkl
├── mia_scores_*.json
└── performance_metrics.json
```

#### **Output Generated:**
```
results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
├── roc_curve_plot.png                [ROC visualization]
├── roc_metrics.json                  [AUC @ different FPR thresholds]
└── final_results.json                [Complete evaluation metrics]
```

---

## Complete File List - Internal Repo Files

### **Core Execution Files** (3 files)
```
1. run_ours_construct_mia_data_custom.py    [Feature extraction - CUSTOM]
2. run_ours_train_lr.py                     [Model training]
3. run_ours_get_roc.py                      [ROC curve generation]
```

### **Utility Files** (2 files)
```
4. utils.py                                 [General utilities]
5. util_features.py                         [Feature engineering & ML]
```

### **Configuration** (1 file)
```
6. conf/config.yaml                         [Hydra configuration]
```

### **MIMIR Module Files** (7 files)
```
7. mimir/config.py                          [Config dataclasses]
8. mimir/utils.py                           [Seed fixing, cache utilities]
9. mimir/models_without_debugging.py        [LanguageModel class]
10. mimir/data_utils.py                     [Data loading utilities]
11. mimir/custom_datasets.py                [Dataset constants & handlers]
12. mimir/models.py                         [Model base classes]
13. mimir/plot_utils.py                     [Plotting utilities] (if called)
```

### **NOT Called** (Skipped for CAMIA-only)
```
❌ run_baselines.py                         [Baseline attacks - SKIPPED]
❌ run_ref_baselines.py                     [Reference model attacks - SKIPPED]
❌ run_ours_different_agg.py                [Different aggregations - OPTIONAL]
❌ mimir/attacks/ (entire folder)           [Attack implementations - NOT USED]
```

---

## External Dependencies Called

### **PyTorch Ecosystem:**
```
torch, torchvision                      [GPU computation, tensor operations]
transformers                            [HuggingFace model loading]
```

### **Data Processing:**
```
numpy, pandas, scipy                    [Numerical/statistical operations]
datasets                                [Hugging Face datasets library]
```

### **ML/Analysis:**
```
sklearn                                 [Logistic regression, PCA, metrics]
scipy.stats                             [Statistical distributions]
shap                                    [Feature importance]
```

### **Configuration & Utilities:**
```
omegaconf, hydra                        [Config management]
tqdm                                    [Progress bars]
pickle, json, os, time                  [I/O and utilities]
```

### **Visualization:**
```
matplotlib                              [Plotting]
```

---

## File I/O Summary

### **Inputs:**
```
/content/seen_books.jsonl               [Member dataset - from colab]
/content/unseen_books.jsonl             [Non-member dataset - from colab]
conf/config.yaml                        [Configuration]
HuggingFace Hub                         [Model: common-pile/comma-v0.1-2t]
```

### **Outputs:**
```
results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
├── all_features_custom.pkl
├── trained_model_*.pkl
├── mia_scores_*.json
├── performance_metrics.json
├── roc_metrics.json
├── roc_curve_plot.png
└── final_results.json
```

---

## Execution Dependency Graph

```
run_camia_custom.sh
│
├─→ Step 1: run_ours_construct_mia_data_custom.py
│   ├─ Loads: utils.py, mimir/config.py, mimir/utils.py
│   ├─ Uses: mimir/models_without_debugging.py
│   ├─ Reads: /content/seen_books.jsonl, /content/unseen_books.jsonl
│   └─ Outputs: all_features_custom.pkl ✓
│
├─→ Step 2: run_ours_train_lr.py
│   ├─ Loads: util_features.py, pickle, sklearn
│   ├─ Reads: results_new/.../all_features_custom.pkl
│   └─ Outputs: trained_model_*.pkl, mia_scores_*.json ✓
│
└─→ Step 3: run_ours_get_roc.py
    ├─ Loads: util_features.py, scipy, sklearn.metrics, matplotlib
    ├─ Reads: results_new/.../trained_model_*.pkl
    └─ Outputs: roc_curve_plot.png, roc_metrics.json ✓
```

---

## Key Notes

### **Files That Need Fixing:**
```
⚠️ mimir/models_without_debugging.py
   - Line 15: from hf_olmo import *    [SHOULD BE REMOVED]
   - Same simplification needed as models.py
```

### **Files That Still Need Cleanup (Optional):**
```
❓ mimir/models.py
   - Already cleaned up ✓
```

### **Execution Time:**
```
Step 1 (Feature extraction):  ~5-10 min
Step 2 (Model training):       ~2-5 min
Step 3 (ROC generation):       ~2-3 min
───────────────────────────
Total: ~10-20 minutes
```

### **Memory Requirements:**
```
- HuggingFace Model: ~4-8 GB
- Feature extraction: ~2-4 GB
- Training: ~1-2 GB
Total: ~8-14 GB (GPU)
```

---

## To Verify Files Are Called

You can trace execution with:
```bash
# Verbose mode
python -v run_ours_construct_mia_data_custom.py 2>&1 | grep "import"

# Or add print statements to track file loading
# (Already present in most files)
```

