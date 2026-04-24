# CAMIA Execution - Complete File Call Trace

## Quick Answer: What Files Are Called

When you run CAMIA (no baselines), these **internal repo files** are executed in order:

### **Phase 1: Feature Extraction**
```
✓ run_ours_construct_mia_data_custom.py
  ├─ utils.py
  ├─ mimir/config.py
  ├─ mimir/utils.py  
  ├─ mimir/models_without_debugging.py
  │  ├─ mimir/config.py
  │  ├─ mimir/custom_datasets.py
  │  └─ mimir/data_utils.py
  ├─ conf/config.yaml (loaded via @hydra.main)
  └─ Output: results_new/.../all_features_custom.pkl
```

### **Phase 2: Model Training**
```
✓ run_ours_train_lr.py
  ├─ util_features.py
  ├─ conf/config.yaml (if used)
  └─ Input: results_new/.../all_features_custom.pkl
     Output: trained_model_*.pkl, mia_scores_*.json
```

### **Phase 3: ROC Generation**
```
✓ run_ours_get_roc.py
  ├─ util_features.py
  ├─ conf/config.yaml (if used)
  └─ Input: trained_model_*.pkl
     Output: roc_curve_plot.png, roc_metrics.json
```

---

## Complete File Inventory

### **Project Root Files** (3 main executables)
```
1. run_ours_construct_mia_data_custom.py     [MAIN - Feature extraction]
2. run_ours_train_lr.py                      [MAIN - Training]
3. run_ours_get_roc.py                       [MAIN - ROC curves]
```

### **Supporting Utility Files** (2 files)
```
4. utils.py                                  [General utilities & functions]
5. util_features.py                          [Feature engineering & analysis]
```

### **Configuration** (1 file)
```
6. conf/config.yaml                          [Hydra configuration]
```

### **MIMIR Module Files** (5 core files)
```
7. mimir/config.py                           [ExperimentConfig, dataclasses]
8. mimir/utils.py                            [fix_seed, environment utilities]
9. mimir/models_without_debugging.py         [LanguageModel class ✓ CLEANED]
10. mimir/data_utils.py                      [Data loading utilities]
11. mimir/custom_datasets.py                 [Dataset handlers & constants]
```

### **Optional MIMIR Files** (called if needed)
```
? mimir/models.py                            [Fallback model definitions]
? mimir/plot_utils.py                        [Plotting utilities]
```

### **NOT Called** (explicitly skipped)
```
❌ run_baselines.py                          [Baseline attacks - SKIPPED]
❌ run_ref_baselines.py                      [Reference attacks - SKIPPED]
❌ run_ours_different_agg.py                 [Optional aggregations - SKIPPED]
❌ mimir/attacks/ (entire folder)            [Attack implementations - NOT USED]
```

---

## File Execution Timeline

```
START: bash run_camia_custom.sh
  │
  ├─ Phase 1: 0-10 minutes
  │  │
  │  └─→ python run_ours_construct_mia_data_custom.py
  │      ├─ Load config from: conf/config.yaml
  │      ├─ Load utilities from: utils.py, mimir/utils.py
  │      ├─ Initialize model: mimir/models_without_debugging.py → LanguageModel
  │      │  └─ Model uses: mimir/config.py, mimir/custom_datasets.py, mimir/data_utils.py
  │      ├─ Load data from: /content/seen_books.jsonl, /content/unseen_books.jsonl
  │      ├─ Extract features
  │      └─ Save: results_new/.../all_features_custom.pkl
  │
  ├─ Phase 2: 10-15 minutes  
  │  │
  │  └─→ python run_ours_train_lr.py
  │      ├─ Load features from: util_features.py, results_new/.../all_features_custom.pkl
  │      ├─ Train classifier
  │      └─ Save: trained_model_*.pkl, mia_scores_*.json
  │
  └─ Phase 3: 15-20 minutes
     │
     └─→ python run_ours_get_roc.py
         ├─ Load model from: util_features.py, results_new/.../trained_model_*.pkl
         ├─ Generate ROC curves
         └─ Save: roc_curve_plot.png, roc_metrics.json

END: All results in results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
```

---

## Updated Status After Cleanup

### **Files Cleaned** ✅
```
mimir/models.py
  - Removed: from hf_olmo import *
  - Removed: OpenAI config checks
  - Removed: Special model type handling
  - Result: Streamlined for HF models only

mimir/models_without_debugging.py  
  - Removed: from hf_olmo import *
  - Removed: OpenAI config checks  
  - Removed: Special model type handling
  - Result: Streamlined for HF models only
```

### **Imports Now Required** ✅
```
✓ torch, transformers
✓ numpy, scipy, sklearn
✓ omegaconf, hydra
✓ pandas, tqdm
✓ json, pickle, os
```

### **Imports Removed** ✅
```
✗ hf_olmo (was causing import errors)
✗ openai (not needed for HF models)
✗ Custom transformers loaders
✗ Special tokenizer handlers
```

---

## Data Flow Diagram

```
External Inputs
├─ /content/seen_books.jsonl
├─ /content/unseen_books.jsonl  
├─ conf/config.yaml
└─ HuggingFace Model Hub: common-pile/comma-v0.1-2t

        │
        ↓
        
[Phase 1] run_ours_construct_mia_data_custom.py
    ├─ Uses: utils.py, mimir/utils.py
    ├─ Uses: mimir/models_without_debugging.py
    ├─ Uses: mimir/config.py, mimir/custom_datasets.py, mimir/data_utils.py
    └─ Outputs: all_features_custom.pkl

        │
        ↓

[Phase 2] run_ours_train_lr.py
    ├─ Uses: util_features.py
    ├─ Input: all_features_custom.pkl
    └─ Outputs: trained_model_*.pkl, mia_scores_*.json

        │
        ↓

[Phase 3] run_ours_get_roc.py
    ├─ Uses: util_features.py
    ├─ Input: trained_model_*.pkl
    └─ Outputs: roc_curve_plot.png, roc_metrics.json

        │
        ↓
        
Final Results
└─ results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/
   ├─ all_features_custom.pkl
   ├─ trained_model_*.pkl
   ├─ mia_scores_*.json
   ├─ performance_metrics.json
   ├─ roc_metrics.json
   └─ roc_curve_plot.png
```

---

## Command Reference

### **Run Everything**
```bash
bash run_camia_custom.sh
```

### **Run Individual Steps**
```bash
# Step 1 only
python run_ours_construct_mia_data_custom.py base_model=common-pile/comma-v0.1-2t

# Step 2 only  
python run_ours_train_lr.py base_model=common-pile/comma-v0.1-2t

# Step 3 only
python run_ours_get_roc.py base_model=common-pile/comma-v0.1-2t
```

---

## Summary Table

| Phase | Main Script | Duration | Input | Output |
|-------|-----------|----------|-------|--------|
| 1 | run_ours_construct_mia_data_custom.py | 5-10m | JSON books | .pkl features |
| 2 | run_ours_train_lr.py | 2-5m | .pkl features | trained model |
| 3 | run_ours_get_roc.py | 2-3m | trained model | ROC + metrics |

---

## Files NOT Called

```
❌ run_baselines.py                    - Baseline attacks (skipped)
❌ run_ref_baselines.py                - Reference model (skipped)
❌ run_ours_different_agg.py           - Different aggregations (optional)
❌ run_ref_baselines.py                - Reference attacks (skipped)
❌ mimir/attacks/all_attacks.py        - Attack implementations (not used)
❌ mimir/attacks/loss.py               - Loss attack (not used)
❌ mimir/attacks/quantile.py           - Quantile attack (not used)
❌ mimir/attacks/neighborhood.py       - Neighborhood attack (not used)
❌ mimir/attacks/min_k.py              - Min-K attack (not used)
❌ mimir/attacks/min_k_plus_plus.py    - Min-K++ attack (not used)
❌ mimir/attacks/gradnorm.py           - GradNorm attack (not used)
❌ mimir/attacks/zlib.py               - ZLIB attack (not used)
❌ mimir/attacks/reference.py          - Reference attack (not used)
```

---

## Key Points

✅ **Only 3 main scripts run** (no baselines, no extra attacks)  
✅ **Only 11 repo files are actually used** during execution  
✅ **All cleanup complete** (hf_olmo removed, OpenAI code removed)  
✅ **Optimized for HF models** (common-pile/comma-v0.1-2t)  
✅ **Fast execution** (~20 min total)  

Ready for Colab! 🚀
