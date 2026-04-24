# CAMIA Custom Dataset Workflow - Simplified Guide

**Modified for:**
- ✅ Custom book dataset from Colab (`/content/seen_books.jsonl`, `/content/unseen_books.jsonl`)
- ✅ No baseline attacks
- ✅ Removed extra imports (olmo, etc.)
- ✅ Model: `common-pile/comma-v0.1-2t`

---

## Data Flow

```
Your Custom Dataset (Colab)
    ↓
    └─→ /content/seen_books.jsonl (members)
    └─→ /content/unseen_books.jsonl (non-members)
    ↓
[run_ours_construct_mia_data_custom.py]
    ↓ Extract token probabilities
    ↓
Feature Vectors (.pkl)
    ↓
[run_ours_train_lr.py]
    ↓ Train classifier
    ↓
MIA Predictions & Scores
    ↓
[run_ours_get_roc.py]
    ↓ Generate ROC curves
    ↓
Results (AUC, TPR/FPR, Plots)
```

---

## Quick Start in Colab

### Option 1: Run All Steps (Recommended)

```bash
cd /path/to/camia/context_aware_mia
bash run_camia_custom.sh
```

### Option 2: Run Individual Steps

```bash
# Step 1: Extract features from both member/non-member datasets
python run_ours_construct_mia_data_custom.py \
    base_model=common-pile/comma-v0.1-2t \
    experiment_name=comma_custom_books

# Step 2: Train classifier
python run_ours_train_lr.py \
    base_model=common-pile/comma-v0.1-2t \
    experiment_name=comma_custom_books

# Step 3: Get ROC curves & final metrics
python run_ours_get_roc.py \
    base_model=common-pile/comma-v0.1-2t \
    experiment_name=comma_custom_books
```

---

## Step-by-Step Explanation

### Step 1: Construct MIA Data (`run_ours_construct_mia_data_custom.py`)

**What it does:**
- Loads your JSONL files from `/content/`
- Extracts token-level probabilities from the model for each text
- Creates repeated text variants (5 repetitions)
- Generates feature vectors for ML training

**Key modifications:**
- Loads from `load_jsonl_dataset()` instead of general data_utils
- Points to `/content/seen_books.jsonl` (members)
- Points to `/content/unseen_books.jsonl` (non-members)
- Skips baseline attack code

**Output:** `results_new/camia_custom_dataset/common-pile_comma-v0.1-2t/all_features_custom.pkl`

**What gets saved:**
```python
{
    "member_preds": [
        {"tk_probs": [...], "labels": [...], "tk_probs_repeated_5": [...], ...},
        ...
    ],
    "nonmember_preds": [
        {"tk_probs": [...], "labels": [...], "tk_probs_repeated_5": [...], ...},
        ...
    ]
}
```

### Step 2: Train Classifier (`run_ours_train_lr.py`)

**What it does:**
- Loads features from Step 1
- Trains Logistic Regression classifier
- Computes MIA scores (membership inference scores)
- Calculates performance metrics (AUC, accuracy, precision, recall)

**Input:** `.pkl` file from Step 1

**Output:** 
- Trained model
- Attack scores for test set
- Performance metrics (saved as JSON/CSV)

### Step 3: Get ROC Curves (`run_ours_get_roc.py`)

**What it does:**
- Loads trained model and predictions
- Generates ROC curves
- Computes various metrics at different FPR thresholds
- Creates visualization plots

**Output:**
- ROC curve plot
- AUC score
- Operating point metrics

---

## File Structure Changes

### New/Modified Files:

```
camia/context_aware_mia/
├── run_camia_custom.sh                  ← NEW: Simplified bash workflow
├── run_ours_construct_mia_data_custom.py ← NEW: Custom dataset version
├── mimir/
│   └── models.py                         ← MODIFIED: Removed hf_olmo import
└── (other files unchanged)
```

### Removed from Workflow:
- ❌ `run_baselines.py` - Not needed
- ❌ `run_ref_baselines.py` - Not needed (no reference attacks)
- ❌ `run_ours_different_agg.py` - Optional (p-value aggregation)

---

## Dataset Format Requirements

Your JSONL files should have this format:

```json
{"text": "This is the first book excerpt..."}
{"text": "This is another book excerpt..."}
{"text": "Another text sample..."}
...
```

**Size requirements (recommended):**
- Each file: 1,000 - 50,000 texts
- Text length: 100-256 words (configurable in config.yaml)
- Equal sizes for member/non-member for best results

---

## Configuration (if needed)

Edit `conf/config.yaml`:

```yaml
base_model: common-pile/comma-v0.1-2t    # ← Your model
experiment_name: comma_custom_books       # ← Your experiment name
max_data: 50000                           # ← Max texts to process
batch_size: 50                            # ← Batch size
min_words: 100                            # ← Min text length
max_words: 200                            # ← Max text length
random_seed: 0                            # ← For reproducibility
```

---

## Output Locations

```
results_new/
├── camia_custom_dataset/
│   └── common-pile_comma-v0.1-2t/
│       ├── all_features_custom.pkl           ← Features from Step 1
│       ├── mia_scores_*.json                 ← Results from Step 2
│       └── roc_curves_*.png                  ← Plots from Step 3
```

---

## Key Metrics from Results

After running all 3 steps, you'll get:

| Metric | Description |
|--------|-------------|
| **AUC** | Area Under ROC Curve (higher=better) |
| **TPR @ FPR=0.001** | True positive rate at 0.1% false positive rate |
| **TPR @ FPR=0.01** | True positive rate at 1% false positive rate |
| **Accuracy** | Percentage of correct classifications |

---

## Troubleshooting

### Issue: "Could not find /content/seen_books.jsonl"
**Solution:** Upload your JSONL files to colab `/content/` directory first

### Issue: Out of memory (OOM)
**Solution:** Reduce `batch_size` or `max_data` in config.yaml

### Issue: "hf_olmo not found"
**Solution:** Already fixed! We removed this import from `models.py`

### Issue: "CUDA out of memory"
**Solution:** Set `CUDA_VISIBLE_DEVICES=0` to use a single GPU or reduce batch size

---

## What's Different from Original?

| Aspect | Original | Modified |
|--------|----------|----------|
| Datasets | The Pile, Wikipedia | Your custom JSONL files |
| Imports | Includes hf_olmo | olmo removed |
| Baselines | Runs 6 scripts | Runs 3 scripts (CAMIA only) |
| Entry point | run.sh | run_camia_custom.sh |
| Workflow | ~45 minutes | ~20 minutes |

---

## Next Steps

1. Prepare your JSONL files in colab: `/content/seen_books.jsonl`, `/content/unseen_books.jsonl`
2. Run: `bash run_camia_custom.sh`
3. Check results in `results_new/` folder
4. Analyze plots and metrics

Happy analyzing! 🚀
