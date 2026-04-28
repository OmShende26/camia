# Inference & Book-wise Detection Guide

## Overview
The `run_inference_detect_books.py` script runs the trained Context-Aware MIA model on a test dataset and reports membership detection results grouped by source book.

## Prerequisites
1. **Trained Model Pickle File** - From `run_ours_train_lr_paper_custom.py`
2. **Test Dataset (JSONL)** - Format: `{"source_file": "BookName.txt", "text": "..."}`
3. **Configuration File** - `conf/config.yaml` with model and environment settings

## Quick Start

### Basic Usage
```bash
python run_inference_detect_books.py \
  --dataset_jsonl /path/to/test_dataset.jsonl \
  --model_file /path/to/trained_model.pkl \
  --config_path conf/config.yaml
```

### Full Example with All Options
```bash
python run_inference_detect_books.py \
  --dataset_jsonl d:/path/to/dataset.jsonl \
  --model_file d:/path/to/model.pkl \
  --config_path conf/config.yaml \
  --threshold 0.5 \
  --output_csv results.csv \
  --seed 42
```

## Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_jsonl` | str | **required** | Path to input JSONL file |
| `--model_file` | str | **required** | Path to trained model pickle |
| `--config_path` | str | `conf/config.yaml` | Path to experiment config |
| `--threshold` | float | `0.5` | Membership probability threshold (0-1) |
| `--output_csv` | str | `mia_results_by_book.csv` | Output CSV filename |
| `--seed` | int | `42` | Random seed for reproducibility |

## Expected Output

### Console Output
```
[INFO] Using device: cuda
[INFO] Loading model: gpt2
[INFO] Loading dataset from /path/to/dataset.jsonl...
[INFO] Loaded 150 texts from 5 unique books

[STEP 1/4] Extracting token probability features...
Extracting token features: 100%|███| 150/150
[INFO] Successfully extracted features for 148 texts

[STEP 2/4] Running inference with trained model...
[INFO] Running inference on 148 samples...
[INFO] Collecting base features...
[INFO] Processing repeated text features...
[INFO] Computing slope and entropy features...
[INFO] Reconstructing feature matrix...
[INFO] Making predictions...

[STEP 3/4] Aggregating results by source book...
[INFO] Aggregated results for 5 unique books

[STEP 4/4] Displaying results...

======================================================================
CONTAMINATION REPORT - Book-wise Membership Detection
======================================================================
Book: The_Lightning_Thief.txt
  - Contamination Rate (TPR): 8.00% (2/25 excerpts flagged)
Book: The_Da_Vinci_Code.txt
  - Contamination Rate (TPR): 12.50% (3/24 excerpts flagged)
...

======================================================================
SUMMARY
======================================================================
Total Books Scanned: 5
Total Texts Classified: 148
Total Texts Flagged: 18
Overall Contamination Rate: 12.16%
======================================================================
```

### CSV Output (`mia_results_by_book.csv`)
```csv
book_name,membership_prob,is_member
The_Lightning_Thief.txt,0.85,1
The_Lightning_Thief.txt,0.23,0
The_Lightning_Thief.txt,0.92,1
...
```

## Data Format

### Input JSONL Format
Each line must be valid JSON with:
- `source_file` (string): Book name/identifier (e.g., "Harry_Potter_1.txt")
- `text` (string): Text excerpt to analyze

**Example:**
```json
{"source_file": "The_Lightning_Thief.txt", "text": "were waiting for us. I suppose it's too late to turn back..."}
{"source_file": "The_Lightning_Thief.txt", "text": "melting. Drinking it, my whole body felt warm and good..."}
{"source_file": "The_Da_Vinci_Code.txt", "text": "from no one! I don't have dreams!..."}
```

## Pipeline Steps

### Step 1: Feature Extraction
- Loads each text from JSONL
- Uses LanguageModel to compute token log-probabilities
- Creates 5x repeated version of each text
- Extracts token probs from repeated text (for computing differences)
- **Output**: Arrays of token probabilities + labels

### Step 2: Feature Preparation
- Computes 12+ statistical features from token probs:
  - Loss, Perplexity, Token Diversity
  - Lempel-Ziv Complexity, Count Above Threshold
  - Count Mean, Calibrated Loss/PPL, Slope, Entropy
- Splits repeated-text features into halves
- Computes differences: `diff_feature = half1 - half2`

### Step 3: Inference
- Reconstructs feature matrix in **same order as training**
- Applies flip mask (sign alignment from training)
- Scales features using trained scaler
- Runs logistic regression: `probs = model.predict_proba(X)`
- **Output**: Membership probability [0, 1] for each text

### Step 4: Aggregation & Reporting
- Groups predictions by source book
- Calculates contamination rate (TPR) per book: `(flagged / total) * 100%`
- Prints formatted results and summary statistics
- Exports detailed CSV with all predictions

## Model Requirements

The trained model pickle must contain:
```python
{
    "model": LogisticRegression(...),      # Fitted classifier
    "scaler": MinMaxScaler(...),           # Feature normalization
    "flip_mask": np.array([1, -1, 1, ...]), # Sign alignment mask
    "final_keys": ["loss", "diff_loss", ...], # Feature order
    "auc_test": 0.85,                      # Metadata (optional)
    "auc_train": 0.92,
    ...
}
```

## Troubleshooting

### Error: `FileNotFoundError: No such file or directory: 'conf/config.yaml'`
**Solution**: Ensure config file exists. Use `--config_path` with absolute path:
```bash
python run_inference_detect_books.py \
  --dataset_jsonl data.jsonl \
  --model_file model.pkl \
  --config_path d:/path/to/conf/config.yaml
```

### Error: `ValueError: Feature matrix shape mismatch`
**Cause**: Trained model expects different number of features
**Solution**: Ensure test data uses same LanguageModel as training (same `base_model` in config)

### Warning: `Skipping very short text (< 3 words)`
**Cause**: Some texts have fewer than 3 words
**Action**: Normal - texts too short for reliable feature extraction are skipped

### Error: `CUDA out of memory`
**Solution**: Reduce batch operations or use CPU:
```bash
CUDA_VISIBLE_DEVICES="" python run_inference_detect_books.py ...
```

## Performance Notes

- **Time**: ~100 texts/min on GPU (depends on text length & model size)
- **Memory**: ~8GB GPU for GPT2, ~24GB for larger models
- **Storage**: Output CSV is ~500 bytes per prediction

## Integration with Training Pipeline

```
Training:
  run_ours_construct_mia_data_custom.py → features.pkl
                    ↓
  run_ours_train_lr_paper_custom.py → trained_model.pkl
                    ↓
Inference:
  run_inference_detect_books.py (uses trained_model.pkl)
                    ↓
  Output: Book-wise contamination report
```

## Next Steps

1. Run inference: `python run_inference_detect_books.py ...`
2. Review console output and summary statistics
3. Analyze CSV file for detailed per-text predictions
4. Adjust threshold if needed (e.g., `--threshold 0.7` for stricter detection)

## Questions?
Check the code comments in `run_inference_detect_books.py` or refer to the training scripts for context.
