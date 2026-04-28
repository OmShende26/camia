# Implementation Summary: Book-wise Membership Detection

## ✅ What Has Been Implemented

### 1. **Main Inference Script** 
   - **File**: `run_inference_detect_books.py` (700+ lines)
   - **Purpose**: Complete inference pipeline from JSONL data to book-wise results
   - **Key Functions**:
     - `load_jsonl_with_books()` - Load texts with source file preservation
     - `extract_features_with_repeated()` - Generate token probability features
     - `run_inference()` - Execute trained model predictions
     - `aggregate_results_by_book()` - Group results by source book
     - `print_results()` - Format and display results matching your requirements

### 2. **Feature Extraction Pipeline**
   - ✅ Loads JSONL dataset while preserving book names from `source_file` field
   - ✅ Uses `LanguageModel.get_probabilities_with_tokens()` for token-level features
   - ✅ Generates repeated-text variants (5x repetition)
   - ✅ Handles edge cases (short texts, errors)
   - ✅ Compatible with training pipeline

### 3. **Model Inference Engine**
   - ✅ Loads trained model pickle file (validates structure)
   - ✅ Reconstructs feature matrix in exact training order
   - ✅ Applies sign-alignment flip mask
   - ✅ Scales features using trained MinMaxScaler
   - ✅ Runs LogisticRegression predictions
   - ✅ Returns membership probabilities [0, 1]

### 4. **Results Aggregation & Reporting**
   - ✅ Groups predictions by source book name
   - ✅ Computes contamination rate (TPR) per book: `(flagged / total) * 100%`
   - ✅ Displays formatted output matching your attachment:
     ```
     Book: The_Lightning_Thief.txt
     - Contamination Rate (TPR): 8.00% (2/25 excerpts flagged)
     ```
   - ✅ Prints overall summary statistics
   - ✅ Exports detailed CSV results

### 5. **Documentation & Testing**
   - ✅ `SETUP_GUIDE.md` - Installation and dependency setup
   - ✅ `INFERENCE_GUIDE.md` - Complete usage documentation
   - ✅ `test_inference_pipeline.py` - Automated test suite
   - ✅ Updated `requirements.txt` with all dependencies

## 📊 Output Format (Matches Your Requirement)

**Console Output**:
```
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
```

**CSV Output** (`mia_results_by_book.csv`):
```csv
book_name,membership_prob,is_member
The_Lightning_Thief.txt,0.85,1
The_Lightning_Thief.txt,0.23,0
...
```

## 🚀 Usage

### Basic Command
```bash
python run_inference_detect_books.py \
  --dataset_jsonl /path/to/dataset.jsonl \
  --model_file /path/to/trained_model.pkl
```

### With All Options
```bash
python run_inference_detect_books.py \
  --dataset_jsonl d:/dataset/test_books.jsonl \
  --model_file d:/models/trained_model.pkl \
  --config_path conf/config.yaml \
  --threshold 0.5 \
  --output_csv results.csv \
  --seed 42
```

### Before First Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python test_inference_pipeline.py

# 3. Run inference
python run_inference_detect_books.py ...
```

## 📋 Data Requirements

### Input: JSONL Format
Each line is JSON with required fields:
- `source_file` (string): Book name/identifier
- `text` (string): Text excerpt to analyze

**Example**:
```json
{"source_file": "The_Lightning_Thief.txt", "text": "were waiting for us..."}
{"source_file": "The_Lightning_Thief.txt", "text": "melting. Drinking it..."}
{"source_file": "The_Da_Vinci_Code.txt", "text": "from no one!..."}
```

### Required Files
1. **Trained model pickle** - From `run_ours_train_lr_paper_custom.py`
   - Must contain: `model`, `scaler`, `flip_mask`, `final_keys`
2. **Config file** - `conf/config.yaml` with model and environment settings
3. **LanguageModel** - Same model used in training (e.g., "gpt2")

## 🔄 Pipeline Flow

```
Input JSONL
    ↓
[Load with book names] → List of texts + book names
    ↓
[Extract features] → Token probs + repeated variants
    ↓
[Compute signals] → 12+ statistical features per text
    ↓
[Reconstruct matrix] → Match training order + apply flip_mask
    ↓
[Run inference] → Membership probabilities [0, 1]
    ↓
[Aggregate by book] → Group predictions by source file
    ↓
[Report results] → Display TPR per book + overall stats
    ↓
Output CSV + Console Report
```

## 🛠️ Technical Architecture

### Feature Engineering (Phase 1)
- Uses trained LanguageModel to compute token log-probabilities
- Creates 5x repeated text for each input
- Extracts token probabilities from both original and repeated text
- Skips texts with < 3 words or < 5 tokens

### Feature Collection (Phase 2-3)
- **Base Features** (12 types):
  - Loss, Perplexity (at cutoffs: -1, 200, 300)
  - Token Diversity, LZ Complexity (bins: 3, 4, 5)
  - Count Above threshold (thresholds: -1, -2, -3)
  - Calibrated Loss/PPL, Count Mean, Slope, Entropy
  
- **Derived Features**:
  - Difference features: `diff_loss = half1_loss - half2_loss`
  - Slope features (3 cutoffs) + Entropy features (3 cutoffs)
  - Total feature count: ~30+ depending on aggregation

### Inference (Phase 4)
- Reconstructs feature matrix in exact training order (uses `final_keys`)
- Applies sign-alignment mask (flips features where member > non-member)
- Normalizes with trained MinMaxScaler
- Runs LogisticRegression.predict_proba() → membership score [0, 1]
- Default threshold: 0.5 (configurable via CLI)

### Aggregation (Phase 5)
- Groups predictions by `source_file` field
- Per-book metrics: Total texts, Flagged count, TPR %
- Overall summary: Books scanned, Total texts, Global TPR

## 📈 What This Enables

✅ **Detect training data membership** in unknown texts at scale  
✅ **Identify contaminated books** - which source books contain training data?  
✅ **Quantify contamination** - TPR % per book  
✅ **Privacy analysis** - Understand model memorization patterns  
✅ **Data auditing** - Find unintended dataset overlaps  

## 🔧 Advanced Features

- **Configurable threshold** - Adjust detection sensitivity
- **Seed control** - Reproducible results
- **CSV export** - Detailed per-text predictions
- **GPU acceleration** - Auto-detects and uses CUDA if available
- **Error handling** - Robust to malformed data
- **Progress tracking** - TQDMs for long-running operations
- **Flexible configuration** - Uses Hydra + OmegaConf from training

## ⚙️ Configuration

Edit `conf/config.yaml` to adjust:
```yaml
base_model: "gpt2"              # LanguageModel to use
random_seed: 42                  # Reproducibility
max_substrs: 1                   # Full docs vs substrings
full_doc: true                   # Process full text
env_config:
  device_map: false              # Large model device mapping
  cache_dir: "./cache"           # HF model cache
```

## 🐛 Troubleshooting Quick Guide

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install torch transformers` |
| `FileNotFoundError: config` | Use absolute path with `--config_path` |
| `Feature matrix shape mismatch` | Ensure same model as training |
| `CUDA out of memory` | Use `--device CPU` or smaller model |
| `Texts skipped (< 3 words)` | Normal - short texts aren't processed |

See `SETUP_GUIDE.md` for detailed troubleshooting.

## 📞 Files Created

- `run_inference_detect_books.py` - Main inference script (700+ lines)
- `test_inference_pipeline.py` - Automated test suite
- `SETUP_GUIDE.md` - Installation and dependency documentation
- `INFERENCE_GUIDE.md` - Complete usage documentation
- `IMPLEMENTATION_SUMMARY.md` - This file

## ✨ Next Steps

1. **Install dependencies** - Follow SETUP_GUIDE.md
2. **Verify setup** - Run `python test_inference_pipeline.py`
3. **Prepare data** - Ensure JSONL has `source_file` + `text` fields
4. **Have trained model ready** - From `run_ours_train_lr_paper_custom.py`
5. **Run inference** - Execute the command with your paths
6. **Analyze results** - Check console output and CSV

For detailed usage, see `INFERENCE_GUIDE.md`.

---

**Status**: ✅ Implementation Complete  
**Ready to Use**: After dependencies installed  
**Last Updated**: April 29, 2026
