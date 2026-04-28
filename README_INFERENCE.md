# Membership Detection Inference System - README

## 🎯 Objective

Detect and quantify training data membership in unknown text datasets by:
1. Running your texts through a trained Membership Inference Attack (MIA) model
2. Identifying texts that appear to have been in the model's training data
3. Aggregating results **by source book** with contamination rates
4. Displaying results in a structured, book-wise format

## 📦 What's Included

### Main Components
- **run_inference_detect_books.py** - Complete inference pipeline (700+ lines)
- **test_inference_pipeline.py** - Automated validation suite
- **Updated requirements.txt** - All dependencies

### Documentation (Start Here!)
1. **QUICKSTART.md** ⚡ - Get started in 5 minutes
2. **SETUP_GUIDE.md** 🔧 - Installation & dependencies
3. **INFERENCE_GUIDE.md** 📖 - Complete usage documentation
4. **IMPLEMENTATION_SUMMARY.md** 🏗️ - Technical architecture

## 🚀 Quick Start (TL;DR)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Test
python test_inference_pipeline.py

# 3. Run
python run_inference_detect_books.py \
  --dataset_jsonl "path/to/test_books.jsonl" \
  --model_file "path/to/trained_model.pkl"
```

**Done!** Results print to console + saved as CSV.

**[→ Go to QUICKSTART.md for full 5-minute guide](QUICKSTART.md)**

## 📊 Expected Output

### Console Report
```
======================================================================
CONTAMINATION REPORT - Book-wise Membership Detection
======================================================================
Book: The_Lightning_Thief.txt
  - Contamination Rate (TPR): 8.00% (2/25 excerpts flagged)
Book: The_Da_Vinci_Code.txt
  - Contamination Rate (TPR): 12.50% (3/24 excerpts flagged)

======================================================================
SUMMARY
======================================================================
Total Books Scanned: 5
Total Texts Classified: 148
Total Texts Flagged: 18
Overall Contamination Rate: 12.16%
```

### CSV Export
```csv
book_name,membership_prob,is_member
The_Lightning_Thief.txt,0.85,1
The_Lightning_Thief.txt,0.23,0
...
```

## 🔄 How It Works

```
Your Test Data (JSONL)
    ↓
[Load texts + book names]
    ↓
[Extract token probabilities]
    ↓
[Generate repeated text features]
    ↓
[Compute statistical features]
    ↓
[Load trained model]
    ↓
[Run inference → membership scores]
    ↓
[Group results by book]
    ↓
[Calculate contamination rates]
    ↓
Console Report + CSV Export
```

**Detailed architecture**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 📋 Requirements

### Data
- **Trained MIA Model** (pickle file from `run_ours_train_lr_paper_custom.py`)
- **Test Dataset** (JSONL format with `source_file` + `text` fields)
- **Config File** (`conf/config.yaml`)

### System
- **Python 3.8+**
- **PyTorch + Transformers** (auto-installed)
- **8 GB RAM** (minimum)
- **GPU optional** (auto-detected)

### Installation
```bash
pip install -r requirements.txt
# With GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Full setup guide**: [→ SETUP_GUIDE.md](SETUP_GUIDE.md)

## 🎮 Usage

### Basic Command
```bash
python run_inference_detect_books.py \
  --dataset_jsonl "d:/data/test.jsonl" \
  --model_file "d:/models/model.pkl"
```

### With Options
```bash
python run_inference_detect_books.py \
  --dataset_jsonl "d:/data/test.jsonl" \
  --model_file "d:/models/model.pkl" \
  --config_path "conf/config.yaml" \
  --threshold 0.5 \
  --output_csv "results.csv" \
  --seed 42
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_jsonl` | str | **required** | Path to test JSONL |
| `--model_file` | str | **required** | Path to trained model |
| `--config_path` | str | `conf/config.yaml` | Hydra config path |
| `--threshold` | float | `0.5` | Detection threshold |
| `--output_csv` | str | `mia_results_by_book.csv` | Output CSV file |
| `--seed` | int | `42` | Random seed |

**Full documentation**: [→ INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)

## 📝 Data Format

### Input JSONL
Each line is JSON with **required** fields:
```json
{
  "source_file": "BookName.txt",
  "text": "Your text excerpt here..."
}
```

**Example**:
```json
{"source_file": "The_Lightning_Thief.txt", "text": "were waiting for us..."}
{"source_file": "The_Lightning_Thief.txt", "text": "melting. Drinking it..."}
{"source_file": "The_Da_Vinci_Code.txt", "text": "from no one!..."}
```

### Output CSV
```csv
book_name,membership_prob,is_member
The_Lightning_Thief.txt,0.85,1
The_Lightning_Thief.txt,0.23,0
The_Lightning_Thief.txt,0.15,0
The_Da_Vinci_Code.txt,0.92,1
...
```

## 🛠️ Technical Details

### Pipeline Stages

**Stage 1: Feature Extraction**
- Load texts with book names from JSONL
- Use LanguageModel to compute token log-probabilities
- Create 5x repeated text variants
- Extract features: skip texts < 3 words

**Stage 2: Feature Preparation**
- Compute 12+ statistical features:
  - Loss, Perplexity (3 cutoffs)
  - Token Diversity, LZ Complexity
  - Count Above, Count Mean
  - Calibrated Loss/PPL, Slope, Entropy
- Split repeated features into halves → compute differences

**Stage 3: Inference**
- Reconstruct feature matrix in training order
- Apply sign-alignment flip mask
- Scale with trained MinMaxScaler
- Run LogisticRegression.predict_proba() → [0, 1]

**Stage 4: Aggregation**
- Group predictions by `source_file`
- Calculate TPR per book: `(flagged / total) * 100%`
- Print formatted report + export CSV

### Feature Engineering
- **Base Features**: 12 types (loss, diversity, complexity, etc.)
- **Repeated Features**: Computed on 5x repeated text
- **Difference Features**: half1 - half2 for discriminative signals
- **Total Dimensions**: ~30+ features per sample

### Model Architecture
- **Classifier**: Logistic Regression (trained on member/non-member data)
- **Normalization**: MinMaxScaler (trained on training data)
- **Decision Function**: Membership probability = `predict_proba()[:, 1]`
- **Threshold**: Default 0.5 (configurable)

**Full technical overview**: [→ IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 🔍 Interpreting Results

### Contamination Rate
Percentage of texts flagged as training data
- **0%** = No detected contamination
- **5-10%** = Low contamination
- **20%+** = Significant overlap with training set

### Membership Probability
Confidence score [0, 1]
- **0.9-1.0** = Very likely in training data (high confidence)
- **0.7-0.9** = Probably in training data
- **0.3-0.7** = Uncertain
- **0.0-0.3** = Probably NOT in training data
- **0.0-0.1** = Very unlikely in training data (high confidence)

### Example Interpretation
```
Book: Harry_Potter_1.txt
- Contamination Rate: 15.2% (8/52 excerpts flagged)

This means:
✓ 44 excerpts NOT detected as training data (good)
✗ 8 excerpts detected as training data (concern)
→ ~15% of this book appears in training set
```

## ⚡ Performance

### Speed (GPU)
- **Small dataset** (100 texts): ~1-2 minutes
- **Medium dataset** (500 texts): ~5-10 minutes  
- **Large dataset** (5000 texts): ~45-60 minutes

### Speed (CPU)
- **Small dataset**: ~5 minutes
- **Medium dataset**: ~15-30 minutes
- **Large dataset**: ~3-4 hours

### Memory Usage
- **GPU**: 1-3 GB (depends on model size)
- **CPU**: 8-16 GB

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: torch` | `pip install torch transformers` |
| `FileNotFoundError: config` | Use `--config_path` with absolute path |
| `Feature shape mismatch` | Same LanguageModel as training |
| `CUDA out of memory` | Smaller model or use CPU |
| `Texts skipped (< 3 words)` | Normal - short texts can't be processed |

**Detailed troubleshooting**: [→ SETUP_GUIDE.md](SETUP_GUIDE.md)

## 📚 Documentation Structure

```
📦 context_aware_mia/
├── run_inference_detect_books.py      ← Main script
├── test_inference_pipeline.py         ← Test suite
├── requirements.txt                   ← Dependencies
└── 📖 Documentation/
    ├── QUICKSTART.md                 ← Start here! (5 min)
    ├── SETUP_GUIDE.md                ← Installation
    ├── INFERENCE_GUIDE.md            ← Full usage
    ├── IMPLEMENTATION_SUMMARY.md      ← Architecture
    └── README.md                      ← This file
```

## 🎯 Workflow

### First Time Setup
1. Read **QUICKSTART.md** (5 minutes)
2. Run **Setup** section from QUICKSTART
3. Run **Test** section - verify setup works
4. Read **SETUP_GUIDE.md** for troubleshooting

### Running Inference
1. Prepare JSONL dataset with `source_file` + `text`
2. Have trained model pickle ready
3. Run single command from **QUICKSTART.md**
4. Review console output + CSV results

### Advanced Usage
- See **INFERENCE_GUIDE.md** for all options
- See **IMPLEMENTATION_SUMMARY.md** for technical details
- Adjust `--threshold`, `--seed`, config file as needed

## 🔗 Integration

### Training Pipeline
```
run_ours_construct_mia_data_custom.py
    ↓ (generates features)
run_ours_train_lr_paper_custom.py
    ↓ (trains model)
model.pkl
    ↓ (used by)
run_inference_detect_books.py ← YOU ARE HERE
    ↓
Report & CSV
```

### Data Flow
```
JSONL Test Data
    ↓
run_inference_detect_books.py
    ├── [Loads] dataset + model
    ├── [Extracts] token features
    ├── [Runs] inference
    ├── [Groups] by book
    └── [Reports] results
```

## ✨ Features

✅ **Complete inference pipeline** - From data to results  
✅ **Book-wise aggregation** - Grouped by source  
✅ **Formatted output** - Console + CSV  
✅ **GPU acceleration** - Auto-detects CUDA  
✅ **Configurable** - Threshold, seed, model choice  
✅ **Well-documented** - 4 comprehensive guides  
✅ **Tested** - Automated test suite  
✅ **Error-resilient** - Handles edge cases  

## 🎓 Learning Resources

### Understanding MIA
- Understand what Membership Inference Attack is
- Learn how token probabilities reveal training membership
- Study feature engineering approach (loss, diversity, etc.)

### Understanding This Implementation
1. Start with **QUICKSTART.md** - See the big picture
2. Read **IMPLEMENTATION_SUMMARY.md** - Learn architecture
3. Run **test_inference_pipeline.py** - Verify setup
4. Inspect **run_inference_detect_books.py** - Study code

## 💡 Tips & Tricks

### For Best Results
- Use same LanguageModel as training
- Ensure texts are >= 3 words
- Adjust threshold based on your risk tolerance
- Run multiple times with different seeds for robustness

### For Troubleshooting
- Always run `test_inference_pipeline.py` first
- Check that `conf/config.yaml` exists
- Verify trained model pickle is valid
- Use absolute paths for file arguments

### For Speed
- Use GPU if available (100x faster)
- Smaller models (GPT2 vs GPT3) are faster
- Batch process large datasets
- Use `--seed` for reproducibility

## 📞 Support

### Getting Help
1. Check **SETUP_GUIDE.md** - 90% of issues are covered
2. Read **INFERENCE_GUIDE.md** - For usage questions
3. Review **IMPLEMENTATION_SUMMARY.md** - For technical questions
4. Check **test_inference_pipeline.py** output - For validation

### Common Questions
- **Q: Why are some texts skipped?**  
  A: Texts with < 3 words or < 5 tokens are too short for reliable features

- **Q: What does contamination rate mean?**  
  A: % of texts detected as in training data. Higher = more overlap.

- **Q: Can I change the threshold?**  
  A: Yes! Use `--threshold 0.7` for stricter detection

- **Q: Does it work on CPU?**  
  A: Yes, but slower. GPU recommended for large datasets.

## 🎉 You're Ready!

→ **[Start with QUICKSTART.md](QUICKSTART.md)** for a 5-minute walkthrough

Or jump to:
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Full documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details

---

**Status**: ✅ Ready to Use  
**Last Updated**: April 29, 2026  
**Components**: 6 files + 4 guides  
**Next Step**: Install dependencies & run tests
