# Quick Start Guide (5 Minutes)

## What You'll Do
Detect which texts in your dataset were in the training set, grouped by source book.

## Prerequisites
- ✅ Trained model pickle file (from `run_ours_train_lr_paper_custom.py`)
- ✅ Test dataset in JSONL format (texts + book names)
- ✅ 5-15 minutes for installation + inference

## Step 1: Install Dependencies (3 min)

```bash
cd context_aware_mia
pip install -r requirements.txt
```

**With GPU (CUDA 11.8)**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Step 2: Verify Setup (1 min)

```bash
python test_inference_pipeline.py
```

Expected: `✓ All tests passed! Ready to run inference.`

## Step 3: Run Inference (1-5 min depending on dataset size)

**Basic**:
```bash
python run_inference_detect_books.py \
  --dataset_jsonl "d:/path/to/your/dataset.jsonl" \
  --model_file "d:/path/to/trained_model.pkl"
```

**Example with actual paths**:
```bash
python run_inference_detect_books.py \
  --dataset_jsonl "d:/data/test_books.jsonl" \
  --model_file "d:/models/model_gpt2.pkl"
```

**Custom options**:
```bash
python run_inference_detect_books.py \
  --dataset_jsonl "d:/data/test_books.jsonl" \
  --model_file "d:/models/model_gpt2.pkl" \
  --threshold 0.5 \
  --output_csv "my_results.csv" \
  --seed 42
```

## Step 4: Read Results

**Console Output** - Shows per-book contamination rates:
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
======================================================================
```

**CSV File** - `mia_results_by_book.csv`:
```
book_name,membership_prob,is_member
The_Lightning_Thief.txt,0.85,1
The_Lightning_Thief.txt,0.23,0
...
```

## Data Format

Your JSONL file should look like:
```json
{"source_file": "Book1.txt", "text": "First text excerpt here..."}
{"source_file": "Book1.txt", "text": "Second text excerpt here..."}
{"source_file": "Book2.txt", "text": "Text from another book..."}
```

## What's Happening Behind the Scenes

1. **Load** - Read JSONL with book names
2. **Extract Features** - Compute token probabilities from your model
3. **Inference** - Use trained model to predict: member or non-member?
4. **Group** - Organize results by source book
5. **Report** - Show contamination rate per book

## Troubleshooting

**Error: `No module named 'torch'`**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Error: `FileNotFoundError: conf/config.yaml`**
```bash
# Ensure you're in the right directory
cd context_aware_mia
python run_inference_detect_books.py ...
```

**Error: `Feature matrix shape mismatch`**
- Make sure you're using the **same LanguageModel** as training
- Check `conf/config.yaml` - `base_model` should match

**GPU not working?**
```bash
# Use CPU instead (slower but works)
CUDA_VISIBLE_DEVICES="" python run_inference_detect_books.py ...
```

## Next Steps

- 📖 Read [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) for advanced options
- ⚙️ Read [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup
- 📋 Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details

## What the Numbers Mean

**Contamination Rate (TPR)**: % of texts from this book that the model flagged as training data
- **0%** = No texts detected as from training set
- **50%** = Half the texts detected  
- **100%** = All texts detected as training data

**membership_prob**: Confidence score [0-1]
- **Close to 1.0** = Very likely in training data
- **Close to 0.5** = Uncertain
- **Close to 0.0** = Very likely NOT in training data

## Typical Results

After inference, you'll see:
- ✅ Per-book contamination rates
- ✅ Total texts scanned
- ✅ Number of flagged texts
- ✅ Overall contamination percentage
- ✅ Detailed CSV export

---

**That's it!** 5 minutes to detect training data membership in your book dataset.

Need help? Check the full docs in [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md).
