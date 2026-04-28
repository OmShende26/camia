# Pre-Inference Checklist ✓

Complete this checklist before running the inference script to ensure everything is ready.

## Phase 1: Prerequisites (Before Installation)

- [ ] Python 3.8+ installed
  ```bash
  python --version  # Should show 3.8 or higher
  ```

- [ ] Test dataset prepared in JSONL format
  - [ ] File exists and is readable
  - [ ] Contains `source_file` field (book name)
  - [ ] Contains `text` field (text excerpt)
  - [ ] At least 10 texts for meaningful results
  ```bash
  # Check file size and format
  wc -l your_dataset.jsonl
  head -5 your_dataset.jsonl
  ```

- [ ] Trained model pickle file ready
  - [ ] File exists: `model.pkl`
  - [ ] File size > 1MB (expected: 5-50MB)
  - [ ] Generated from `run_ours_train_lr_paper_custom.py`
  ```bash
  # Verify it's a valid pickle
  python -c "import pickle; pickle.load(open('model.pkl', 'rb'))"
  ```

- [ ] Config file present
  - [ ] File exists: `conf/config.yaml`
  - [ ] Contains `base_model` setting (e.g., "gpt2")
  ```bash
  # Check config format
  cat conf/config.yaml | head -20
  ```

## Phase 2: Installation

- [ ] Installed core dependencies
  ```bash
  pip install torch transformers
  ```

- [ ] Installed project requirements
  ```bash
  cd context_aware_mia
  pip install -r requirements.txt
  ```

- [ ] GPU setup (if using GPU)
  ```bash
  # Verify PyTorch can see GPU
  python -c "import torch; print(torch.cuda.is_available())"
  # Should print: True (if GPU available)
  ```

- [ ] Verified all imports work
  ```bash
  python -c "
  import torch
  import transformers
  import sklearn
  from util_features import collect_all_features
  print('✓ All imports successful')
  "
  ```

## Phase 3: Environment Setup

- [ ] Working directory confirmed
  ```bash
  pwd  # Should show: .../context_aware_mia
  ls -la run_inference_detect_books.py  # Should exist
  ```

- [ ] Config file readable
  ```bash
  python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('conf/config.yaml'); print(cfg)"
  ```

- [ ] Model file can be loaded
  ```bash
  python -c "
  import pickle
  with open('PATH_TO_MODEL.pkl', 'rb') as f:
      model = pickle.load(f)
  print(f'Keys: {list(model.keys())}')
  print('✓ Model loaded successfully')
  "
  ```

- [ ] JSONL file can be loaded
  ```bash
  python -c "
  import json
  with open('PATH_TO_JSONL.jsonl', 'r') as f:
      line = f.readline()
      data = json.loads(line)
      print(f'Fields: {list(data.keys())}')
      assert 'source_file' in data
      assert 'text' in data
  print('✓ JSONL format valid')
  "
  ```

## Phase 4: Testing

- [ ] Run test suite
  ```bash
  python test_inference_pipeline.py
  ```
  Expected output: `✓ All tests passed!`

- [ ] Test on small sample (optional)
  ```bash
  # Extract first 5 lines from your JSONL for testing
  head -5 your_dataset.jsonl > test_small.jsonl
  
  # Run inference on small sample
  python run_inference_detect_books.py \
    --dataset_jsonl test_small.jsonl \
    --model_file your_model.pkl
  ```

## Phase 5: File Preparation

- [ ] JSONL file verified
  ```bash
  # Check format
  python -c "
  import json
  with open('YOUR_FILE.jsonl') as f:
      for i, line in enumerate(f):
          if i >= 3: break
          data = json.loads(line)
          print(f'Line {i+1}: {list(data.keys())}, Text length: {len(data.get(\"text\", \"\"))}')
  "
  ```

- [ ] Dataset statistics checked
  ```bash
  python -c "
  import json
  books = set()
  count = 0
  with open('YOUR_FILE.jsonl') as f:
      for line in f:
          data = json.loads(line)
          books.add(data['source_file'])
          count += 1
  print(f'Total texts: {count}')
  print(f'Unique books: {len(books)}')
  print(f'Books: {sorted(books)[:5]}')  # First 5
  "
  ```

- [ ] Paths verified (absolute preferred)
  ```bash
  # Get absolute paths
  cd context_aware_mia
  pwd  # Project directory
  realpath conf/config.yaml  # Config
  realpath ../your_model.pkl  # Model
  realpath ../dataset.jsonl  # Data
  ```

## Phase 6: Pre-Execution

- [ ] Command line arguments prepared
  ```bash
  # Variables to use (update with your paths):
  DATASET="/full/path/to/dataset.jsonl"
  MODEL="/full/path/to/model.pkl"
  CONFIG="conf/config.yaml"
  THRESHOLD="0.5"
  OUTPUT="results.csv"
  SEED="42"
  ```

- [ ] Full command verified
  ```bash
  echo "Command to run:"
  echo "python run_inference_detect_books.py \\"
  echo "  --dataset_jsonl '$DATASET' \\"
  echo "  --model_file '$MODEL' \\"
  echo "  --config_path '$CONFIG' \\"
  echo "  --threshold '$THRESHOLD' \\"
  echo "  --output_csv '$OUTPUT' \\"
  echo "  --seed '$SEED'"
  ```

- [ ] Disk space verified
  ```bash
  # Need at least 5-10GB free
  df -h  # Unix/Mac
  # or
  Get-Volume  # Windows PowerShell
  ```

- [ ] RAM available
  ```bash
  # At least 8GB recommended
  free -h  # Unix/Mac
  # or
  Get-ComputerInfo -Property TotalPhysicalMemory  # Windows
  ```

## Phase 7: Execution Ready

- [ ] All items above checked ✓

- [ ] Test inference script exists
  ```bash
  ls -la run_inference_detect_books.py
  ```

- [ ] Documentation reviewed
  - [ ] Read QUICKSTART.md
  - [ ] Read INFERENCE_GUIDE.md (for your use case)

- [ ] Support resources noted
  - [ ] SETUP_GUIDE.md (for troubleshooting)
  - [ ] IMPLEMENTATION_SUMMARY.md (for technical details)

## Phase 8: Ready to Run!

```bash
# Run with your actual paths
python run_inference_detect_books.py \
  --dataset_jsonl "path/to/dataset.jsonl" \
  --model_file "path/to/model.pkl" \
  --config_path "conf/config.yaml" \
  --threshold 0.5 \
  --output_csv "mia_results_by_book.csv"
```

Expected output:
```
[INFO] Using device: cuda (or cpu)
[INFO] Loading model: gpt2
[INFO] Loaded X texts from Y unique books
...
======================================================================
CONTAMINATION REPORT - Book-wise Membership Detection
======================================================================
Book: ...
```

## Troubleshooting During Execution

If you encounter errors:

1. **Import error** → Run `pip install -r requirements.txt` again
2. **Config error** → Check `conf/config.yaml` exists and is readable
3. **Model error** → Verify model.pkl is a valid pickle file
4. **JSONL error** → Check format with: `head -5 your_file.jsonl`
5. **Memory error** → Use smaller model or reduce batch size
6. **GPU error** → Switch to CPU: `CUDA_VISIBLE_DEVICES="" python ...`

## After Execution

- [ ] Console output reviewed
  - [ ] No error messages
  - [ ] Per-book results displayed
  - [ ] Summary statistics shown

- [ ] CSV file generated
  ```bash
  ls -la mia_results_by_book.csv  # Should exist
  head -5 mia_results_by_book.csv  # Should show data
  ```

- [ ] Results analyzed
  - [ ] Contamination rates per book noted
  - [ ] Overall statistics reviewed
  - [ ] Any anomalies identified

## Success Criteria ✓

All items completed means:
- ✅ Dependencies installed correctly
- ✅ Files properly formatted and accessible
- ✅ Configuration valid
- ✅ Model loaded successfully
- ✅ Inference script ready to run
- ✅ You understand what to expect

## Quick Reference

```bash
# Copy this checklist items as commands:

# 1. Verify Python
python --version

# 2. Install dependencies
pip install torch transformers
pip install -r requirements.txt

# 3. Verify imports
python test_inference_pipeline.py

# 4. Verify model
python -c "import pickle; pickle.load(open('model.pkl', 'rb')); print('✓')"

# 5. Verify JSONL
python -c "
import json
with open('dataset.jsonl') as f:
    d = json.loads(f.readline())
    print(f'✓ Fields: {list(d.keys())}')
"

# 6. Run inference
python run_inference_detect_books.py \
  --dataset_jsonl dataset.jsonl \
  --model_file model.pkl
```

---

**When all items are checked ✓**, you're ready to run inference!

Next: [Go to QUICKSTART.md](QUICKSTART.md)
