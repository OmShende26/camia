# Setup & Installation Guide

## Prerequisites

Before running the inference pipeline, ensure you have:

1. **Python 3.8+** - Check with: `python --version`
2. **PyTorch** - For deep learning model execution
3. **Transformers** - For LLM tokenization and inference

## Installation Steps

### Option A: Install All Dependencies (Recommended)

```bash
# Navigate to the project directory
cd context_aware_mia

# Install all required packages
pip install -r requirements.txt
```

**Note**: PyTorch installation may require GPU-specific packages. If you have CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option B: Manual Installation (if needed)

```bash
# Core dependencies
pip install torch transformers scikit-learn pandas numpy

# Additional tools
pip install hydra-core omegaconf tqdm scikit-learn matplotlib nltk

# Advanced features
pip install simple-parsing zstandard ninja
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "from sklearn import linear_model; print('scikit-learn OK')"
```

All three commands should output successfully without errors.

## Quick Verification

```bash
# Run the test suite
python test_inference_pipeline.py
```

Expected output:
```
======================================================================
INFERENCE PIPELINE TEST SUITE
======================================================================
[TEST 1] Checking imports...
✓ All required imports successful
[TEST 2] Checking config file...
✓ Config file exists at conf/config.yaml
[TEST 3] Testing JSONL loading...
✓ Successfully loaded 3 texts from 2 books
[TEST 4] Checking script structure...
✓ All required functions found in script

======================================================================
SUMMARY: 4/4 tests passed
======================================================================

✓ All tests passed! Ready to run inference.
```

## GPU Acceleration (Optional but Recommended)

### Check if GPU is Available

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Using GPU

The script automatically detects and uses GPU if available. To verify it's being used:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.current_device())"
```

To force CPU usage (if needed):

```bash
CUDA_VISIBLE_DEVICES="" python run_inference_detect_books.py ...
```

## Troubleshooting Installation

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution 1**: Install PyTorch correctly for your system
```bash
# Check CUDA version first (if you have GPU)
nvidia-smi
# Then install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu<YOUR_CUDA_VERSION>
```

**Solution 2**: Install for CPU only (simpler, slower)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: `ModuleNotFoundError: No module named 'transformers'`

```bash
pip install transformers
```

### Issue: Version Conflicts

```bash
# Create a fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: CUDA/GPU Not Working

```bash
# Verify PyTorch was installed with CUDA support
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# If False, reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## System Requirements

### Minimum
- **RAM**: 8 GB
- **Storage**: 5 GB (for models)
- **Processor**: Any modern CPU

### Recommended for Faster Inference
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 4060, A100, etc.)
- **CUDA**: 11.8 or compatible version
- **RAM**: 16 GB

### For Large Models (GPT3/Llama)
- **GPU**: 24GB+ VRAM (A100, RTX 4090, etc.)
- **RAM**: 32 GB
- **Storage**: 20+ GB

## Model Size Reference

| Model | GPU Memory | CPU Memory | Inference Speed (100 texts) |
|-------|-----------|-----------|----------------------------|
| gpt2 | 1 GB | 3 GB | ~1 min |
| gpt2-medium | 2 GB | 5 GB | ~2 min |
| gpt2-large | 3 GB | 8 GB | ~3 min |
| distilbert-base | 1 GB | 2 GB | ~30 sec |
| llama-7b | 16 GB | 28 GB | ~30 min |

## Next Steps

After successful installation:

1. ✅ Verify: `python test_inference_pipeline.py`
2. ✅ Prepare your test JSONL dataset
3. ✅ Run inference: `python run_inference_detect_books.py --dataset_jsonl <file> --model_file <model.pkl>`
4. ✅ Check results in console and `mia_results_by_book.csv`

## Support

For issues:
1. Check that all tests pass: `python test_inference_pipeline.py`
2. Verify config file exists: `ls conf/config.yaml`
3. Check model file exists: `ls <path_to_model.pkl>`
4. Review console output for specific error messages

## Advanced Configuration

### Using Different Models

Edit `conf/config.yaml`:
```yaml
base_model: "gpt2"  # Change to any model on Hugging Face Hub
env_config:
  device_map: false  # Use device_map=true for large models with accelerate
```

### Custom Cache Directory

```bash
python run_inference_detect_books.py ... --cache_dir /custom/cache/path
```

See `INFERENCE_GUIDE.md` for full usage instructions after setup is complete.
