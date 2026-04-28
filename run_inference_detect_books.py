"""
run_inference_detect_books.py

Detect training data membership for input texts and report results grouped by source book.

Pipeline:
1. Load JSONL dataset (format: {"source_file": "BookName.txt", "text": "..."})
2. Generate token probability features using LanguageModel
3. Load trained MIA model from pickle
4. Run inference on extracted features
5. Aggregate and display results by source book
"""

import torch
import json
import pickle
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple

from util_features import (
    collect_all_features,
    get_slope,
    approximate_entropy,
)
from mimir.utils import fix_seed
from mimir.models_without_debugging import LanguageModel
from mimir.config import ExperimentConfig
from omegaconf import DictConfig

warnings.filterwarnings("ignore")


def load_jsonl_with_books(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """
    Load texts and book names from JSONL file.
    Expected format: {"source_file": "BookName.txt", "text": "..."}
    
    Returns:
        Tuple of (texts_list, book_names_list)
    """
    texts = []
    book_names = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    texts.append(obj.get('text', ''))
                    book_names.append(obj.get('source_file', 'Unknown'))
    except Exception as e:
        print(f"Error loading {jsonl_path}: {e}")
        raise
    
    return texts, book_names


def extract_features_with_repeated(
    texts: List[str],
    target_model: LanguageModel,
    config: ExperimentConfig,
    batch_size: int = 10,
) -> Tuple[List[List[float]], List[List[int]], List[List[float]]]:
    """
    Extract token probabilities and repeated-text features from texts.
    
    Returns:
        Tuple of (token_probs_list, labels_list, token_probs_repeated_list)
    """
    num_repeatitions = 5
    fix_seed(config.random_seed)
    
    x_all = []
    label_all = []
    x_repeated_all = []
    
    for text in tqdm(texts, desc="Extracting token features"):
        # Skip very short texts
        if len(text.split()) < 3:
            print(f"[WARN] Skipping very short text (< 3 words)")
            continue
        
        try:
            # Extract token probabilities from original text
            s_tk_probs, s_all_probs, labels = (
                target_model.get_probabilities_with_tokens(
                    text, return_all_probs=True
                )
            )
            
            # Convert to list if necessary
            s_tk_probs = s_tk_probs.tolist() if hasattr(s_tk_probs, 'tolist') else list(s_tk_probs)
            labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
            
            # Skip if not enough tokens
            if len(s_tk_probs) < 5:
                print(f"[WARN] Skipping text with too few tokens ({len(s_tk_probs)})")
                continue
            
            x_all.append(s_tk_probs)
            label_all.append(labels)
            
            # Extract from repeated text
            all_str = text
            for r_idx in range(num_repeatitions):
                all_str = all_str + " " + text
            
            s_tk_probs_rep, s_all_probs_rep, labels_rep = (
                target_model.get_probabilities_with_tokens(
                    all_str, return_all_probs=True
                )
            )
            s_tk_probs_rep = s_tk_probs_rep.tolist() if hasattr(s_tk_probs_rep, 'tolist') else list(s_tk_probs_rep)
            x_repeated_all.append(s_tk_probs_rep)
            
        except Exception as e:
            print(f"[ERROR] Failed to extract features: {e}")
            continue
    
    return x_all, label_all, x_repeated_all


def extract_rep_half_split(x_repeated_all: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Split repeated text features into two halves."""
    half1, half2 = [], []
    for probs in x_repeated_all:
        mid = len(probs) // 2
        half1.append(probs[:mid])
        half2.append(probs[mid:])
    return half1, half2


def run_inference(
    x_all: List[List[float]],
    label_all: List[List[int]],
    x_repeated_all: List[List[float]],
    trained_model_path: str,
) -> np.ndarray:
    """
    Run inference using trained model on extracted features.
    
    Returns:
        Array of membership probabilities [0, 1]
    """
    with open(trained_model_path, "rb") as f:
        trained = pickle.load(f)
    
    num_sample = len(x_all)
    
    print(f"\n[INFO] Running inference on {num_sample} samples...")
    
    # Extract all features
    print("[INFO] Collecting base features...")
    base_feats = collect_all_features(x_all, label_all)
    
    # Split repeated features
    print("[INFO] Processing repeated text features...")
    h1, h2 = extract_rep_half_split(x_repeated_all)
    fh1 = collect_all_features(h1, label_all)
    fh2 = collect_all_features(h2, label_all)
    
    # Compute slope and entropy
    print("[INFO] Computing slope and entropy features...")
    cut_offs = [100, 200, 300]
    x_slope = np.column_stack([get_slope(x_repeated_all, end_time=c) for c in cut_offs])
    x_entropy = np.column_stack([
        np.array([approximate_entropy(p[:c], 8, 0.8) for p in x_repeated_all])
        for c in cut_offs
    ])
    
    # Build feature map
    feature_map = {}
    
    def _to_2d(val):
        a = np.array(val).reshape(num_sample, -1) if np.array(val).ndim == 1 else np.array(val)
        return a if a.shape[0] == num_sample else a.T
    
    for key in sorted(base_feats.keys()):
        if key == "find_t":
            continue
        feature_map[key] = _to_2d(base_feats[key])
        if key != "token_diversity" and key in fh1:
            feature_map[f"diff_{key}"] = _to_2d(fh1[key]) - _to_2d(fh2[key])
    
    feature_map["slope"] = _to_2d(x_slope)
    feature_map["entropy"] = _to_2d(x_entropy)
    
    # Reconstruct feature matrix in training order
    print("[INFO] Reconstructing feature matrix...")
    ordered_keys = trained["final_keys"]
    x_attack = np.concatenate([feature_map[k] for k in ordered_keys], axis=1)
    
    # Apply flip mask (sign alignment from training)
    x_attack *= trained["flip_mask"]
    x_attack = np.nan_to_num(x_attack)
    
    # Predict
    print("[INFO] Making predictions...")
    scaler = trained["scaler"]
    model = trained["model"]
    
    probs = model.predict_proba(scaler.transform(x_attack))[:, 1]
    
    return probs


def aggregate_results_by_book(
    probs: np.ndarray,
    book_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict]:
    """
    Aggregate detection results by source book.
    
    Returns:
        Dictionary with book names as keys and result dicts as values
    """
    results = defaultdict(lambda: {"total": 0, "flagged": 0, "probs": []})
    
    for prob, book_name in zip(probs, book_names):
        results[book_name]["total"] += 1
        results[book_name]["probs"].append(prob)
        if prob >= threshold:
            results[book_name]["flagged"] += 1
    
    return results


def print_results(
    results: Dict[str, Dict],
    threshold: float = 0.5,
) -> None:
    """
    Print detection results in formatted output matching the provided format.
    """
    print("\n" + "="*70)
    print("CONTAMINATION REPORT - Book-wise Membership Detection")
    print("="*70)
    
    total_texts = 0
    total_flagged = 0
    
    # Sort by book name for consistent output
    for book_name in sorted(results.keys()):
        result = results[book_name]
        total = result["total"]
        flagged = result["flagged"]
        tpr = (flagged / total * 100) if total > 0 else 0.0
        
        print(f"Book: {book_name}")
        print(f"  - Contamination Rate (TPR): {tpr:.2f}% ({flagged}/{total} excerpts flagged)")
        
        total_texts += total
        total_flagged += flagged
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    overall_tpr = (total_flagged / total_texts * 100) if total_texts > 0 else 0.0
    print(f"Total Books Scanned: {len(results)}")
    print(f"Total Texts Classified: {total_texts}")
    print(f"Total Texts Flagged: {total_flagged}")
    print(f"Overall Contamination Rate: {overall_tpr:.2f}%")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Detect training data membership in unknown texts (grouped by source book)"
    )
    parser.add_argument(
        "--dataset_jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file with texts (format: {\"source_file\": \"Book.txt\", \"text\": \"...\"})"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to trained MIA model pickle file"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="conf/config.yaml",
        help="Path to experiment config"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Membership probability threshold for flagging texts (default: 0.5)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="mia_results_by_book.csv",
        help="Output CSV file with detailed results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    fix_seed(args.seed)
    
    # Load config from YAML
    from hydra import compose, initialize_config_dir
    config_dir = os.path.abspath(os.path.dirname(args.config_path))
    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name=config_name)
    
    # Load model
    print(f"[INFO] Loading model: {config.base_model}")
    base_model = LanguageModel(config)
    if not config.env_config.device_map:
        base_model.to(device)
    
    # Load dataset
    print(f"[INFO] Loading dataset from {args.dataset_jsonl}...")
    texts, book_names = load_jsonl_with_books(args.dataset_jsonl)
    print(f"[INFO] Loaded {len(texts)} texts from {len(set(book_names))} unique books")
    
    # Extract features
    print("\n[STEP 1/4] Extracting token probability features...")
    x_all, label_all, x_repeated_all = extract_features_with_repeated(
        texts,
        base_model,
        config,
    )
    print(f"[INFO] Successfully extracted features for {len(x_all)} texts")
    
    # Filter book_names to match extracted features (some texts may be skipped)
    valid_indices = []
    current_text_idx = 0
    for orig_idx, text in enumerate(texts):
        if len(text.split()) >= 3:
            try:
                # Check if this text was successfully extracted
                # (heuristic: compare with feature extraction loop)
                if current_text_idx < len(x_all):
                    valid_indices.append(orig_idx)
                    current_text_idx += 1
            except:
                pass
    
    # More robust filtering: rebuild mapping
    texts_valid = []
    books_valid = []
    for text, book_name in zip(texts, book_names):
        if len(text.split()) >= 3:
            texts_valid.append(text)
            books_valid.append(book_name)
    
    # Adjust book_names to match extracted features count
    if len(books_valid) > len(x_all):
        books_valid = books_valid[:len(x_all)]
    
    print(f"[INFO] Using {len(books_valid)} valid text-book pairs for inference")
    
    # Run inference
    print("\n[STEP 2/4] Running inference with trained model...")
    probs = run_inference(
        x_all,
        label_all,
        x_repeated_all,
        args.model_file,
    )
    print(f"[INFO] Inference complete. Generated {len(probs)} predictions")
    
    # Aggregate results by book
    print("\n[STEP 3/4] Aggregating results by source book...")
    results = aggregate_results_by_book(
        probs,
        books_valid,
        threshold=args.threshold,
    )
    print(f"[INFO] Aggregated results for {len(results)} unique books")
    
    # Print results
    print("\n[STEP 4/4] Displaying results...")
    print_results(results, threshold=args.threshold)
    
    # Save detailed results to CSV
    print(f"[INFO] Saving detailed results to {args.output_csv}...")
    detailed_results = []
    for prob, book_name in zip(probs, books_valid):
        detailed_results.append({
            "book_name": book_name,
            "membership_prob": prob,
            "is_member": 1 if prob >= args.threshold else 0,
        })
    
    df = pd.DataFrame(detailed_results)
    df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Saved {len(df)} predictions to {args.output_csv}")
    
    print("\n[SUCCESS] Inference and analysis complete!")


if __name__ == "__main__":
    main()
