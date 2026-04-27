"""
run_inference_mia.py

Use a trained Context-Aware MIA model to predict membership on unknown data.
Now supports Evaluation Mode (calculates AUC/TPR) if ground-truth labels are present.
"""

import pickle
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from util_features import (
    collect_all_features,
    get_slope,
    approximate_entropy,
)

warnings.filterwarnings("ignore")

def load_custom_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    m_src = data.get("member_preds", [])
    nm_src = data.get("nonmember_preds", [])
    
    def _probs(sample_dict, key):
        v = sample_dict.get(key, [])
        if not v: return None
        entry = v[0]
        return entry.tolist() if hasattr(entry, "tolist") else list(entry)

    def _labels(sample_dict):
        v = sample_dict.get("labels", [])
        if not v: return []
        entry = v[0]
        if hasattr(entry, "squeeze"): entry = entry.squeeze()
        return entry.tolist() if hasattr(entry, "tolist") else list(entry)

    all_src = m_src + nm_src
    rep_key = next((k for k in ["tk_probs_repeated_5", "tk_probs_repeated_10"] if all_src and k in all_src[0]), None)
    
    x_all, label_all, x_repeated_all, y_true = [], [], [], []
    
    # Process Members (Label 1)
    for sample in m_src:
        probs, labels, rep = _probs(sample, "tk_probs"), _labels(sample), _probs(sample, rep_key)
        if probs is None or rep is None or len(probs) < 5: continue
        x_all.append(probs); label_all.append(labels); x_repeated_all.append(rep); y_true.append(1)
        
    # Process Non-Members (Label 0)
    for sample in nm_src:
        probs, labels, rep = _probs(sample, "tk_probs"), _labels(sample), _probs(sample, rep_key)
        if probs is None or rep is None or len(probs) < 5: continue
        x_all.append(probs); label_all.append(labels); x_repeated_all.append(rep); y_true.append(0)

    return x_all, label_all, x_repeated_all, np.array(y_true)

def extract_rep_half_split(x_repeated_all):
    half1, half2 = [], []
    for probs in x_repeated_all:
        mid = len(probs) // 2
        half1.append(probs[:mid]); half2.append(probs[mid:])
    return half1, half2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, required=True, help="Path to book features .pkl")
    parser.add_argument("--model_file", type=str, required=True, help="Trained .pkl from WikiMIA run")
    args = parser.parse_args()

    with open(args.model_file, "rb") as f:
        trained = pickle.load(f)
    
    x_all, label_all, x_rep, y_true = load_custom_pickle(args.features_file)
    num_sample = len(x_all)
    num_members = int(np.sum(y_true))

    print(f"Loaded {num_sample} samples ({num_members} members, {num_sample - num_members} non-members).")

    print("Extracting signals...")
    base_feats = collect_all_features(x_all, label_all)
    h1, h2 = extract_rep_half_split(x_rep)
    fh1, fh2 = collect_all_features(h1, label_all), collect_all_features(h2, label_all)

    # Fixed cut-offs to match training
    cut_offs = [100, 200, 300]
    x_slope = np.column_stack([get_slope(x_rep, end_time=c) for c in cut_offs])
    x_entropy = np.column_stack([np.array([approximate_entropy(p[:c], 8, 0.8) for p in x_rep]) for c in cut_offs])

    feature_map = {}
    def _to_2d(val):
        a = np.array(val).reshape(num_sample, -1) if np.array(val).ndim == 1 else np.array(val)
        return a if a.shape[0] == num_sample else a.T

    for key in sorted(base_feats.keys()):
        if key == "find_t": continue
        feature_map[key] = _to_2d(base_feats[key])
        if key != "token_diversity" and key in fh1:
            feature_map[f"diff_{key}"] = _to_2d(fh1[key]) - _to_2d(fh2[key])
    feature_map["slope"] = _to_2d(x_slope)
    feature_map["entropy"] = _to_2d(x_entropy)

    # RECONSTRUCT MATRIX
    ordered_keys = trained["final_keys"]
    x_attack = np.concatenate([feature_map[k] for k in ordered_keys], axis=1)
    
    # APPLY SAVED FLIP MASK 
    x_attack *= trained["flip_mask"]
    x_attack = np.nan_to_num(x_attack)

    # PREDICT
    scaler, model = trained["scaler"], trained["model"]
    probs = model.predict_proba(scaler.transform(x_attack))[:, 1]
    predictions = (probs > 0.5).astype(int)

    # PRINT SUMMARY
    print("\n" + "="*40)
    print("DETECTION RESULTS")
    print("="*40)
    
    # Analyze Member Preds specifically if requested
    mem_probs = probs[:num_members]
    mem_preds = predictions[:num_members]
    print(f"MEMBER_PREDS Average Score: {np.mean(mem_probs)*100:.2f}%")
    print(f"MEMBER_PREDS Correctly Identified: {np.sum(mem_preds)} / {num_members}")

    if num_sample > num_members:
        nm_probs = probs[num_members:]
        print(f"NON-MEMBER Average Score: {np.mean(nm_probs)*100:.2f}%")

    # CALCULATE METRICS IF BOTH CLASSES EXIST
    if num_members > 0 and num_members < num_sample:
        auc = roc_auc_score(y_true, probs)
        fpr_c, tpr_c, _ = roc_curve(y_true, probs)
        idx = np.where(fpr_c <= 0.01)[0]
        tpr_at_1fpr = tpr_c[idx[-1]] * 100 if len(idx) > 0 else 0.0
        
        print("\n" + "-"*40)
        print("GENRALIZATION PERFORMANCE ON BOOKS")
        print(f"ROC AUC: {auc:.4f}")
        print(f"TPR at 1% FPR: {tpr_at_1fpr:.2f}%")
        print("-"*40)

    # Save to CSV
    results_df = pd.DataFrame({
        "sample_id": range(1, num_sample + 1),
        "true_label": y_true,
        "is_detected_member": predictions,
        "membership_probability": probs
    })
    results_df.to_csv("mia_inference_results.csv", index=False)
    print(f"\nDetailed CSV saved to: mia_inference_results.csv")

if __name__ == "__main__":
    main()
