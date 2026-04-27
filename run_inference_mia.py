"""
run_inference_mia.py

Use a trained Context-Aware MIA model to predict membership on unknown data.
Now updated to handle ground-truth labels (if available) to calculate ROC AUC and TPR@5% FPR.
"""

import pickle
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        
    member_preds = data.get("member_preds", [])
    nonmember_preds = data.get("nonmember_preds", [])
    
    # Quick fix to combine them but remember the true label
    # 1 for members, 0 for nonmembers
    src = [(sample, 1) for sample in member_preds] + [(sample, 0) for sample in nonmember_preds]
    
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

    rep_key = next((k for k in ["tk_probs_repeated_5", "tk_probs_repeated_10"] if src and k in src[0][0]), None)
    
    x_all, label_all, x_repeated_all, y_true = [], [], [], []
    for sample, label in src:
        probs, labels, rep = _probs(sample, "tk_probs"), _labels(sample), _probs(sample, rep_key)
        if probs is None or rep is None or len(probs) < 5: continue
        x_all.append(probs); label_all.append(labels); x_repeated_all.append(rep)
        y_true.append(label)
    
    return x_all, label_all, x_repeated_all, np.array(y_true)

def extract_rep_half_split(x_repeated_all):
    half1, half2 = [], []
    for probs in x_repeated_all:
        mid = len(probs) // 2
        half1.append(probs[:mid]); half2.append(probs[mid:])
    return half1, half2

def plot_roc(fpr, tpr, auc, path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "darkorange", lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "navy", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC – Inference on Test Data")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.model_file, "rb") as f:
        trained = pickle.load(f)
    
    x_all, label_all, x_rep, y_true = load_custom_pickle(args.features_file)
    num_sample = len(x_all)

    print("Extracting signals for unknown data...")
    base_feats = collect_all_features(x_all, label_all)
    h1, h2 = extract_rep_half_split(x_rep)
    fh1, fh2 = collect_all_features(h1, label_all), collect_all_features(h2, label_all)

    # Constant cut-offs to ensure feature shape match with trained model
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

    # RECONSTRUCT MATRIX IN SAME ORDER AS TRAINING
    ordered_keys = trained["final_keys"]
    x_attack = np.concatenate([feature_map[k] for k in ordered_keys], axis=1)
    
    # APPLY SAVED FLIP MASK
    x_attack *= trained["flip_mask"]
    x_attack = np.nan_to_num(x_attack)

    # PREDICT
    scaler, model = trained["scaler"], trained["model"]
    probs = model.predict_proba(scaler.transform(x_attack))[:, 1]
    predictions = (probs > 0.5).astype(int)

    print("\n" + "="*40)
    print("INFERENCE RESULTS")
    print("="*40)
    
    # Count how many members and nonmembers we have
    has_mixed_labels = (len(np.unique(y_true)) > 1)
    
    if has_mixed_labels:
        auc = roc_auc_score(y_true, probs)
        fpr, tpr, _ = roc_curve(y_true, probs)
        
        # Calculate TPR @ 5% FPR (FPR <= 0.05)
        idx_5 = np.where(fpr <= 0.05)[0]
        tpr_at_5_fpr = tpr[idx_5[-1]] * 100 if len(idx_5) > 0 else 0.0
        
        # Calculate TPR @ 1% FPR for comparison
        idx_1 = np.where(fpr <= 0.01)[0]
        tpr_at_1_fpr = tpr[idx_1[-1]] * 100 if len(idx_1) > 0 else 0.0
        
        print(f"Test AUC             : {auc:.4f}")
        print(f"Test TPR @ 5% FPR    : {tpr_at_5_fpr:.2f}%")
        print(f"Test TPR @ 1% FPR    : {tpr_at_1_fpr:.2f}%")
        
        plot_roc(fpr, tpr, auc, "roc_curve_inference.png")
    else:
        print("Note: The test data only contains one class (either all members or all non-members).")
        print("Cannot compute AUC or ROC Curve.")

    member_count = np.sum(y_true == 1)
    nonmember_count = np.sum(y_true == 0)
    print(f"\nBreakdown:")
    print(f"- Ground-truth Members: {member_count}")
    print(f"- Ground-truth Non-Members: {nonmember_count}")
    print(f"- Model Predicted Members: {np.sum(predictions)}")
    
    pd.DataFrame({
        "true_label": y_true,
        "probability": probs, 
        "prediction": predictions
    }).to_csv("mia_inference_results.csv", index=False)
    
    print("\nDetailed results saved to mia_inference_results.csv")

if __name__ == "__main__":
    main()
