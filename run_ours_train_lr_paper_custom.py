"""
run_ours_train_lr_paper_custom.py

Drop-in replacement for run_ours_train_lr_custom.py that uses the
rigorous paper signals from util_features.py instead of naive aggregates.

Saves the Model, Scaler, and Flip-Mask for use in real-world inference.
"""

import pickle
import argparse
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from util_features import (
    collect_all_features,
    get_slope,
    approximate_entropy,
)

warnings.filterwarnings("ignore")

def load_custom_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    member_preds    = data["member_preds"]
    nonmember_preds = data["nonmember_preds"]

    def _probs(sample_dict, key):
        v = sample_dict.get(key, [])
        if len(v) == 0: return None
        entry = v[0]
        return entry.tolist() if hasattr(entry, "tolist") else list(entry)

    def _labels(sample_dict):
        v = sample_dict.get("labels", [])
        if len(v) == 0: return []
        entry = v[0]
        if hasattr(entry, "squeeze"): entry = entry.squeeze()
        return entry.tolist() if hasattr(entry, "tolist") else list(entry)

    rep_key = None
    for candidate in ["tk_probs_repeated_5", "tk_probs_repeated_10"]:
        src = member_preds if len(member_preds) > 0 else nonmember_preds
        if len(src) > 0 and candidate in src[0]:
            rep_key = candidate
            break
    if rep_key is None: raise ValueError("Could not find repetition key.")

    x_all, label_all, x_repeated_all = [], [], []
    valid_nonmember, valid_member = 0, 0
    for preds, label_val in [(nonmember_preds, 0), (member_preds, 1)]:
        for sample in preds:
            probs, labels, rep = _probs(sample, "tk_probs"), _labels(sample), _probs(sample, rep_key)
            if probs is None or rep is None or len(probs) < 5: continue
            x_all.append(probs); label_all.append(labels); x_repeated_all.append(rep)
            if label_val == 0: valid_nonmember += 1
            else: valid_member += 1

    y_all = np.array([0] * valid_nonmember + [1] * valid_member, dtype=float)
    return {"x_all": x_all, "label_all": label_all, "y_all": y_all, "all_preds_copies": x_repeated_all}

def extract_rep_half_split(x_repeated_all):
    half1, half2 = [], []
    for probs in x_repeated_all:
        mid = len(probs) // 2
        half1.append(probs[:mid]); half2.append(probs[mid:])
    return half1, half2

def train_model_single(x_train, x_test, y_train, y_test, n_components=10):
    scaler = MinMaxScaler()
    x_train_s, x_test_s = scaler.fit_transform(x_train), scaler.transform(x_test)
    clf = LogisticRegression(max_iter=5000, solver="liblinear")
    clf.fit(x_train_s, y_train)
    p_test, p_train = clf.predict_proba(x_test_s)[:, 1], clf.predict_proba(x_train_s)[:, 1]
    
    results = {"auc_test": roc_auc_score(y_test, p_test), "auc_train": roc_auc_score(y_train, p_train), "model": clf, "scaler": scaler}
    
    fpr, tpr, _ = roc_curve(y_test, p_test)
    idx = np.where(fpr <= 0.01)[0]
    results["tpr"] = tpr[idx[-1]] * 100 if len(idx) > 0 else 0.0
    results["fpr_curve"], results["tpr_curve"] = fpr, tpr

    if x_train.shape[1] >= 2:
        pca = PCA(n_components=min(n_components, x_train.shape[1]))
        x_tr_p, x_te_p = pca.fit_transform(x_train), pca.transform(x_test)
        clf_p = LogisticRegression(max_iter=5000, solver="liblinear").fit(x_tr_p, y_train)
        results["pca_auc_test"], results["pca_model"], results["pca"] = roc_auc_score(y_test, clf_p.predict_proba(x_te_p)[:, 1]), clf_p, pca
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, default="all_features_custom.pkl")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()

    data_dict = load_custom_pickle(args.features_file)
    x_all, label_all, y_all, all_preds_copies = data_dict["x_all"], data_dict["label_all"], data_dict["y_all"], data_dict["all_preds_copies"]
    num_sample = len(y_all)

    print("Extracting signals...")
    base_feats = collect_all_features(x_all, label_all)
    h1, h2 = extract_rep_half_split(all_preds_copies)
    fh1, fh2 = collect_all_features(h1, label_all), collect_all_features(h2, label_all)

    safe_len = min(len(p) for p in all_preds_copies) if all_preds_copies else 200
    cut_offs = sorted({min(c, safe_len) for c in [200, 400, safe_len]})
    if len(cut_offs) == 1: cut_offs = cut_offs * 2
    x_slope = np.column_stack([get_slope(all_preds_copies, end_time=c) for c in cut_offs])
    x_entropy = np.column_stack([np.array([approximate_entropy(p[:c], 8, 0.8) for p in all_preds_copies]) for c in cut_offs])

    feature_map = {}
    ordered_keys = sorted(base_feats.keys())
    for key in ordered_keys:
        if key == "find_t": continue
        def _to_2d(val): 
            a = np.array(val).reshape(num_sample, -1) if np.array(val).ndim == 1 else np.array(val)
            return a if a.shape[0] == num_sample else a.T
        
        feature_map[key] = _to_2d(base_feats[key])
        if key != "token_diversity" and key in fh1:
            feature_map[f"diff_{key}"] = _to_2d(fh1[key]) - _to_2d(fh2[key])

    feature_map["slope"] = _to_2d(x_slope)
    feature_map["entropy"] = _to_2d(x_entropy)
    
    # Final matrix assembly
    final_keys = sorted(feature_map.keys())
    x_attack = np.concatenate([feature_map[k] for k in final_keys], axis=1)
    
    # Sign Alignment (Flip Mask)
    flip_mask = np.ones(x_attack.shape[1])
    for j in range(x_attack.shape[1]):
        if np.nanmean(x_attack[:, j][y_all == 1]) > np.nanmean(x_attack[:, j][y_all == 0]):
            flip_mask[j] = -1
            x_attack[:, j] *= -1

    x_attack = np.nan_to_num(x_attack)
    shuffle = np.random.permutation(num_sample)
    split = int(num_sample * 0.8)
    x_tr, x_te = x_attack[shuffle[:split]], x_attack[shuffle[split:]]
    y_tr, y_te = y_all[shuffle[:split]], y_all[shuffle[split:]]

    results = train_model_single(x_tr, x_te, y_tr, y_te)
    results["flip_mask"], results["final_keys"] = flip_mask, final_keys

    print(f"Test AUC: {results['auc_test']:.4f}")
    with open(os.path.join(args.output_dir, "trained_model_paper_custom.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
