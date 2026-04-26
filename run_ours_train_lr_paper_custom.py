import pickle
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from util_features import collect_all_features, load_data_from_model_history, get_slope, approximate_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore') # ignore polyfit rank warnings

def get_model():
    clf = LogisticRegression(max_iter=5000, solver="liblinear")
    return clf

def train_model_single(x_train, x_test, y_train, y_test, n_components=10):
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    clf = get_model()
    clf.fit(x_train_scaled, y_train)
    pred_probs = clf.predict_proba(x_test_scaled)[:, 1]
    pred_probs_train = clf.predict_proba(x_train_scaled)[:, 1]
    
    auc_test = roc_auc_score(y_test, pred_probs)
    auc_train = roc_auc_score(y_train, pred_probs_train)
    
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    valid_indices = np.where(fpr <= 0.01)[0]
    tpr_test = tpr[valid_indices[-1]] * 100 if len(valid_indices) > 0 else 0.0
    
    fpr_train, tpr_train, _ = roc_curve(y_train, pred_probs_train)
    valid_indices_train = np.where(fpr_train <= 0.01)[0]
    tpr_train_val = tpr_train[valid_indices_train[-1]] * 100 if len(valid_indices_train) > 0 else 0.0
    
    results = {
        "auc_test": auc_test, "auc_train": auc_train,
        "tpr": tpr_test, "tpr_train": tpr_train_val,
        "fpr_curve": fpr, "tpr_curve": tpr,
    }
    
    if x_train.shape[1] >= n_components:
        pca = PCA(n_components=n_components)
        pca.fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)
        
        clf_pca = get_model()
        clf_pca.fit(x_train_pca, y_train)
        pred_probs_pca = clf_pca.predict_proba(x_test_pca)[:, 1]
        pred_probs_train_pca = clf_pca.predict_proba(x_train_pca)[:, 1]
        
        results["pca_auc_test"] = roc_auc_score(y_test, pred_probs_pca)
        results["pca_auc_train"] = roc_auc_score(y_train, pred_probs_train_pca)
        
        fpr_pca, tpr_pca, _ = roc_curve(y_test, pred_probs_pca)
        valid_indices_pca = np.where(fpr_pca <= 0.01)[0]
        results["pca_tpr"] = tpr_pca[valid_indices_pca[-1]] * 100 if len(valid_indices_pca) > 0 else 0.0
        
        fpr_train_pca, tpr_train_pca, _ = roc_curve(y_train, pred_probs_train_pca)
        valid_indices_train_pca = np.where(fpr_train_pca <= 0.01)[0]
        results["pca_tpr_train"] = tpr_train_pca[valid_indices_train_pca[-1]] * 100 if len(valid_indices_train_pca) > 0 else 0.0

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, default="all_features_custom.pkl")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--n_components", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading raw features from {args.features_file} and extracting mathematical signals...")

    # Load native signals from base pickle using paper's utility script
    print("Aligning repetition sequences (this might take a few moments)...")
    try:
        data_dict = load_data_from_model_history(args.features_file)
    except Exception as e:
        print(f"Error executing load_data_from_model_history. The lengths of repeated arrays probably couldn't be automatically aligned by util_features.py: {e}")
        return

    x_all = data_dict["x_all"]
    label_all = data_dict["label_all"]
    y_all = data_dict["y_all"]
    x_copied_all = data_dict["x_copied_all"]
    num_sample = len(y_all)

    print("Extracting base signals (Slope, Calibrated Loss, PPL, Lempel-Ziv, etc)...")
    extracted_features_single = collect_all_features(x_all, label_all)

    num_copies = min(len(x_copied_all[0]) if len(x_copied_all) > 0 else 0, 9)
    extracted_features_copied = []
    
    print(f"Extracting repetition copied signals for {num_copies} copies...")
    for i in range(num_copies):
        extracted_signal = collect_all_features(
            [x_copied_all[j][i] for j in range(len(x_copied_all))], label_all
        )
        extracted_features_copied.append(extracted_signal)

    # Approximate Entropy array extraction
    m = 8
    r = 0.8
    # Fallback context spans to prevent out_of_bounds if input is short
    arr_len = len(data_dict["all_preds_copies"][0]) if data_dict["all_preds_copies"] else 200
    cut_offs = [min(600, arr_len), min(800, arr_len), min(1000, arr_len)]
    cut_offs = sorted(list(set(cut_offs)))
    
    x_approximate_entropy = np.array([
        np.array([approximate_entropy(data_dict["all_preds_copies"][idx][:cut_off], m, r)
                  for idx in range(num_sample)])
        for cut_off in cut_offs
    ]).T

    # Slope mapping
    x_slope_signal = np.array([
        get_slope(data_dict["all_preds_copies"], end_time=c).T for c in cut_offs
    ]).T

    # Assemble everything into a final dictionary mimicking paper logic
    final_features = {}
    filter_features = ["find_t"]

    print("Bundling standard signals with f_Rep differentials...")
    for keys in extracted_features_single.keys():
        if keys in filter_features: continue
        final_features[keys] = extracted_features_single[keys]
        
        # Calculate repetitions diffs (f_Rep equivalent)
        if keys not in ["token_diversity"] and num_copies >= 3:
            final_features[f"overall_diff_1_{keys}"] = extracted_features_copied[0][keys] - extracted_features_copied[1][keys]
            final_features[f"overall_diff_2_{keys}"] = extracted_features_copied[0][keys] - extracted_features_copied[2][keys]

    final_features["slope"] = x_slope_signal
    final_features["approximate_entropy"] = x_approximate_entropy

    # Count above pre
    signals = []
    for cut_off in [min(1000, len(x_all[0])), 200, 300]:
        x_pre_mean = [[np.mean(x_all[sample_idx][:t]) for t in range(1, min(cut_off, len(x_all[sample_idx])))] for sample_idx in range(len(x_all))]
        x_cur = [[x_all[sample_idx][t] for t in range(1, min(cut_off, len(x_all[sample_idx])))] for sample_idx in range(len(x_all))]
        signals.append(np.array([np.mean(np.array(x) > np.array(x_pre)) if len(x) > 0 else 0.0 for x, x_pre in zip(x_cur, x_pre_mean)]))
    final_features["count_above_pre"] = np.array(signals).T

    # Uniform Direction logic
    for keys in final_features.keys():
        if final_features[keys].T[y_all == 1].mean() > final_features[keys].T[y_all == 0].mean():
            final_features[keys] = -final_features[keys]

    print("Building flat ML matrix out of final_features dictionary...")
    x_matrix_cols = []
    for keys in sorted(final_features.keys()):
        val = final_features[keys]
        if len(val.shape) == 1:
            val = val.reshape(-1, 1)
        if val.shape[1] == num_sample and val.shape[0] != num_sample:
            val = val.T
        x_matrix_cols.append(val)
    
    x_attack = np.concatenate(x_matrix_cols, axis=1)
    print(f"Final Signal Feature Matrix shape: {x_attack.shape} [Samples x Aggregated Metrics]")
    
    # Simple Train-Test split
    shuffle_idx = np.random.permutation(len(x_attack))
    x_attack = x_attack[shuffle_idx]
    y_all_shuffled = y_all[shuffle_idx]
    
    split_point = int(len(x_attack) * (1 - args.test_split))
    x_train, x_test = x_attack[:split_point], x_attack[split_point:]
    y_train, y_test = y_all_shuffled[:split_point], y_all_shuffled[split_point:]
    
    print("\nTraining logistic regression models on rigorous paper signals...")
    results = train_model_single(x_train, x_test, y_train, y_test, min(args.n_components, x_train.shape[1]))
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (Using Context-Aware MIA Math)")
    print("="*50)
    print(f"Test AUC: {results['auc_test']:.4f}")
    print(f"Train AUC: {results['auc_train']:.4f}")
    print(f"Test TPR@FPR<=0.01: {results['tpr']:.2f}%")
    print(f"Train TPR@FPR<=0.01: {results['tpr_train']:.2f}%")
    
    if "pca_auc_test" in results:
        print(f"\nWith PCA ({min(args.n_components, x_train.shape[1])} components):")
        print(f"  Test AUC: {results['pca_auc_test']:.4f}")
        print(f"  Train AUC: {results['pca_auc_train']:.4f}")
        print(f"  Test TPR@FPR<=0.01: {results['pca_tpr']:.2f}%")
        print(f"  Train TPR@FPR<=0.01: {results['pca_tpr_train']:.2f}%")
    print("="*50)
    
    model_file = os.path.join(args.output_dir, "trained_model_paper_custom.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(results, f)
    print(f"\nModel strictly using paper features saved to {model_file}")

if __name__ == "__main__":
    main()
