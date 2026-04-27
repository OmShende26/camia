"""
run_ours_train_lr_paper_custom.py

Drop-in replacement for run_ours_train_lr_custom.py that uses the
rigorous paper signals from util_features.py instead of naive aggregates.

The custom pickle format (from run_ours_construct_mia_data_custom.py) has:
  {
    "member_preds":    [defaultdict, ...],   # each dict has tk_probs, labels, tk_probs_repeated_5, labels_repeated_5
    "nonmember_preds": [defaultdict, ...],
  }

Since load_data_from_model_history requires torch tensors stored as numpy objects
(Pythia-dataset specific), we write our own compatible loader here.
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
    get_token_diversity,
    get_loss,
    get_count_above,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Custom data loader for our custom pickle format
# ---------------------------------------------------------------------------
def load_custom_pickle(path):
    """
    Load the custom pickle produced by run_ours_construct_mia_data_custom.py.

    Returns a dict with:
      x_all          – list[list[float]]  per-token log-probs (single pass)
      label_all      – list[list[int]]    token ids
      y_all          – np.ndarray of 0/1 labels
      all_preds_copies – list[list[float]] per-token log-probs of the REPEATED text
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    member_preds    = data["member_preds"]
    nonmember_preds = data["nonmember_preds"]

    # Helper: pull the first entry that has actual data
    def _probs(sample_dict, key):
        v = sample_dict.get(key, [])
        if len(v) == 0:
            return None
        entry = v[0]
        if hasattr(entry, "tolist"):          # torch tensor or numpy array
            return entry.tolist()
        return list(entry)                    # already a plain list

    def _labels(sample_dict):
        v = sample_dict.get("labels", [])
        if len(v) == 0:
            return []
        entry = v[0]
        if hasattr(entry, "squeeze"):         # torch tensor
            entry = entry.squeeze()
            if hasattr(entry, "tolist"):
                return entry.tolist()
        return list(entry)

    # Detect the repetition key (5 or 10 repetitions)
    rep_key = None
    for candidate in ["tk_probs_repeated_5", "tk_probs_repeated_10"]:
        src = member_preds if len(member_preds) > 0 else nonmember_preds
        if len(src) > 0 and candidate in src[0]:
            rep_key = candidate
            break

    if rep_key is None:
        raise ValueError("Could not find tk_probs_repeated_N key in the pickle. "
                         "Make sure the construction script ran successfully.")

    print(f"  Using repetition key: {rep_key}")

    x_all, label_all, x_repeated_all = [], [], []
    valid_nonmember, valid_member = 0, 0

    for preds, label_val in [(nonmember_preds, 0), (member_preds, 1)]:
        for sample in preds:
            probs  = _probs(sample, "tk_probs")
            labels = _labels(sample)
            rep    = _probs(sample, rep_key)

            # Skip malformed entries
            if probs is None or rep is None or len(probs) < 5:
                continue

            x_all.append(probs)
            label_all.append(labels)
            x_repeated_all.append(rep)
            if label_val == 0:
                valid_nonmember += 1
            else:
                valid_member += 1

    non_count = valid_nonmember
    mem_count = valid_member
    y_all = np.array([0] * non_count + [1] * mem_count, dtype=float)

    print(f"  Loaded {non_count} non-members   {mem_count} members   ({len(x_all)} total)")
    return {
        "x_all":            x_all,
        "label_all":        label_all,
        "y_all":            y_all,
        "all_preds_copies": x_repeated_all,
    }


# ---------------------------------------------------------------------------
# Repetition-based signal: the paper computes differences between copies.
# Since our custom script stores ONE repeated string (text * N concat), we
# approximate f_Rep by splitting the repeated array into two halves and
# computing per-token features on each half, then differencing.
# ---------------------------------------------------------------------------
def extract_rep_signal_from_repeated(x_repeated_all):
    """
    Split the full repeated-text token array into two equal halves and
    return (first_half_list, second_half_list) of per-token log-prob lists.
    This mimics how the paper grabs two distinct copies of the text from
    the concatenated repeated string.
    """
    half1, half2 = [], []
    for probs in x_repeated_all:
        n = len(probs)
        mid = n // 2
        half1.append(probs[:mid])
        half2.append(probs[mid:])
    return half1, half2


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def train_model_single(x_train, x_test, y_train, y_test, n_components=10):
    scaler = MinMaxScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s  = scaler.transform(x_test)

    clf = LogisticRegression(max_iter=5000, solver="liblinear")
    clf.fit(x_train_s, y_train)
    p_test  = clf.predict_proba(x_test_s)[:, 1]
    p_train = clf.predict_proba(x_train_s)[:, 1]

    auc_test  = roc_auc_score(y_test,  p_test)
    auc_train = roc_auc_score(y_train, p_train)

    fpr, tpr, _ = roc_curve(y_test, p_test)
    idx = np.where(fpr <= 0.01)[0]
    tpr_test = tpr[idx[-1]] * 100 if len(idx) > 0 else 0.0

    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_train)
    idx_tr = np.where(fpr_tr <= 0.01)[0]
    tpr_train = tpr_tr[idx_tr[-1]] * 100 if len(idx_tr) > 0 else 0.0

    results = dict(auc_test=auc_test, auc_train=auc_train,
                   tpr=tpr_test, tpr_train=tpr_train,
                   fpr_curve=fpr, tpr_curve=tpr)

    n_feat = x_train.shape[1]
    pca_k  = min(n_components, n_feat)
    if pca_k >= 2:
        pca = PCA(n_components=pca_k)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca  = pca.transform(x_test)

        clf_pca = LogisticRegression(max_iter=5000, solver="liblinear")
        clf_pca.fit(x_train_pca, y_train)
        pp_test  = clf_pca.predict_proba(x_test_pca)[:, 1]
        pp_train = clf_pca.predict_proba(x_train_pca)[:, 1]

        results["pca_auc_test"]  = roc_auc_score(y_test,  pp_test)
        results["pca_auc_train"] = roc_auc_score(y_train, pp_train)

        fpr_p, tpr_p, _ = roc_curve(y_test, pp_test)
        idx_p = np.where(fpr_p <= 0.01)[0]
        results["pca_tpr"] = tpr_p[idx_p[-1]] * 100 if len(idx_p) > 0 else 0.0

        fpr_ptr, tpr_ptr, _ = roc_curve(y_train, pp_train)
        idx_ptr = np.where(fpr_ptr <= 0.01)[0]
        results["pca_tpr_train"] = tpr_ptr[idx_ptr[-1]] * 100 if len(idx_ptr) > 0 else 0.0

    return results


def plot_roc(fpr, tpr, auc, path):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "darkorange", lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "navy", lw=1.5, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC – Context-Aware MIA (Paper Signals)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, default="all_features_custom.pkl")
    parser.add_argument("--output_dir",    type=str, default="./")
    parser.add_argument("--test_split",    type=float, default=0.2)
    parser.add_argument("--n_components",  type=int, default=10)
    parser.add_argument("--random_seed",   type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {args.features_file} ...")
    if not os.path.exists(args.features_file):
        raise FileNotFoundError(f"File not found: {args.features_file}")
    data_dict = load_custom_pickle(args.features_file)

    x_all            = data_dict["x_all"]          # per-token log-probs (original text)
    label_all        = data_dict["label_all"]       # token ids
    y_all            = data_dict["y_all"]           # 0/1 labels
    all_preds_copies = data_dict["all_preds_copies"] # per-token log-probs (repeated text)
    num_sample       = len(y_all)

    # ── 2. Paper Signals ─────────────────────────────────────────────────────
    print("\n[2/4] Computing paper mathematical signals from raw token log-probs ...")

    # ─ Signal 1: Token Diversity Calibration  f_Cal = loss / d_X ─
    # ─ Signal 2: Truncated Loss               f_Cut (end_time limits) ─
    # ─ Signal 3: Low-Loss Counting             count_above thresholds ─
    # ─ Signal 4: Calibrated PPL ─
    # All via collect_all_features:
    print("  ▷ collect_all_features (base signals) ...")
    base_feats = collect_all_features(x_all, label_all)

    # ─ Signal 5 (f_Rep): Repetition differential ─
    print("  ▷ Computing f_Rep via repeated-text half-split ...")
    half1, half2 = extract_rep_signal_from_repeated(all_preds_copies)
    feats_half1  = collect_all_features(half1, label_all)
    feats_half2  = collect_all_features(half2, label_all)

    # ─ Slope signal (f_Slope) ─
    print("  ▷ Computing Slope signals ...")
    safe_len = min(len(p) for p in all_preds_copies) if all_preds_copies else 200
    cut_offs = sorted({min(c, safe_len) for c in [200, 400, safe_len]})
    if len(cut_offs) == 1:
        cut_offs = cut_offs * 2          # polyfit needs at least 2 points anyway

    try:
        slope_cols = [get_slope(all_preds_copies, end_time=c) for c in cut_offs]
        x_slope    = np.column_stack(slope_cols)   # (N, len(cut_offs))
    except Exception as e:
        print(f"  ⚠  Slope signal skipped: {e}")
        x_slope = None

    # ─ Approximate Entropy ─
    print("  ▷ Computing Approximate Entropy ...")
    m, r = 8, 0.8
    try:
        ent_cols = []
        for c in cut_offs:
            col = np.array([approximate_entropy(p[:c], m, r)
                            for p in all_preds_copies])
            ent_cols.append(col)
        x_entropy = np.column_stack(ent_cols)      # (N, len(cut_offs))
    except Exception as e:
        print(f"  ⚠  Entropy signal skipped: {e}")
        x_entropy = None

    # ── 3. Assemble Feature Matrix ───────────────────────────────────────────
    print("\n[3/4] Assembling final feature matrix ...")

    filter_keys = {"find_t"}        # not stable on short sequences
    feature_blocks = []

    def _to_2d(arr):
        """Ensure shape is (num_sample, k)."""
        a = np.array(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.shape[0] != num_sample and a.shape[1] == num_sample:
            a = a.T
        assert a.shape[0] == num_sample, f"Shape mismatch: {a.shape}"
        return a

    for key, feats in base_feats.items():
        if key in filter_keys:
            continue
        block = _to_2d(feats)
        feature_blocks.append(block)

        # f_Rep differentials (only for keys present in half-split features)
        if key not in {"token_diversity"} and key in feats_half1 and key in feats_half2:
            diff1 = _to_2d(feats_half1[key]) - _to_2d(feats_half2[key])
            feature_blocks.append(diff1)

    if x_slope   is not None:  feature_blocks.append(_to_2d(x_slope))
    if x_entropy is not None:  feature_blocks.append(_to_2d(x_entropy))

    x_attack = np.concatenate(feature_blocks, axis=1)
    print(f"  Feature matrix shape: {x_attack.shape}  (samples × signals)")

    # Flip sign so that lower value ⟹ member (paper convention)
    for j in range(x_attack.shape[1]):
        col = x_attack[:, j]
        if np.nanmean(col[y_all == 1]) > np.nanmean(col[y_all == 0]):
            x_attack[:, j] = -col

    # Replace any NaN/Inf with zero
    x_attack = np.nan_to_num(x_attack, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 4. Train & Evaluate ──────────────────────────────────────────────────
    print("\n[4/4] Training & evaluating Logistic Regression ...")

    shuffle = np.random.permutation(num_sample)
    x_shuf, y_shuf = x_attack[shuffle], y_all[shuffle]

    split  = int(num_sample * (1 - args.test_split))
    x_tr, x_te = x_shuf[:split], x_shuf[split:]
    y_tr, y_te = y_shuf[:split], y_shuf[split:]

    print(f"  Train: {len(x_tr)}  |  Test: {len(x_te)}")
    results = train_model_single(x_tr, x_te, y_tr, y_te,
                                 n_components=args.n_components)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESULTS  ─  Context-Aware MIA (Paper Signals)")
    print("=" * 55)
    print(f"  Test  AUC            : {results['auc_test']:.4f}")
    print(f"  Train AUC            : {results['auc_train']:.4f}")
    print(f"  Test  TPR@FPR≤0.01   : {results['tpr']:.2f}%")
    print(f"  Train TPR@FPR≤0.01   : {results['tpr_train']:.2f}%")
    if "pca_auc_test" in results:
        k = min(args.n_components, x_tr.shape[1])
        print(f"\n  With PCA ({k} components):")
        print(f"    Test  AUC          : {results['pca_auc_test']:.4f}")
        print(f"    Train AUC          : {results['pca_auc_train']:.4f}")
        print(f"    Test  TPR@FPR≤0.01 : {results['pca_tpr']:.2f}%")
        print(f"    Train TPR@FPR≤0.01 : {results['pca_tpr_train']:.2f}%")
    print("=" * 55)

    # ── Save outputs ─────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "trained_model_paper_custom.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nModel saved → {model_path}")

    roc_path = os.path.join(args.output_dir, "roc_curve_paper_custom.png")
    plot_roc(results["fpr_curve"], results["tpr_curve"], results["auc_test"], roc_path)


if __name__ == "__main__":
    main()