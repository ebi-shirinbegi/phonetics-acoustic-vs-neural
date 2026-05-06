"""Section 6.2: inter-phoneme distances, bootstrap CIs, NNC classifier with LOSO, McNemar."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform, mahalanobis
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]
PAIRS_OF_INTEREST = [("e", "ɛ"), ("o", "u"), ("y", "u"), ("i", "y")]


def acoustic_centroids(df, vowels):
    rows, kept = [], []
    for v in vowels:
        s = df[(df["phoneme"] == v) &
               df["F1_lobanov"].notna() & df["F2_lobanov"].notna()]
        if len(s) < 5:
            continue
        rows.append([s["F1_lobanov"].mean(), s["F2_lobanov"].mean()])
        kept.append(v)
    return np.array(rows), kept


def neural_centroids(emb, labels, vowels):
    rows, kept = [], []
    for v in vowels:
        m = labels == v
        if m.sum() < 5:
            continue
        rows.append(emb[m].mean(axis=0))
        kept.append(v)
    return np.vstack(rows), kept


def euclidean_dm(X):
    return squareform(pdist(X, metric="euclidean"))


def cosine_dm(X):
    return squareform(pdist(X, metric="cosine"))


def mahalanobis_dm(X, cov_inv):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = mahalanobis(X[i], X[j], cov_inv)
            D[i, j] = D[j, i] = d
    return D


def pooled_inv_cov(df, vowels):
    rows = []
    for v in vowels:
        s = df[(df["phoneme"] == v) &
               df["F1_lobanov"].notna() & df["F2_lobanov"].notna()]
        if len(s) < 5:
            continue
        X = s[["F1_lobanov", "F2_lobanov"]].to_numpy()
        rows.append(X - X.mean(axis=0))
    pooled = np.vstack(rows)
    cov = np.cov(pooled.T)
    return np.linalg.inv(cov)


def bootstrap_pair_distance(data, index, v1, v2, kind, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "acoustic":
        speakers = data["speaker"].unique()
    else:
        speakers = index["speaker"].unique()
    boot_d = []
    for _ in range(n_boot):
        sample = rng.choice(speakers, size=len(speakers), replace=True)
        if kind == "acoustic":
            sub = data[data["speaker"].isin(sample)]
            s1 = sub[sub["phoneme"] == v1][["F1_lobanov", "F2_lobanov"]].dropna()
            s2 = sub[sub["phoneme"] == v2][["F1_lobanov", "F2_lobanov"]].dropna()
            if len(s1) < 3 or len(s2) < 3:
                continue
            d = float(np.linalg.norm(s1.mean(axis=0) - s2.mean(axis=0)))
        else:
            spk = index["speaker"].to_numpy()
            phn = index["phoneme"].to_numpy()
            spk_mask = np.isin(spk, sample)
            e1 = data[spk_mask & (phn == v1)]
            e2 = data[spk_mask & (phn == v2)]
            if len(e1) < 3 or len(e2) < 3:
                continue
            d = float(cosine_distances(e1.mean(axis=0, keepdims=True),
                                        e2.mean(axis=0, keepdims=True))[0, 0])
        boot_d.append(d)
    arr = np.array(boot_d)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def nc_classify(X_train, y_train, X_test, metric):
    classes = np.unique(y_train)
    centroids = np.vstack([X_train[y_train == c].mean(axis=0) for c in classes])
    if metric == "euclidean":
        D = np.linalg.norm(X_test[:, None, :] - centroids[None, :, :], axis=2)
    else:
        D = cosine_distances(X_test, centroids)
    return classes[np.argmin(D, axis=1)]


def loso_classifier(X, y, speakers, metric, vowels):
    pred = np.full(len(y), "", dtype=object)
    classified = np.zeros(len(y), dtype=bool)
    for spk in np.unique(speakers):
        test_mask = speakers == spk
        train_mask = ~test_mask
        keep_tr = np.isin(y[train_mask], vowels)
        X_tr = X[train_mask][keep_tr]
        y_tr = y[train_mask][keep_tr]
        if len(X_tr) == 0:
            continue
        keep_te = np.isin(y[test_mask], vowels)
        if keep_te.sum() == 0:
            continue
        X_te = X[test_mask][keep_te]
        p = nc_classify(X_tr, y_tr, X_te, metric)
        idx_test = np.where(test_mask)[0]
        idx_keep = idx_test[keep_te]
        pred[idx_keep] = p
        classified[idx_keep] = True
    return pred, classified


def per_class_f1(y_true, y_pred, classes):
    rows = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"class": c, "precision": prec, "recall": rec, "f1": f1,
                     "support": int((y_true == c).sum())})
    return pd.DataFrame(rows)


def confusion_matrix_df(y_true, y_pred, classes):
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            M[i, j] = ((y_true == c_true) & (y_pred == c_pred)).sum()
    return pd.DataFrame(M, index=classes, columns=classes)


def plot_confusion(cm, title, out):
    fig, ax = plt.subplots(figsize=(7, 6))
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    sns.heatmap(norm, annot=cm, fmt="d", cmap="rocket_r", ax=ax,
                cbar_kws={"label": "row-normalised proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def mcnemar_pair(y_true, p_a, p_b, name_a, name_b, mask):
    a_correct = (p_a == y_true) & mask
    b_correct = (p_b == y_true) & mask
    n01 = int(((~a_correct) & b_correct).sum())
    n10 = int((a_correct & (~b_correct)).sum())
    table = [[0, n01], [n10, 0]]
    res = mcnemar(table, exact=False, correction=True)
    return {"a": name_a, "b": name_b, "n_a_only": n10,
            "n_b_only": n01, "stat": res.statistic, "p": res.pvalue}


def main():
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    index = pd.read_csv(DATA / "features_neural_index.csv")
    emb_wh = np.load(DATA / "features_whisper_L20.npz")["embeddings"]
    emb_xl = np.load(DATA / "features_xlsr_L12.npz")["embeddings"]

    print("=== Distance matrices on vowel centroids ===")
    X_ac, vowels = acoustic_centroids(df_ac, ORAL_VOWELS)
    X_wh, _ = neural_centroids(emb_wh, index["phoneme"].to_numpy(), vowels)
    X_xl, _ = neural_centroids(emb_xl, index["phoneme"].to_numpy(), vowels)

    D_eu = euclidean_dm(X_ac)
    cov_inv = pooled_inv_cov(df_ac, vowels)
    D_mh = mahalanobis_dm(X_ac, cov_inv)
    D_wh = cosine_dm(X_wh)
    D_xl = cosine_dm(X_xl)

    np.savez(DATA / "section6_2_dms.npz",
             eu=D_eu, mh=D_mh, wh=D_wh, xl=D_xl, vowels=np.array(vowels))
    print("Saved section6_2_dms.npz")

    front_back = {"i": 0, "e": 0, "ɛ": 0, "a": 1, "ɑ": 2, "o": 2,
                  "u": 2, "y": 0, "ø": 0, "ə": 1}
    high_low = {"i": 0, "e": 1, "ɛ": 2, "a": 3, "ɑ": 3, "o": 1,
                "u": 0, "y": 0, "ø": 1, "ə": 1}
    ideal = np.zeros_like(D_eu)
    for i, v1 in enumerate(vowels):
        for j, v2 in enumerate(vowels):
            ideal[i, j] = abs(front_back[v1] - front_back[v2]) + abs(high_low[v1] - high_low[v2])
    iu = np.triu_indices(len(vowels), k=1)
    print("\n=== Spearman r vs IPA-trapezoid ideal distance ===")
    for name, D in [("Euclidean (acoustic)", D_eu),
                    ("Mahalanobis (acoustic)", D_mh),
                    ("Cosine (Whisper L20)", D_wh),
                    ("Cosine (XLS-R L12)", D_xl)]:
        r, p = spearmanr(D[iu], ideal[iu])
        print(f"  {name:28s}  r={r:+.3f}  p={p:.4f}")

    print("\n=== Bootstrap 95% CIs on selected vowel pairs ===")
    rows = []
    for v1, v2 in PAIRS_OF_INTEREST:
        m, lo, hi = bootstrap_pair_distance(df_ac, None, v1, v2, "acoustic")
        rows.append({"v1": v1, "v2": v2, "rep": "acoustic",
                     "mean": m, "ci_lo": lo, "ci_hi": hi})
        m, lo, hi = bootstrap_pair_distance(emb_wh, index, v1, v2, "neural")
        rows.append({"v1": v1, "v2": v2, "rep": "whisper_L20",
                     "mean": m, "ci_lo": lo, "ci_hi": hi})
        m, lo, hi = bootstrap_pair_distance(emb_xl, index, v1, v2, "neural")
        rows.append({"v1": v1, "v2": v2, "rep": "xlsr_L12",
                     "mean": m, "ci_lo": lo, "ci_hi": hi})
    boot_df = pd.DataFrame(rows)
    boot_df.to_csv(DATA / "section6_2_bootstrap.csv", index=False)
    print(boot_df.round(4).to_string(index=False))

    print("\n=== Nearest-centroid classifier (leave-one-speaker-out) ===")
    speakers = index["speaker"].to_numpy()
    phon = index["phoneme"].to_numpy()
    L1 = index["L1"].to_numpy()

    df_ac_idx = df_ac.merge(
        index.reset_index().rename(columns={"index": "row_idx"}),
        on=["speaker", "L1", "gender", "sentence_id", "repetition",
            "phoneme", "onset", "offset"],
        how="right"
    ).sort_values("row_idx")
    X_ac_full = df_ac_idx[["F1_lobanov", "F2_lobanov"]].fillna(0).to_numpy()

    classifiers = {
        "acoustic":    (X_ac_full, "euclidean"),
        "whisper_L20": (emb_wh, "cosine"),
        "xlsr_L12":    (emb_xl, "cosine"),
    }

    preds = {}
    masks = {}
    for name, (X, metric) in classifiers.items():
        p, m = loso_classifier(X, phon, speakers, metric, vowels)
        preds[name] = p
        masks[name] = m
        common = m & np.isin(phon, vowels)
        acc = (p[common] == phon[common]).mean()
        print(f"  {name:14s}  accuracy={acc:.3f}  (n={common.sum()})")
        f1 = per_class_f1(phon[common], p[common], vowels)
        f1.to_csv(DATA / f"section6_2_f1_{name}.csv", index=False)
        cm = confusion_matrix_df(phon[common], p[common], vowels)
        cm.to_csv(DATA / f"section6_2_cm_{name}.csv")
        plot_confusion(cm, f"Confusion matrix — {name}",
                       FIG / f"fig_6_2_cm_{name}.png")

    print("\n=== L1 vs L2 split accuracy ===")
    rows = []
    for name in classifiers:
        common = masks[name] & np.isin(phon, vowels)
        for grp in ["fr", "ru"]:
            mask = common & (L1 == grp)
            acc = (preds[name][mask] == phon[mask]).mean()
            rows.append({"rep": name, "L1": grp, "n": int(mask.sum()),
                         "accuracy": acc})
    split_df = pd.DataFrame(rows)
    split_df.to_csv(DATA / "section6_2_l1l2_accuracy.csv", index=False)
    print(split_df.round(4).to_string(index=False))

    print("\n=== McNemar tests across representations ===")
    common_all = masks["acoustic"] & masks["whisper_L20"] & masks["xlsr_L12"] & np.isin(phon, vowels)
    rows = []
    pairs = [("acoustic", "whisper_L20"), ("acoustic", "xlsr_L12"),
             ("whisper_L20", "xlsr_L12")]
    for a, b in pairs:
        rows.append(mcnemar_pair(phon, preds[a], preds[b], a, b, common_all))
    mcnemar_df = pd.DataFrame(rows)
    mcnemar_df.to_csv(DATA / "section6_2_mcnemar.csv", index=False)
    print(mcnemar_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()