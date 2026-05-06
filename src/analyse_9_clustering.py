"""Section 9: Hierarchical clustering of vowels, vowels+consonants, and speakers."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]
CONSONANTS = ["s", "ʃ", "p", "k", "l", "n"]

FRONT_BACK = {
    "i": "front", "e": "front", "ɛ": "front", "y": "front", "ø": "front",
    "a": "central", "ə": "central",
    "ɑ": "back", "o": "back", "u": "back",
}
HIGH_LOW = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid", "ɛ": "mid", "ø": "mid", "o": "mid", "ə": "mid",
    "a": "low", "ɑ": "low",
}


def acoustic_centroid_per_phoneme(df, phonemes):
    rows, kept = [], []
    for p in phonemes:
        sub = df[(df["phoneme"] == p) &
                 df["F1_lobanov"].notna() & df["F2_lobanov"].notna()]
        if len(sub) < 5:
            continue
        rows.append([sub["F1_lobanov"].mean(), sub["F2_lobanov"].mean()])
        kept.append(p)
    return np.array(rows), kept


def neural_centroid_per_phoneme(emb, labels, phonemes):
    rows, kept = [], []
    for p in phonemes:
        m = labels == p
        if m.sum() < 5:
            continue
        rows.append(emb[m].mean(axis=0))
        kept.append(p)
    return np.vstack(rows), kept


def cluster_and_score(X, labels, partition_name, partition_dict, metric, k):
    """Hierarchical clustering with k clusters, return ARI vs given partition."""
    if metric == "euclidean":
        Z = linkage(X, method="ward")
    else:
        D = squareform(pdist(X, metric="cosine"))
        Z = linkage(squareform(D, checks=False), method="average")
    cl = fcluster(Z, t=k, criterion="maxclust")
    truth = np.array([partition_dict[lab] for lab in labels])
    return adjusted_rand_score(truth, cl), cl, Z


def plot_dendrogram(Z, labels, title, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=11)
    ax.set_title(title)
    ax.set_ylabel("Linkage distance")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def section_9_1(df_ac, emb_wh, emb_xl, index):
    print("=== 9.1 Vowel clustering ===")
    X_ac, labs = acoustic_centroid_per_phoneme(df_ac, ORAL_VOWELS)
    X_wh, _ = neural_centroid_per_phoneme(emb_wh, index["phoneme"].to_numpy(), labs)
    X_xl, _ = neural_centroid_per_phoneme(emb_xl, index["phoneme"].to_numpy(), labs)

    rows = []
    for name, X, metric in [
        ("acoustic_F1F2", X_ac, "euclidean"),
        ("whisper_L20", X_wh, "cosine"),
        ("xlsr_L12", X_xl, "cosine"),
    ]:
        # Front/back: 3 clusters expected
        ari_fb, _, Z = cluster_and_score(X, labs, "front_back",
                                         FRONT_BACK, metric, k=3)
        # High/low: 3 clusters
        ari_hl, _, _ = cluster_and_score(X, labs, "high_low",
                                         HIGH_LOW, metric, k=3)
        rows.append({"representation": name,
                     "ARI_front_back": ari_fb, "ARI_high_low": ari_hl})
        print(f"  {name:14s}  ARI front/back={ari_fb:+.3f}  "
              f"ARI high/low={ari_hl:+.3f}")
        plot_dendrogram(Z, labs,
                        f"Vowel dendrogram — {name}",
                        FIG / f"fig_9_1_dendro_{name}.png")
    pd.DataFrame(rows).to_csv(DATA / "section9_1_vowels.csv", index=False)


def section_9_2(df_ac, emb_wh, emb_xl, index):
    print("\n=== 9.2 Consonants vs vowels ===")
    targets = ORAL_VOWELS + CONSONANTS
    X_ac, labs_ac = acoustic_centroid_per_phoneme(df_ac, targets)
    # For acoustic, supplement F1/F2 with duration and SCG (per the project spec)
    rows_ac = []
    kept_ac = []
    for p in targets:
        sub = df_ac[df_ac["phoneme"] == p]
        if len(sub) < 5:
            continue
        rows_ac.append([
            sub["F1_lobanov"].dropna().mean() if sub["F1_lobanov"].notna().any() else 0.0,
            sub["F2_lobanov"].dropna().mean() if sub["F2_lobanov"].notna().any() else 0.0,
            sub["duration_ms"].mean(),
            sub["scg_mean"].dropna().mean() if sub["scg_mean"].notna().any() else 0.0,
        ])
        kept_ac.append(p)
    X_ac = np.array(rows_ac)
    # standardise columns so duration and SCG don't dominate
    X_ac = (X_ac - X_ac.mean(axis=0)) / (X_ac.std(axis=0) + 1e-9)

    X_wh, kept_wh = neural_centroid_per_phoneme(emb_wh, index["phoneme"].to_numpy(), targets)
    X_xl, kept_xl = neural_centroid_per_phoneme(emb_xl, index["phoneme"].to_numpy(), targets)

    cv_truth = {p: ("V" if p in ORAL_VOWELS else "C") for p in targets}
    rows = []
    for name, X, labs, metric in [
        ("acoustic_extended", X_ac, kept_ac, "euclidean"),
        ("whisper_L20", X_wh, kept_wh, "cosine"),
        ("xlsr_L12", X_xl, kept_xl, "cosine"),
    ]:
        ari, cl, Z = cluster_and_score(X, labs, "C_V", cv_truth, metric, k=2)
        # Confusion of cluster vs C/V truth
        rows.append({"representation": name, "n_phonemes": len(labs),
                     "ARI_C_vs_V": ari})
        print(f"  {name:18s}  ARI C/V={ari:+.3f}  (n={len(labs)})")
        plot_dendrogram(Z, labs,
                        f"Consonants vs vowels — {name}",
                        FIG / f"fig_9_2_dendro_{name}.png")
    pd.DataFrame(rows).to_csv(DATA / "section9_2_cv.csv", index=False)


def section_9_3(df_ac, emb_wh, emb_xl, index):
    print("\n=== 9.3 Speaker clustering ===")
    speakers = sorted(df_ac["speaker"].unique())
    L1_truth = df_ac.drop_duplicates("speaker").set_index("speaker")["L1"].to_dict()
    G_truth = df_ac.drop_duplicates("speaker").set_index("speaker")["gender"].to_dict()

    # Build per-speaker feature: concatenated per-vowel mean over ORAL_VOWELS.
    # Speakers missing a vowel get NaN; we fill with the global mean for that vowel
    # so missing values do not destabilise the distance.
    def build_acoustic(df_ac):
        out = []
        for s in speakers:
            row = []
            for v in ORAL_VOWELS:
                sub = df_ac[(df_ac["speaker"] == s) & (df_ac["phoneme"] == v)]
                if len(sub) > 0:
                    row.extend([sub["F1_lobanov"].mean(), sub["F2_lobanov"].mean()])
                else:
                    row.extend([np.nan, np.nan])
            out.append(row)
        X = np.array(out)
        # Fill column means
        col_means = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])
        return X

    def build_neural(emb):
        labels = index["phoneme"].to_numpy()
        spk_arr = index["speaker"].to_numpy()
        out = []
        for s in speakers:
            row = []
            for v in ORAL_VOWELS:
                m = (spk_arr == s) & (labels == v)
                if m.sum() > 0:
                    row.extend(emb[m].mean(axis=0).tolist())
                else:
                    row.extend([np.nan] * emb.shape[1])
            out.append(row)
        X = np.array(out)
        col_means = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])
        return X

    X_ac = build_acoustic(df_ac)
    X_wh = build_neural(emb_wh)
    X_xl = build_neural(emb_xl)

    truth_l1 = np.array([L1_truth[s] for s in speakers])
    truth_g  = np.array([G_truth[s]  for s in speakers])

    rows = []
    for name, X, metric in [
        ("acoustic", X_ac, "euclidean"),
        ("whisper_L20", X_wh, "cosine"),
        ("xlsr_L12", X_xl, "cosine"),
    ]:
        if metric == "euclidean":
            Z = linkage(X, method="ward")
        else:
            D = squareform(pdist(X, metric="cosine"))
            Z = linkage(squareform(D, checks=False), method="average")
        cl2 = fcluster(Z, t=2, criterion="maxclust")
        ari_l1 = adjusted_rand_score(truth_l1, cl2)
        ari_g  = adjusted_rand_score(truth_g, cl2)
        rows.append({"representation": name,
                     "ARI_L1": ari_l1, "ARI_gender": ari_g})
        print(f"  {name:14s}  ARI vs L1={ari_l1:+.3f}  ARI vs gender={ari_g:+.3f}")
        plot_dendrogram(Z, speakers,
                        f"Speaker dendrogram — {name}",
                        FIG / f"fig_9_3_dendro_{name}.png")
    pd.DataFrame(rows).to_csv(DATA / "section9_3_speakers.csv", index=False)


def section_9_4(df_ac, emb_xl, index):
    """Number-of-clusters justification on XLS-R L12 vowel centroids
    (the strongest representation by previous sections)."""
    print("\n=== 9.4 Number of clusters (silhouette) ===")
    X_xl, labs = neural_centroid_per_phoneme(emb_xl,
                                              index["phoneme"].to_numpy(),
                                              ORAL_VOWELS)
    D = squareform(pdist(X_xl, metric="cosine"))
    Z = linkage(squareform(D, checks=False), method="average")
    rows = []
    for k in range(2, min(9, len(labs))):
        cl = fcluster(Z, t=k, criterion="maxclust")
        if len(set(cl)) < 2:
            continue
        sil = silhouette_score(D, cl, metric="precomputed")
        rows.append({"k": k, "silhouette": sil})
        print(f"  k={k}  silhouette={sil:+.3f}")
    pd.DataFrame(rows).to_csv(DATA / "section9_4_k.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    df_sil = pd.DataFrame(rows)
    ax.plot(df_sil["k"], df_sil["silhouette"], "o-")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette coefficient (cosine)")
    ax.set_title("Choosing k on XLS-R L12 vowel centroids")
    ax.axhline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_9_4_silhouette.png", dpi=150)
    plt.close(fig)


def main():
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    index = pd.read_csv(DATA / "features_neural_index.csv")
    emb_wh = np.load(DATA / "features_whisper_L20.npz")["embeddings"]
    emb_xl = np.load(DATA / "features_xlsr_L12.npz")["embeddings"]

    section_9_1(df_ac, emb_wh, emb_xl, index)
    section_9_2(df_ac, emb_wh, emb_xl, index)
    section_9_3(df_ac, emb_wh, emb_xl, index)
    section_9_4(df_ac, emb_xl, index)


if __name__ == "__main__":
    main()