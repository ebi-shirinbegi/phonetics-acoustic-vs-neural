"""Section 5.3: Representational Similarity Matrices and Mantel test."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]


def centroid_matrix_neural(emb, labels, classes):
    rows = []
    kept = []
    for c in classes:
        mask = labels == c
        if mask.sum() < 5:
            continue
        rows.append(emb[mask].mean(axis=0))
        kept.append(c)
    return np.vstack(rows), kept


def centroid_matrix_acoustic(df, classes):
    rows = []
    kept = []
    for c in classes:
        sub = df[(df["phoneme"] == c) &
                 df["F1_lobanov"].notna() & df["F2_lobanov"].notna()]
        if len(sub) < 5:
            continue
        rows.append([sub["F1_lobanov"].mean(), sub["F2_lobanov"].mean()])
        kept.append(c)
    return np.array(rows), kept


def cosine_distance_matrix(X):
    return squareform(pdist(X, metric="cosine"))


def euclidean_distance_matrix(X):
    return squareform(pdist(X, metric="euclidean"))


def mantel_test(D1, D2, n_perm=5000, seed=0):
    """Spearman rank correlation between upper triangles of two distance
    matrices. p-value via permutation of rows/cols of D2."""
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    v1 = D1[iu]
    v2 = D2[iu]
    r_obs, _ = spearmanr(v1, v2)

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        r_perm, _ = spearmanr(v1, D2p[iu])
        if abs(r_perm) >= abs(r_obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return r_obs, p


def main():
    index = pd.read_csv(DATA / "features_neural_index.csv")
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")

    # Build the three centroid matrices over the same vowel set
    X_ac, vowels = centroid_matrix_acoustic(df_ac, ORAL_VOWELS)
    print(f"Vowels kept: {vowels}")

    emb_wh = np.load(DATA / "features_whisper_L20.npz")["embeddings"]
    emb_xl = np.load(DATA / "features_xlsr_L12.npz")["embeddings"]
    labels = index["phoneme"].to_numpy()

    X_wh, _ = centroid_matrix_neural(emb_wh, labels, vowels)
    X_xl, _ = centroid_matrix_neural(emb_xl, labels, vowels)

    D_ac = euclidean_distance_matrix(X_ac)
    D_wh = cosine_distance_matrix(X_wh)
    D_xl = cosine_distance_matrix(X_xl)

    print("\n=== Mantel tests (Spearman, 5000 permutations) ===")
    pairs = [("Acoustic", "Whisper L20", D_ac, D_wh),
             ("Acoustic", "XLS-R L12",   D_ac, D_xl),
             ("Whisper L20", "XLS-R L12", D_wh, D_xl)]
    rows = []
    for n1, n2, A, B in pairs:
        r, p = mantel_test(A, B, n_perm=5000)
        rows.append({"a": n1, "b": n2, "mantel_r": r, "p_value": p})
        print(f"  {n1:14s} vs {n2:14s}  r={r:+.3f}  p={p:.4f}")

    pd.DataFrame(rows).to_csv(DATA / "section5_3_mantel.csv", index=False)
    print("\nWrote section5_3_mantel.csv")

    # Visualise the three RSMs side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mat, title in zip(axes, [D_ac, D_wh, D_xl],
                              ["Acoustic (Euclidean)",
                               "Whisper L20 (cosine)",
                               "XLS-R L12 (cosine)"]):
        sns.heatmap(mat, ax=ax, xticklabels=vowels, yticklabels=vowels,
                    cmap="rocket_r", square=True, cbar_kws={"shrink": 0.7})
        ax.set_title(title)
    fig.suptitle("Representational Distance Matrices over French oral vowels", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_5_3_rsm.png", dpi=150)
    plt.close(fig)
    print("Saved fig_5_3_rsm.png")


if __name__ == "__main__":
    main()