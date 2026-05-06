"""Section 5.2: Descriptive stats on neural representations."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

LAYERS = ["whisper_L06", "whisper_L20", "xlsr_L03", "xlsr_L12", "xlsr_L20"]
ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]


def between_class_variance_ratio(coords, labels):
    """Ratio of between-class variance to total variance (in 2D)."""
    df = pd.DataFrame(coords, columns=["d1", "d2"])
    df["lab"] = labels
    overall = df[["d1", "d2"]].mean()
    n_total = len(df)
    between_ss = 0.0
    for lab, g in df.groupby("lab"):
        mu = g[["d1", "d2"]].mean()
        between_ss += len(g) * ((mu - overall) ** 2).sum()
    total_ss = ((df[["d1", "d2"]] - overall) ** 2).sum().sum()
    return between_ss / total_ss if total_ss > 0 else np.nan


def cosine_within_between(emb, labels):
    """Mean pairwise cosine similarity within same-label pairs vs across labels."""
    sim = cosine_similarity(emb)
    n = len(labels)
    labels = np.array(labels)
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    sims = sim[iu]
    mean_within = sims[same].mean()
    mean_between = sims[~same].mean()
    return mean_within, mean_between, mean_within / mean_between


def plot_layer(name, index, out_path):
    """6-panel figure: PCA & UMAP, each coloured by phoneme / L1 / gender."""
    pca = np.load(DATA / f"reduced_{name}_pca2.npz")["embeddings"]
    ump = np.load(DATA / f"reduced_{name}_umap2.npz")["embeddings"]

    sub_idx = index["phoneme"].isin(ORAL_VOWELS).to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row, (coords, method) in enumerate([(pca, "PCA"), (ump, "UMAP")]):
        for col, color_by in enumerate(["phoneme", "L1", "gender"]):
            ax = axes[row, col]
            if color_by == "phoneme":
                idx = sub_idx
                hue = index.loc[idx, "phoneme"].astype(str)
                hue_order = ORAL_VOWELS
            else:
                idx = np.ones(len(index), dtype=bool)
                hue = index.loc[idx, color_by].astype(str)
                hue_order = sorted(hue.unique())
            sns.scatterplot(x=coords[idx, 0], y=coords[idx, 1],
                            hue=hue, hue_order=hue_order,
                            palette="husl" if color_by == "phoneme" else "Set1",
                            s=10, alpha=0.6, linewidth=0, ax=ax,
                            legend="brief" if col == 2 else False)
            ax.set_title(f"{method} — coloured by {color_by}")
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            if col == 2:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
                          fontsize=8, title=color_by)
            sns.despine(ax=ax)

    fig.suptitle(f"Neural representations — {name}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    index = pd.read_csv(DATA / "features_neural_index.csv")
    print(f"Loaded index with {len(index)} tokens")

    rows_var = []
    rows_cos = []

    for name in LAYERS:
        print(f"\n=== {name} ===")
        out_fig = FIG / f"fig_5_2_{name}.png"
        plot_layer(name, index, out_fig)
        print(f"  saved {out_fig.name}")

        # Between-class variance ratio (in 2D PCA & UMAP, on vowels)
        sub_idx = index["phoneme"].isin(ORAL_VOWELS).to_numpy()
        labels = index.loc[sub_idx, "phoneme"].to_numpy()
        for method in ["pca2", "umap2"]:
            coords = np.load(DATA / f"reduced_{name}_{method}.npz")["embeddings"]
            r = between_class_variance_ratio(coords[sub_idx], labels)
            rows_var.append({"layer": name, "method": method, "between_var_ratio": r})
            print(f"  {method:5s}  between/total var = {r:.3f}")

        # Cosine within / between on full embeddings (vowels only)
        emb = np.load(DATA / f"features_{name}.npz")["embeddings"]
        w, b, ratio = cosine_within_between(emb[sub_idx], labels)
        rows_cos.append({"layer": name,
                         "within_phoneme": w,
                         "between_phoneme": b,
                         "ratio_within_between": ratio})
        print(f"  cosine within={w:.4f}  between={b:.4f}  ratio={ratio:.4f}")

    pd.DataFrame(rows_var).to_csv(DATA / "section5_2_variance_ratio.csv", index=False)
    pd.DataFrame(rows_cos).to_csv(DATA / "section5_2_cosine_ratio.csv", index=False)
    print("\nWrote section5_2_variance_ratio.csv and section5_2_cosine_ratio.csv")


if __name__ == "__main__":
    main()