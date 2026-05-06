"""Stage 4: Normalisation and dimensionality reduction.

Two parts:
  - Lobanov normalisation of F1/F2 (and F3) per speaker, computed
    only on vowel tokens, then applied back to all tokens of that
    speaker.
  - PCA (d=2 and d=50) and UMAP (d=2) on each of the 5 neural
    layer representations.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# French oral + nasal vowels for Lobanov
VOWELS = {"i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "œ", "ə",
          "ɑ̃", "ɛ̃", "œ̃", "ɔ̃"}

NEURAL_FILES = {
    "whisper_L06": "features_whisper_L06.npz",
    "whisper_L20": "features_whisper_L20.npz",
    "xlsr_L03":    "features_xlsr_L03.npz",
    "xlsr_L12":    "features_xlsr_L12.npz",
    "xlsr_L20":    "features_xlsr_L20.npz",
}


def lobanov_normalise(df):
    """Per-speaker z-score of F1, F2, F3, computed over vowel tokens."""
    out = df.copy()
    is_vowel = out["phoneme"].isin(VOWELS)

    for col in ["F1_mid", "F2_mid", "F3_mid"]:
        norm_col = col.replace("_mid", "_lobanov")
        out[norm_col] = np.nan
        for spk, idx in out.groupby("speaker").groups.items():
            sub = out.loc[idx]
            vowel_mask = sub["phoneme"].isin(VOWELS)
            ref = sub.loc[vowel_mask, col].dropna()
            if len(ref) < 5:
                continue
            mu, sigma = ref.mean(), ref.std(ddof=1)
            if sigma == 0 or np.isnan(sigma):
                continue
            out.loc[idx, norm_col] = (sub[col] - mu) / sigma
    return out


def reduce_neural(name, npz_path):
    """Compute PCA(2), PCA(50), UMAP(2) for one neural layer."""
    arr = np.load(npz_path)["embeddings"].astype(np.float32)
    print(f"  [{name}]  input shape {arr.shape}")

    pca2 = PCA(n_components=2, random_state=0).fit_transform(arr)
    pca50 = PCA(n_components=50, random_state=0).fit_transform(arr)
    reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=30,
                        min_dist=0.1, metric="cosine")
    ump2 = reducer.fit_transform(arr)

    np.savez(DATA / f"reduced_{name}_pca2.npz", embeddings=pca2)
    np.savez(DATA / f"reduced_{name}_pca50.npz", embeddings=pca50)
    np.savez(DATA / f"reduced_{name}_umap2.npz", embeddings=ump2)

    pca50_var = PCA(n_components=50, random_state=0).fit(arr).explained_variance_ratio_.sum()
    print(f"  [{name}]  PCA50 variance kept: {pca50_var*100:.1f}%")


def main():
    print("=== Lobanov normalisation ===")
    df = pd.read_csv(DATA / "features_acoustic.csv")
    df_norm = lobanov_normalise(df)
    out_csv = DATA / "features_acoustic_norm.csv"
    df_norm.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.name}")

    n_normalised = df_norm["F1_lobanov"].notna().sum()
    print(f"Tokens with normalised F1: {n_normalised} / {len(df_norm)}")

    print()
    print("=== Dimensionality reduction (PCA + UMAP) ===")
    for name, fname in NEURAL_FILES.items():
        reduce_neural(name, DATA / fname)


if __name__ == "__main__":
    main()