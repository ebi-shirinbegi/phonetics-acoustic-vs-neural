"""Section 6.1: L1 vs L2 group comparisons."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics.pairwise import cosine_distances

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]


def normality_ok(values, alpha=0.05):
    if len(values) < 3 or len(values) > 5000:
        return False
    try:
        return stats.shapiro(values).pvalue > alpha
    except Exception:
        return False


def acoustic_l1_l2_tests(df):
    rows = []
    for vowel in ORAL_VOWELS:
        sub = df[df["phoneme"] == vowel]
        for col in ["F1_lobanov", "F2_lobanov"]:
            l1_vals = sub[sub["L1"] == "fr"][col].dropna().to_numpy()
            l2_vals = sub[sub["L1"] == "ru"][col].dropna().to_numpy()
            if len(l1_vals) < 5 or len(l2_vals) < 5:
                continue
            normal_l1 = normality_ok(l1_vals)
            normal_l2 = normality_ok(l2_vals)
            equal_var = stats.levene(l1_vals, l2_vals).pvalue > 0.05
            if normal_l1 and normal_l2:
                t = stats.ttest_ind(l1_vals, l2_vals, equal_var=equal_var)
                test = "t-test" if equal_var else "Welch"
                stat, p = t.statistic, t.pvalue
            else:
                u = stats.mannwhitneyu(l1_vals, l2_vals, alternative="two-sided")
                test, stat, p = "Mann-Whitney", u.statistic, u.pvalue
            rows.append({
                "vowel": vowel, "feature": col,
                "n_L1": len(l1_vals), "n_L2": len(l2_vals),
                "mean_L1": l1_vals.mean(), "mean_L2": l2_vals.mean(),
                "test": test, "stat": stat, "p_raw": p,
            })
    out = pd.DataFrame(rows)
    if len(out):
        rej, p_adj, _, _ = multipletests(out["p_raw"], method="fdr_bh")
        out["p_fdr"] = p_adj
        out["sig_fdr"] = rej
    return out


def gender_test(df):
    rows = []
    for vowel in ORAL_VOWELS:
        sub = df[df["phoneme"] == vowel]
        for col in ["F1_lobanov", "F2_lobanov"]:
            spk_means = (sub.groupby(["speaker", "gender"])[col]
                            .mean().reset_index())
            f_vals = spk_means[spk_means["gender"] == "f"][col].dropna()
            m_vals = spk_means[spk_means["gender"] == "m"][col].dropna()
            if len(f_vals) < 3 or len(m_vals) < 3:
                continue
            u = stats.mannwhitneyu(f_vals, m_vals, alternative="two-sided")
            rows.append({
                "vowel": vowel, "feature": col,
                "n_F": len(f_vals), "n_M": len(m_vals),
                "median_F": f_vals.median(), "median_M": m_vals.median(),
                "stat": u.statistic, "p_raw": u.pvalue,
            })
    out = pd.DataFrame(rows)
    if len(out):
        rej, p_adj, _, _ = multipletests(out["p_raw"], method="fdr_bh")
        out["p_fdr"] = p_adj
        out["sig_fdr"] = rej
    return out


def permutation_centroid(emb_l1, emb_l2, n_perm=5000, seed=0):
    c1 = emb_l1.mean(axis=0, keepdims=True)
    c2 = emb_l2.mean(axis=0, keepdims=True)
    obs = float(cosine_distances(c1, c2)[0, 0])
    pooled = np.vstack([emb_l1, emb_l2])
    n1 = len(emb_l1)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        a = pooled[idx[:n1]].mean(axis=0, keepdims=True)
        b = pooled[idx[n1:]].mean(axis=0, keepdims=True)
        d = float(cosine_distances(a, b)[0, 0])
        if d >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return obs, p


def neural_l1_l2_tests(emb, labels_phon, labels_l1, layer_name):
    rows = []
    for vowel in ORAL_VOWELS:
        mask = labels_phon == vowel
        e1 = emb[mask & (labels_l1 == "fr")]
        e2 = emb[mask & (labels_l1 == "ru")]
        if len(e1) < 10 or len(e2) < 10:
            continue
        d, p = permutation_centroid(e1, e2, n_perm=5000)
        rows.append({
            "layer": layer_name, "vowel": vowel,
            "n_L1": len(e1), "n_L2": len(e2),
            "centroid_cosine_dist": d, "p_raw": p,
        })
    out = pd.DataFrame(rows)
    if len(out):
        rej, p_adj, _, _ = multipletests(out["p_raw"], method="fdr_bh")
        out["p_fdr"] = p_adj
        out["sig_fdr"] = rej
    return out


def main():
    print("=== Acoustic L1/L2 tests ===")
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    ac = acoustic_l1_l2_tests(df_ac)
    ac.to_csv(DATA / "section6_1_acoustic.csv", index=False)
    print(ac[["vowel", "feature", "test", "p_raw", "p_fdr", "sig_fdr"]]
            .round(4).to_string(index=False))
    print(f"  Significant after FDR: {ac['sig_fdr'].sum()}/{len(ac)}")

    print("\n=== Gender effect (speaker-level, after Lobanov) ===")
    g = gender_test(df_ac)
    g.to_csv(DATA / "section6_1_gender.csv", index=False)
    print(g[["vowel", "feature", "p_raw", "p_fdr", "sig_fdr"]]
            .round(4).to_string(index=False))
    print(f"  Significant after FDR: {g['sig_fdr'].sum()}/{len(g)}")

    print("\n=== Neural L1/L2 tests ===")
    index = pd.read_csv(DATA / "features_neural_index.csv")
    labels_phon = index["phoneme"].to_numpy()
    labels_l1 = index["L1"].to_numpy()

    all_neural = []
    for layer in ["whisper_L20", "xlsr_L12"]:
        emb = np.load(DATA / f"features_{layer}.npz")["embeddings"]
        out = neural_l1_l2_tests(emb, labels_phon, labels_l1, layer)
        all_neural.append(out)
        print(f"\n  --- {layer} ---")
        print(out[["vowel", "centroid_cosine_dist", "p_raw", "p_fdr", "sig_fdr"]]
                .round(4).to_string(index=False))
        print(f"    Significant after FDR: {out['sig_fdr'].sum()}/{len(out)}")
    pd.concat(all_neural).to_csv(DATA / "section6_1_neural.csv", index=False)


if __name__ == "__main__":
    main()
