"""Section 7: Linear Mixed-Effects models on acoustic and neural responses.

I follow the project's model-building sequence:
  M0  null:           y ~ 1 + (1|speaker)
  M1  main effects:   y ~ L2 + Male + (1|speaker)
  M2  full:           y ~ L2 * Male + (1|speaker)
  M3  + context:      y ~ L2 * Male + height + (1|speaker)
  M4  random slope:   y ~ L2 * Male + height + (1 + L2|speaker)

ML estimation throughout for likelihood-ratio tests.
For the acoustic side, I model F1_lobanov on each vowel separately and on
the pooled vowel set (with vowel as a fixed effect). For the neural side,
I project Whisper L20 and XLS-R L12 to 5 PCs and fit one LME per PC.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")  # statsmodels emits many ConvergenceWarnings

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]
HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid", "o": "mid", "ə": "mid",
    "ɛ": "mid",
    "a": "low",  "ɑ": "low",
}


def add_design_columns(df):
    df = df.copy()
    df["L2"] = (df["L1"].str.lower() == "ru").astype(int)
    df["Male"] = (df["gender"].str.lower() == "m").astype(int)
    df["height"] = df["phoneme"].map(HEIGHT)
    return df


def fit_lme(formula, data, vc=None, re_formula=None):
    """Fit an LME with ML. vc/re_formula optional."""
    kwargs = {"data": data, "groups": "speaker"}
    if re_formula is not None:
        kwargs["re_formula"] = re_formula
    model = smf.mixedlm(formula, **kwargs)
    return model.fit(reml=False, method=["lbfgs"])


def loglik(res):
    return res.llf


def lr_test(small, big, df_diff):
    """Likelihood-ratio test: 2*(LL_big - LL_small) ~ chi2(df_diff)."""
    stat = 2 * (big.llf - small.llf)
    p = chi2.sf(stat, df_diff) if stat > 0 else 1.0
    return stat, p


def icc_from_null(null_res):
    """ICC = sigma_u^2 / (sigma_u^2 + sigma^2). Statsmodels stores
    the random intercept variance under cov_re and the residual under scale."""
    sigma_u2 = float(null_res.cov_re.iloc[0, 0])
    sigma2 = float(null_res.scale)
    return sigma_u2 / (sigma_u2 + sigma2)


def marginal_conditional_r2(res, data, y_col):
    """Nakagawa & Schielzeth-style R^2 for mixed models.
    marginal: variance of fixed-effect predictions / total variance
    conditional: (var_fixed + var_random) / total variance
    """
    fitted_fixed = res.fittedvalues
    var_fixed = float(np.var(fitted_fixed, ddof=0))
    var_random = float(res.cov_re.iloc[0, 0])
    var_resid = float(res.scale)
    total = var_fixed + var_random + var_resid
    if total <= 0:
        return np.nan, np.nan
    return var_fixed / total, (var_fixed + var_random) / total


def run_lme_sequence(data, y_col, label, store):
    """Run M0 to M4 on the given data with response y_col. Append rows to store."""
    print(f"\n--- {label}  (n={len(data)}, response={y_col}) ---")
    has_height = data["height"].nunique() > 1

    # M0: null
    m0 = fit_lme(f"{y_col} ~ 1", data)
    icc = icc_from_null(m0)
    print(f"  M0 null:           ll={m0.llf:.2f}   ICC={icc:.3f}")

    # M1: main effects
    m1 = fit_lme(f"{y_col} ~ L2 + Male", data)
    s1, p1 = lr_test(m0, m1, df_diff=2)
    r2m1_m, r2m1_c = marginal_conditional_r2(m1, data, y_col)
    print(f"  M1 main:           ll={m1.llf:.2f}   "
          f"LR vs M0: chi2={s1:.2f} p={p1:.4f}   "
          f"R2m={r2m1_m:.3f} R2c={r2m1_c:.3f}")

    # M2: full (interaction)
    m2 = fit_lme(f"{y_col} ~ L2 * Male", data)
    s2, p2 = lr_test(m1, m2, df_diff=1)
    r2m2_m, r2m2_c = marginal_conditional_r2(m2, data, y_col)
    print(f"  M2 full:           ll={m2.llf:.2f}   "
          f"LR vs M1: chi2={s2:.2f} p={p2:.4f}   "
          f"R2m={r2m2_m:.3f} R2c={r2m2_c:.3f}")

    # M3: extended with context (only if multiple heights present)
    if has_height:
        m3 = fit_lme(f"{y_col} ~ L2 * Male + C(height)", data)
        s3, p3 = lr_test(m2, m3, df_diff=data["height"].nunique() - 1)
        r2m3_m, r2m3_c = marginal_conditional_r2(m3, data, y_col)
        print(f"  M3 + height:       ll={m3.llf:.2f}   "
              f"LR vs M2: chi2={s3:.2f} p={p3:.4f}   "
              f"R2m={r2m3_m:.3f} R2c={r2m3_c:.3f}")
    else:
        m3 = m2
        s3 = p3 = np.nan
        r2m3_m = r2m2_m
        r2m3_c = r2m2_c

    # M4: random slope for L2 by speaker
    try:
        m4 = fit_lme(f"{y_col} ~ L2 * Male" + (" + C(height)" if has_height else ""),
                     data, re_formula="~L2")
        s4, p4 = lr_test(m3, m4, df_diff=2)
        # marginal/conditional with random slope: approximate using cov_re trace
        var_fixed = float(np.var(m4.fittedvalues, ddof=0))
        var_random = float(np.trace(m4.cov_re))
        var_resid = float(m4.scale)
        total = var_fixed + var_random + var_resid
        r2m4_m = var_fixed / total
        r2m4_c = (var_fixed + var_random) / total
        print(f"  M4 + L2 slope:     ll={m4.llf:.2f}   "
              f"LR vs M3: chi2={s4:.2f} p={p4:.4f}   "
              f"R2m={r2m4_m:.3f} R2c={r2m4_c:.3f}")
    except Exception as e:
        s4 = p4 = np.nan
        r2m4_m = r2m4_c = np.nan
        print(f"  M4 + L2 slope:     FAILED ({type(e).__name__})")

    store.append({
        "label": label, "n": len(data),
        "ICC": icc,
        "ll_M0": m0.llf, "ll_M1": m1.llf, "ll_M2": m2.llf,
        "ll_M3": (m3.llf if has_height else np.nan),
        "lr_M1_vs_M0": s1, "p_M1_vs_M0": p1,
        "lr_M2_vs_M1": s2, "p_M2_vs_M1": p2,
        "lr_M3_vs_M2": s3, "p_M3_vs_M2": p3,
        "lr_M4_vs_M3": s4, "p_M4_vs_M3": p4,
        "R2m_M2": r2m2_m, "R2c_M2": r2m2_c,
        "R2m_M3": r2m3_m, "R2c_M3": r2m3_c,
        "L2_beta_M2": m2.params.get("L2", np.nan),
        "L2_p_M2": m2.pvalues.get("L2", np.nan),
        "Male_beta_M2": m2.params.get("Male", np.nan),
        "Male_p_M2": m2.pvalues.get("Male", np.nan),
        "L2xMale_beta_M2": m2.params.get("L2:Male", np.nan),
        "L2xMale_p_M2": m2.pvalues.get("L2:Male", np.nan),
    })


def acoustic_pipeline():
    df = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df = add_design_columns(df)
    df = df[df["phoneme"].isin(ORAL_VOWELS)]
    df = df.dropna(subset=["F1_lobanov"])

    store = []
    print("\n=== Acoustic LME ===")

    # Pooled model across all vowels (height becomes meaningful here)
    run_lme_sequence(df, "F1_lobanov", "F1 pooled", store)

    # Per-vowel model on the largest vowels (n >= 100)
    for v in ORAL_VOWELS:
        sub = df[df["phoneme"] == v]
        if len(sub) < 100:
            continue
        run_lme_sequence(sub, "F1_lobanov", f"F1 [{v}]", store)

    out = pd.DataFrame(store)
    out.to_csv(DATA / "section7_acoustic.csv", index=False)
    print(f"\nWrote section7_acoustic.csv ({len(out)} rows)")


def neural_pipeline():
    index = pd.read_csv(DATA / "features_neural_index.csv")
    index = add_design_columns(index)

    for layer in ["whisper_L20", "xlsr_L12"]:
        emb = np.load(DATA / f"features_{layer}.npz")["embeddings"]
        # Filter to vowels we care about
        mask = index["phoneme"].isin(ORAL_VOWELS).to_numpy()
        emb_v = emb[mask]
        index_v = index[mask].reset_index(drop=True)

        # PCA to 5 components, on vowel subset only
        pcs = PCA(n_components=5, random_state=0).fit_transform(emb_v)
        for k in range(5):
            index_v[f"PC{k+1}"] = pcs[:, k]

        store = []
        print(f"\n=== Neural LME — {layer} ===")
        for k in range(5):
            run_lme_sequence(index_v, f"PC{k+1}", f"{layer} PC{k+1}", store)

        out = pd.DataFrame(store)
        out.to_csv(DATA / f"section7_neural_{layer}.csv", index=False)
        print(f"\nWrote section7_neural_{layer}.csv")


def main():
    acoustic_pipeline()
    neural_pipeline()


if __name__ == "__main__":
    main()