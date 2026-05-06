"""Section 7 (robust refit): clean LR statistics for the pooled headline models.

For each of the 6 headline responses (F1 pooled, Whisper L20 PC1-PC2,
XLS-R L12 PC1-PC2-PC3), I fit M0 -> M4 with multiple optimizer starts
and keep the best ML log-likelihood. This eliminates the spurious
negative LR statistics that come from local optima.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]
HEIGHT = {
    "i": "high", "y": "high", "u": "high",
    "e": "mid",  "ø": "mid", "o": "mid", "ə": "mid",
    "ɛ": "mid",
    "a": "low",  "ɑ": "low",
}

OPTIMIZERS = ["lbfgs", "bfgs", "powell", "nm"]


def add_design(df):
    df = df.copy()
    df["L2"] = (df["L1"].str.lower() == "ru").astype(int)
    df["Male"] = (df["gender"].str.lower() == "m").astype(int)
    df["height"] = df["phoneme"].map(HEIGHT)
    return df


def fit_best(formula, data, re_formula=None, n_restarts=4):
    """Try several optimizers and return the fit with highest log-likelihood."""
    best = None
    for opt in OPTIMIZERS[:n_restarts]:
        try:
            kwargs = {"data": data, "groups": "speaker"}
            if re_formula is not None:
                kwargs["re_formula"] = re_formula
            res = smf.mixedlm(formula, **kwargs).fit(reml=False, method=[opt])
            if best is None or res.llf > best.llf:
                best = res
        except Exception:
            continue
    return best


def lr(small, big, df_diff):
    if small is None or big is None:
        return np.nan, np.nan
    stat = 2 * (big.llf - small.llf)
    if stat < 0:
        # If still negative after restarts, both fits are local optima at
        # the same level — treat as "no improvement" rather than reporting
        # a nonsensical negative chi^2.
        return 0.0, 1.0
    return stat, chi2.sf(stat, df_diff)


def icc(res):
    su = float(res.cov_re.iloc[0, 0])
    se = float(res.scale)
    return su / (su + se) if (su + se) > 0 else np.nan


def r2_marg_cond(res):
    var_fixed = float(np.var(res.fittedvalues, ddof=0))
    var_random = float(np.trace(res.cov_re))
    var_resid = float(res.scale)
    total = var_fixed + var_random + var_resid
    if total <= 0:
        return np.nan, np.nan
    return var_fixed / total, (var_fixed + var_random) / total


def run_one(label, data, y, store):
    print(f"\n--- {label}  (n={len(data)}, y={y}) ---")
    has_height = data["height"].nunique() > 1

    m0 = fit_best(f"{y} ~ 1", data)
    m1 = fit_best(f"{y} ~ L2 + Male", data)
    m2 = fit_best(f"{y} ~ L2 * Male", data)
    m3 = fit_best(f"{y} ~ L2 * Male + C(height)", data) if has_height else m2
    try:
        m4 = fit_best(
            f"{y} ~ L2 * Male" + (" + C(height)" if has_height else ""),
            data, re_formula="~L2")
    except Exception:
        m4 = None

    s10, p10 = lr(m0, m1, 2)
    s21, p21 = lr(m1, m2, 1)
    s32, p32 = lr(m2, m3, data["height"].nunique() - 1) if has_height else (np.nan, np.nan)
    s43, p43 = lr(m3, m4, 2)

    icc0 = icc(m0) if m0 else np.nan
    r2m_m2, r2c_m2 = r2_marg_cond(m2) if m2 else (np.nan, np.nan)
    r2m_m3, r2c_m3 = r2_marg_cond(m3) if m3 else (np.nan, np.nan)

    print(f"  M0 ll={m0.llf:9.2f}   ICC={icc0:.3f}")
    print(f"  M1 ll={m1.llf:9.2f}   LR vs M0: chi2={s10:6.2f} p={p10:.4f}")
    print(f"  M2 ll={m2.llf:9.2f}   LR vs M1: chi2={s21:6.2f} p={p21:.4f}   "
          f"R2m={r2m_m2:.3f} R2c={r2c_m2:.3f}")
    if has_height:
        print(f"  M3 ll={m3.llf:9.2f}   LR vs M2: chi2={s32:6.2f} p={p32:.4f}   "
              f"R2m={r2m_m3:.3f} R2c={r2c_m3:.3f}")
    if m4 is not None:
        print(f"  M4 ll={m4.llf:9.2f}   LR vs M3: chi2={s43:6.2f} p={p43:.4f}")
    else:
        print(f"  M4 FAILED")

    row = {
        "label": label, "n": len(data), "ICC": icc0,
        "ll_M0": m0.llf if m0 else np.nan,
        "ll_M1": m1.llf if m1 else np.nan,
        "ll_M2": m2.llf if m2 else np.nan,
        "ll_M3": m3.llf if has_height and m3 else np.nan,
        "ll_M4": m4.llf if m4 else np.nan,
        "LR_M1_M0": s10, "p_M1_M0": p10,
        "LR_M2_M1": s21, "p_M2_M1": p21,
        "LR_M3_M2": s32, "p_M3_M2": p32,
        "LR_M4_M3": s43, "p_M4_M3": p43,
        "R2m_M2": r2m_m2, "R2c_M2": r2c_m2,
        "R2m_M3": r2m_m3, "R2c_M3": r2c_m3,
    }
    if m2 is not None:
        for term in ["L2", "Male", "L2:Male"]:
            row[f"{term}_beta_M2"] = m2.params.get(term, np.nan)
            row[f"{term}_p_M2"] = m2.pvalues.get(term, np.nan)
    store.append(row)


def main():
    # --- Acoustic ---
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df_ac = add_design(df_ac)
    df_ac = df_ac[df_ac["phoneme"].isin(ORAL_VOWELS)].dropna(subset=["F1_lobanov"])

    store = []
    print("\n=== ACOUSTIC (robust) ===")
    run_one("F1 pooled", df_ac, "F1_lobanov", store)

    pd.DataFrame(store).to_csv(DATA / "section7_robust_acoustic.csv", index=False)
    print("\nWrote section7_robust_acoustic.csv")

    # --- Neural ---
    index = pd.read_csv(DATA / "features_neural_index.csv")
    index = add_design(index)
    mask = index["phoneme"].isin(ORAL_VOWELS).to_numpy()

    for layer in ["whisper_L20", "xlsr_L12"]:
        emb = np.load(DATA / f"features_{layer}.npz")["embeddings"]
        emb_v = emb[mask]
        idx_v = index[mask].reset_index(drop=True)
        pcs = PCA(n_components=5, random_state=0).fit_transform(emb_v)
        for k in range(5):
            idx_v[f"PC{k+1}"] = pcs[:, k]

        store = []
        print(f"\n=== NEURAL {layer} (robust) ===")
        # Top 3 PCs are enough as headline; PC4-PC5 carry less variance
        for k in range(3):
            run_one(f"{layer} PC{k+1}", idx_v, f"PC{k+1}", store)

        pd.DataFrame(store).to_csv(DATA / f"section7_robust_{layer}.csv", index=False)
        print(f"\nWrote section7_robust_{layer}.csv")


if __name__ == "__main__":
    main()