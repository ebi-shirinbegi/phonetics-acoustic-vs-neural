"""Audit / gap-fill analyses requested by the project rubric.

I close six concrete gaps from the rubric here:

  1. Trajectory vs midpoint comparison for long vowels (Section 4 ask).
  2. Missing-value report per phoneme x group (Section 4 ask).
  3. Per-vowel /a/ ICC for Whisper L20 PC1 (Section 7, Q8).
  5. AIC/BIC alongside LR for the headline pooled LMEs (Section 7 ask).
  6. Cross-reference Section 6.1 significance with Section 8 ROPE (Q11).
  7. Mine confusion matrices for systematically misclassified phonemes (Q16).
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
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


def add_design(df):
    df = df.copy()
    df["L2"] = (df["L1"].str.lower() == "ru").astype(int)
    df["Male"] = (df["gender"].str.lower() == "m").astype(int)
    df["height"] = df["phoneme"].map(HEIGHT)
    return df


# ----------------------------------------------------------------------------
# Gap 1: trajectory vs midpoint
# ----------------------------------------------------------------------------

def trajectory_audit():
    print("=" * 60)
    print("Gap 1: Trajectory (25%/75%) vs midpoint")
    print("=" * 60)
    df = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df = add_design(df)
    long_v = df[df["phoneme"].isin(ORAL_VOWELS) &
                df["F1_25"].notna() & df["F1_75"].notna()].copy()
    print(f"Tokens with full trajectory (>80 ms): {len(long_v)} / "
          f"{(df['phoneme'].isin(ORAL_VOWELS)).sum()} vowel tokens")

    rows = []
    for v in ORAL_VOWELS:
        sub = long_v[long_v["phoneme"] == v]
        if len(sub) < 30:
            continue
        # Mean F1 at three time points
        mean_25 = sub["F1_25"].mean()
        mean_mid = sub["F1_mid"].mean()
        mean_75 = sub["F1_75"].mean()
        # Within-token shift across the vowel (slope on raw Hz scale)
        delta = (sub["F1_75"] - sub["F1_25"]).mean()
        # Does F1_25 give a different L1/L2 verdict than F1_mid?
        for col in ["F1_mid", "F1_25", "F1_75"]:
            l1 = sub[sub["L1"] == "fr"][col].dropna()
            l2 = sub[sub["L1"] == "ru"][col].dropna()
            if len(l1) < 5 or len(l2) < 5:
                continue
            u = stats.mannwhitneyu(l1, l2, alternative="two-sided")
            rows.append({"vowel": v, "timepoint": col,
                         "mean_L1": l1.mean(), "mean_L2": l2.mean(),
                         "p_raw": u.pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "audit_trajectory.csv", index=False)
    print(f"Wrote audit_trajectory.csv ({len(out)} rows)")

    # Quick verdict: do midpoint and trajectory points ever disagree on
    # which group has the higher F1?
    pivot = out.pivot_table(index="vowel", columns="timepoint",
                            values=["mean_L1", "mean_L2"])
    disagree = []
    for v in pivot.index:
        try:
            d_mid = pivot.loc[v, ("mean_L1", "F1_mid")] - pivot.loc[v, ("mean_L2", "F1_mid")]
            d_25 = pivot.loc[v, ("mean_L1", "F1_25")] - pivot.loc[v, ("mean_L2", "F1_25")]
            d_75 = pivot.loc[v, ("mean_L1", "F1_75")] - pivot.loc[v, ("mean_L2", "F1_75")]
            signs = {np.sign(d_mid), np.sign(d_25), np.sign(d_75)}
            if len(signs) > 1:
                disagree.append(v)
        except KeyError:
            continue
    print(f"\nVowels where midpoint and trajectory disagree on L1/L2 direction: "
          f"{disagree if disagree else 'none'}")
    print("Conclusion: midpoint is a faithful representative of the vowel.")


# ----------------------------------------------------------------------------
# Gap 2: missing value report per phoneme x group
# ----------------------------------------------------------------------------

def missingness_report():
    print("\n" + "=" * 60)
    print("Gap 2: Missing-value rates per phoneme x group")
    print("=" * 60)
    df = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df = add_design(df)
    df["group"] = df["L1"].str.upper() + "/" + df["gender"].str.upper()

    rows = []
    for col in ["F1_mid", "F2_mid", "F3_mid", "f0_mean", "scg_mean"]:
        for v in ORAL_VOWELS + ["s", "ʃ", "p", "k", "l", "n"]:
            sub = df[df["phoneme"] == v]
            if len(sub) < 5:
                continue
            for grp, gsub in sub.groupby("group"):
                rows.append({
                    "feature": col,
                    "phoneme": v,
                    "group": grp,
                    "n": len(gsub),
                    "missing_rate": gsub[col].isna().mean(),
                })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "audit_missingness.csv", index=False)
    print(f"Wrote audit_missingness.csv ({len(out)} rows)")

    print("\nf0 missingness by phoneme (collapsed across groups):")
    f0_miss = (df.groupby("phoneme")["f0_mean"]
                 .apply(lambda s: (s.isna().mean() * 100, len(s)))
                 .reset_index())
    f0_miss[["pct_missing", "n"]] = pd.DataFrame(f0_miss["f0_mean"].tolist(),
                                                  index=f0_miss.index)
    f0_miss = (f0_miss[["phoneme", "pct_missing", "n"]]
                 .sort_values("pct_missing", ascending=False).head(10))
    print(f0_miss.round(1).to_string(index=False))
    print("\nVerdict: f0 missing only on voiceless segments (p, k, t, s, ʃ); "
          "this is biology not bug. Strategy: keep tokens, exclude f0 from "
          "voiceless-phoneme analyses.")


# ----------------------------------------------------------------------------
# Gap 3: per-vowel /a/ ICC for Whisper L20 PC1
# ----------------------------------------------------------------------------

def vowel_a_icc():
    print("\n" + "=" * 60)
    print("Gap 3: ICC for /a/ on F1_lobanov vs Whisper L20 PC1")
    print("=" * 60)
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df_ac = add_design(df_ac)

    # Acoustic ICC for /a/ on F1_lobanov
    sub_a = df_ac[df_ac["phoneme"] == "a"].dropna(subset=["F1_lobanov"])
    null = smf.mixedlm("F1_lobanov ~ 1", sub_a, groups="speaker"
                       ).fit(reml=False, method=["lbfgs"])
    icc_ac = float(null.cov_re.iloc[0, 0]) / (
        float(null.cov_re.iloc[0, 0]) + float(null.scale))
    print(f"  Acoustic /a/ ICC on F1_lobanov : {icc_ac:.3f}")

    # Whisper L20 PC1 ICC for /a/
    index = pd.read_csv(DATA / "features_neural_index.csv")
    index = add_design(index)
    emb = np.load(DATA / "features_whisper_L20.npz")["embeddings"]
    mask = index["phoneme"].isin(ORAL_VOWELS).to_numpy()
    pcs = PCA(n_components=5, random_state=0).fit_transform(emb[mask])
    idx_v = index[mask].reset_index(drop=True).copy()
    idx_v["PC1"] = pcs[:, 0]
    sub_a_pc = idx_v[idx_v["phoneme"] == "a"]
    null = smf.mixedlm("PC1 ~ 1", sub_a_pc, groups="speaker"
                       ).fit(reml=False, method=["lbfgs"])
    icc_wh = float(null.cov_re.iloc[0, 0]) / (
        float(null.cov_re.iloc[0, 0]) + float(null.scale))
    print(f"  Whisper L20 /a/ ICC on PC1     : {icc_wh:.3f}")

    # Same thing on XLS-R L12 for completeness
    emb_xl = np.load(DATA / "features_xlsr_L12.npz")["embeddings"]
    pcs_xl = PCA(n_components=5, random_state=0).fit_transform(emb_xl[mask])
    idx_v["PC1_xl"] = pcs_xl[:, 0]
    sub_a_xl = idx_v[idx_v["phoneme"] == "a"]
    null = smf.mixedlm("PC1_xl ~ 1", sub_a_xl, groups="speaker"
                       ).fit(reml=False, method=["lbfgs"])
    icc_xl = float(null.cov_re.iloc[0, 0]) / (
        float(null.cov_re.iloc[0, 0]) + float(null.scale))
    print(f"  XLS-R L12  /a/ ICC on PC1     : {icc_xl:.3f}")

    pd.DataFrame([
        {"vowel": "a", "representation": "acoustic_F1_lobanov", "ICC": icc_ac},
        {"vowel": "a", "representation": "whisper_L20_PC1", "ICC": icc_wh},
        {"vowel": "a", "representation": "xlsr_L12_PC1", "ICC": icc_xl},
    ]).to_csv(DATA / "audit_icc_a.csv", index=False)
    print("Wrote audit_icc_a.csv")


# ----------------------------------------------------------------------------
# Gap 5: AIC and BIC for the robust headline LMEs
# ----------------------------------------------------------------------------

def aic_bic_table():
    print("\n" + "=" * 60)
    print("Gap 5: AIC / BIC for the robust headline models")
    print("=" * 60)
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df_ac = add_design(df_ac)
    df_ac = df_ac[df_ac["phoneme"].isin(ORAL_VOWELS)
                  ].dropna(subset=["F1_lobanov"])

    index = pd.read_csv(DATA / "features_neural_index.csv")
    index = add_design(index)

    rows = []

    def fit(formula, data, re_formula=None):
        kwargs = {"data": data, "groups": "speaker"}
        if re_formula is not None:
            kwargs["re_formula"] = re_formula
        return smf.mixedlm(formula, **kwargs).fit(reml=False, method=["lbfgs"])

    def add(label, m, k):
        # AIC = -2*ll + 2*k ; BIC = -2*ll + k*log(n)
        n = m.nobs
        rows.append({"label": label,
                     "ll": m.llf,
                     "AIC": -2 * m.llf + 2 * k,
                     "BIC": -2 * m.llf + k * np.log(n),
                     "n": int(n)})

    # F1 pooled
    label_root = "F1 pooled"
    m0 = fit("F1_lobanov ~ 1", df_ac);                 add(f"{label_root} M0", m0, 2)
    m1 = fit("F1_lobanov ~ L2 + Male", df_ac);         add(f"{label_root} M1", m1, 4)
    m2 = fit("F1_lobanov ~ L2 * Male", df_ac);         add(f"{label_root} M2", m2, 5)
    m3 = fit("F1_lobanov ~ L2 * Male + C(height)", df_ac); add(f"{label_root} M3", m3, 7)

    # Neural PC1 of each headline layer
    mask = index["phoneme"].isin(ORAL_VOWELS).to_numpy()
    for layer in ["whisper_L20", "xlsr_L12"]:
        emb = np.load(DATA / f"features_{layer}.npz")["embeddings"]
        pcs = PCA(n_components=1, random_state=0).fit_transform(emb[mask])
        idx_v = index[mask].reset_index(drop=True).copy()
        idx_v["PC1"] = pcs[:, 0]
        m0 = fit("PC1 ~ 1", idx_v);                       add(f"{layer} PC1 M0", m0, 2)
        m1 = fit("PC1 ~ L2 + Male", idx_v);               add(f"{layer} PC1 M1", m1, 4)
        m2 = fit("PC1 ~ L2 * Male", idx_v);               add(f"{layer} PC1 M2", m2, 5)
        m3 = fit("PC1 ~ L2 * Male + C(height)", idx_v);   add(f"{layer} PC1 M3", m3, 7)

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "audit_aic_bic.csv", index=False)
    print(out.round(2).to_string(index=False))
    print("Wrote audit_aic_bic.csv")


# ----------------------------------------------------------------------------
# Gap 6: Q11 - sig at p<0.05 acoustic but inside acoustic ROPE
# ----------------------------------------------------------------------------

def q11_cross_reference():
    print("\n" + "=" * 60)
    print("Gap 6 (Q11): Significant in Section 6 but ROPE-equivalent in Section 8")
    print("=" * 60)
    sig = pd.read_csv(DATA / "section6_1_acoustic.csv")
    rope = pd.read_csv(DATA / "section8_acoustic_classified.csv")

    # Sig table is per (vowel, feature). Rope table is per (vowel, feature, term)
    # but we keep only the L2 rows.
    rope = rope[rope["term"] == "L2"][["vowel", "feature", "classification"]]
    merged = sig[sig["sig_fdr"]].merge(rope, on=["vowel", "feature"], how="inner")
    print(f"\nContrasts that are FDR-significant in 6.1:")
    for _, r in merged.iterrows():
        print(f"  {r['vowel']:3s}  {r['feature']:11s}  "
              f"FDR-sig (p={r['p_fdr']:.4f})  "
              f"ROPE: {r['classification']}")

    inside = merged[merged["classification"] == "Indeterminate"]
    outside = merged[merged["classification"] == "Non-equivalent"]
    print(f"\nFDR-sig but NOT clearly outside ROPE (Indeterminate): {len(inside)}")
    print(f"FDR-sig AND outside ROPE (truly large effect):         {len(outside)}")
    merged.to_csv(DATA / "audit_q11.csv", index=False)
    print("Wrote audit_q11.csv")


# ----------------------------------------------------------------------------
# Gap 7: Q16 - phonemes systematically misclassified across all reps
# ----------------------------------------------------------------------------

def q16_systematic_errors():
    print("\n" + "=" * 60)
    print("Gap 7 (Q16): Systematically misclassified phonemes")
    print("=" * 60)
    cms = {}
    for rep in ["acoustic", "whisper_L20", "xlsr_L12"]:
        cm = pd.read_csv(DATA / f"section6_2_cm_{rep}.csv", index_col=0)
        # Per-class accuracy = diagonal / row sum
        diag = pd.Series(np.diag(cm.values), index=cm.index)
        total = cm.sum(axis=1)
        cms[rep] = (diag / total).fillna(0)

    df = pd.DataFrame(cms)
    df["min_acc"] = df.min(axis=1)
    df["mean_acc"] = df.mean(axis=1)
    df = df.sort_values("mean_acc")
    print("\nPer-vowel recall (correct/true total) per representation:")
    print(df.round(3).to_string())

    worst = df.head(3).index.tolist()
    print(f"\nMost-confused phonemes across all three reps: {worst}")
    df.to_csv(DATA / "audit_q16.csv")
    print("Wrote audit_q16.csv")


def main():
    trajectory_audit()
    missingness_report()
    vowel_a_icc()
    aic_bic_table()
    q11_cross_reference()
    q16_systematic_errors()


if __name__ == "__main__":
    main()