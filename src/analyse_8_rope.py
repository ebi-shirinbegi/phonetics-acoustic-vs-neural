"""Section 8: Wald and bootstrap CIs, ROPE classification, forest plots.

8.1 Per-vowel F1 and F2 LME with Wald CIs on the L2 fixed effect (and L2:Male).
8.2 Per-phoneme bootstrap (speaker-level) on the L1-vs-L2 centroid cosine
    distance, in Whisper L20 and XLS-R L12 space.
8.3 ROPE: acoustic = +/- 20Hz/sd_speaker (Lobanov scale);
    neural = [0, mean_intra_speaker_cosine_distance].
8.4 Equivalent / Non-equivalent / Indeterminate classification.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics.pairwise import cosine_distances

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "ə"]
from _params import load_params
_PARAMS = load_params()
N_BOOT = _PARAMS["statistics"]["bootstrap_n"]
ACOUSTIC_JND_HZ = _PARAMS["rope"]["acoustic_jnd_hz"]


def add_design(df):
    df = df.copy()
    df["L2"] = (df["L1"].str.lower() == "ru").astype(int)
    df["Male"] = (df["gender"].str.lower() == "m").astype(int)
    return df


# ---------- 8.1 acoustic CIs --------------------------------------------------

def fit_per_vowel(df, vowel, response):
    sub = df[df["phoneme"] == vowel].dropna(subset=[response])
    if len(sub) < 30 or sub["speaker"].nunique() < 4:
        return None
    res = smf.mixedlm(f"{response} ~ L2 * Male", sub, groups="speaker"
                      ).fit(reml=False, method=["lbfgs"])
    return res


def acoustic_cis(df):
    rows = []
    for vowel in ORAL_VOWELS:
        for resp in ["F1_lobanov", "F2_lobanov"]:
            res = fit_per_vowel(df, vowel, resp)
            if res is None:
                continue
            ci = res.conf_int(alpha=0.05)
            for term in ["L2", "L2:Male"]:
                if term not in res.params.index:
                    continue
                rows.append({
                    "vowel": vowel,
                    "feature": resp,
                    "term": term,
                    "estimate": float(res.params[term]),
                    "ci_lo": float(ci.loc[term, 0]),
                    "ci_hi": float(ci.loc[term, 1]),
                    "p": float(res.pvalues[term]),
                })
    return pd.DataFrame(rows)


# ---------- 8.2 neural bootstrap CIs ------------------------------------------

def neural_centroid_distance(emb, mask_l1, mask_l2):
    if mask_l1.sum() < 3 or mask_l2.sum() < 3:
        return np.nan
    c1 = emb[mask_l1].mean(axis=0, keepdims=True)
    c2 = emb[mask_l2].mean(axis=0, keepdims=True)
    return float(cosine_distances(c1, c2)[0, 0])
def neural_centroid_distance_idx(emb, idx_l1, idx_l2):
    """Cosine distance between centroids defined by token-row indices."""
    if len(idx_l1) == 0 or len(idx_l2) == 0:
        return float("nan")
    c1 = emb[idx_l1].mean(axis=0, keepdims=True)
    c2 = emb[idx_l2].mean(axis=0, keepdims=True)
    return float(cosine_distances(c1, c2)[0, 0])


def bootstrap_neural_l1l2(emb, index, layer_name, vowels=ORAL_VOWELS,
                          n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    speakers = index["speaker"].unique()
    spk_arr = index["speaker"].to_numpy()
    phon = index["phoneme"].to_numpy()
    L1 = index["L1"].to_numpy()

    rows = []
    for v in vowels:
        v_mask = phon == v
        if v_mask.sum() < 30:
            continue
        # Observed
        obs = neural_centroid_distance(emb,
                                       v_mask & (L1 == "fr"),
                                       v_mask & (L1 == "ru"))
        # Pre-index tokens by speaker for proper resample-with-replacement
        spk_to_idx = {s: np.where(spk_arr == s)[0] for s in speakers}
        boot = []
        for _ in range(n_boot):
            sample = rng.choice(speakers, size=len(speakers), replace=True)
            boot_idx = np.concatenate([spk_to_idx[s] for s in sample])
            # Build masks restricted to the resampled token indices
            in_boot = np.zeros(len(emb), dtype=bool)
            # Some indices may repeat: this still gives all the duplicate
            # rows when we select with boot_idx directly below.
            v_phon = phon[boot_idx]
            v_l1 = L1[boot_idx]
            v_in_phon = (v_phon == v)
            d = neural_centroid_distance_idx(
                emb, boot_idx[v_in_phon & (v_l1 == "fr")],
                boot_idx[v_in_phon & (v_l1 == "ru")],
            )
            if not np.isnan(d):
                boot.append(d)
        boot = np.array(boot)
        if len(boot) < 50:
            continue
        rows.append({
            "layer": layer_name,
            "phoneme": v,
            "estimate": obs,
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "boot_mean": float(boot.mean()),
        })
    return pd.DataFrame(rows)


# ---------- 8.3 ROPE definitions ----------------------------------------------

def acoustic_rope_lobanov(df):
    """Map +/- 20 Hz on raw F1 to the Lobanov scale by dividing by the
    mean per-speaker SD of F1."""
    sds = df.groupby("speaker")["F1_mid"].std(ddof=1).dropna()
    mean_sd = sds.mean()
    delta = ACOUSTIC_JND_HZ / mean_sd
    print(f"  Mean per-speaker SD of F1 (Hz): {mean_sd:.1f}")
    print(f"  Acoustic ROPE on Lobanov scale: +/- {delta:.3f}")
    return -delta, delta


def neural_rope(emb, index, layer_name):
    """Mean intra-speaker, same-phoneme, different-token cosine distance.
    This is the noise floor of the representation."""
    spk = index["speaker"].to_numpy()
    phon = index["phoneme"].to_numpy()
    distances = []
    for v in ORAL_VOWELS:
        for s in np.unique(spk):
            m = (spk == s) & (phon == v)
            if m.sum() < 2:
                continue
            sub = emb[m]
            sims = cosine_distances(sub)
            iu = np.triu_indices(len(sub), k=1)
            distances.extend(sims[iu].tolist())
    arr = np.array(distances)
    delta = float(arr.mean())
    print(f"  {layer_name} mean intra-speaker cosine distance: {delta:.4f}  "
          f"(n_pairs={len(arr)})")
    return 0.0, delta


# ---------- 8.4 ROPE classification + forest plots ----------------------------

def classify(ci_lo, ci_hi, rope_lo, rope_hi):
    """Equivalent: CI fully inside ROPE.
    Non-equivalent: CI fully outside ROPE.
    Indeterminate: CI overlaps a ROPE boundary."""
    if ci_lo >= rope_lo and ci_hi <= rope_hi:
        return "Equivalent"
    if ci_hi < rope_lo or ci_lo > rope_hi:
        return "Non-equivalent"
    return "Indeterminate"


def plot_forest_acoustic(df, rope_lo, rope_hi, out):
    df = df[df["term"] == "L2"].copy()
    if df.empty:
        return
    # Drop fits where CIs are absurdly wide (model failed to converge)
    span = df["ci_hi"] - df["ci_lo"]
    df = df[span < 5.0]
    df = df.sort_values(["feature", "vowel"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for ax, feat in zip(axes, ["F1_lobanov", "F2_lobanov"]):
        sub = df[df["feature"] == feat]
        if sub.empty:
            continue
        y = np.arange(len(sub))
        lower_err = np.maximum(0, sub["estimate"] - sub["ci_lo"])
        upper_err = np.maximum(0, sub["ci_hi"] - sub["estimate"])
        ax.errorbar(sub["estimate"], y,
                    xerr=[lower_err, upper_err],
                    fmt="o", color="black", capsize=3)
        ax.axvspan(rope_lo, rope_hi, color="grey", alpha=0.25, label="ROPE")
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["vowel"])
        ax.set_xlabel(f"L2 effect on {feat} (Lobanov z)")
        ax.set_title(feat)
        ax.set_xlim(-1.2, 1.2)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Acoustic L1/L2 contrasts — 95% Wald CIs vs ROPE")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_forest_neural(df, ropes_by_layer, out):
    if df.empty:
        return
    layers = df["layer"].unique()
    fig, axes = plt.subplots(1, len(layers), figsize=(11, 6), sharey=True)
    if len(layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, layers):
        sub = df[df["layer"] == layer].sort_values("phoneme")
        y = np.arange(len(sub))
        lower_err = np.maximum(0, sub["estimate"] - sub["ci_lo"])
        upper_err = np.maximum(0, sub["ci_hi"] - sub["estimate"])
        ax.errorbar(sub["estimate"], y,
                    xerr=[lower_err, upper_err],
                    fmt="o", color="black", capsize=3)
        rope_lo, rope_hi = ropes_by_layer[layer]
        ax.axvspan(rope_lo, rope_hi, color="grey", alpha=0.2, label="ROPE")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["phoneme"])
        ax.set_xlabel("Cosine distance L1–L2 centroids")
        ax.set_title(layer)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Neural L1/L2 contrasts — 95% bootstrap CIs vs ROPE")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------- main --------------------------------------------------------------

def main():
    print("=== 8.1 Acoustic Wald CIs (per vowel, per feature) ===")
    df_ac = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df_ac = add_design(df_ac)
    df_ac = df_ac[df_ac["phoneme"].isin(ORAL_VOWELS)]
    ac_ci = acoustic_cis(df_ac)
    ac_ci.to_csv(DATA / "section8_acoustic_ci.csv", index=False)
    print(f"Wrote section8_acoustic_ci.csv ({len(ac_ci)} rows)")

    print("\n=== 8.3 ROPE definitions ===")
    print("Acoustic:")
    rope_ac_lo, rope_ac_hi = acoustic_rope_lobanov(df_ac)

    print("\n=== 8.2 Neural bootstrap CIs ===")
    index = pd.read_csv(DATA / "features_neural_index.csv")
    ropes_by_layer = {}
    all_neural = []
    for layer in ["whisper_L20", "xlsr_L12"]:
        emb = np.load(DATA / f"features_{layer}.npz")["embeddings"]
        print(f"\n  Layer {layer}: bootstrap with B={N_BOOT}")
        nci = bootstrap_neural_l1l2(emb, index, layer)
        all_neural.append(nci)
        print(f"  rows: {len(nci)}")
        rope_lo, rope_hi = neural_rope(emb, index, layer)
        ropes_by_layer[layer] = (rope_lo, rope_hi)
    neural_ci = pd.concat(all_neural, ignore_index=True)
    neural_ci.to_csv(DATA / "section8_neural_ci.csv", index=False)
    print(f"\nWrote section8_neural_ci.csv ({len(neural_ci)} rows)")

    print("\n=== 8.4 ROPE classification ===")
    ac_l2 = ac_ci[ac_ci["term"] == "L2"].copy()
    ac_l2["classification"] = ac_l2.apply(
        lambda r: classify(r["ci_lo"], r["ci_hi"], rope_ac_lo, rope_ac_hi), axis=1)
    print("\nAcoustic L2 contrasts:")
    print(ac_l2[["vowel", "feature", "estimate", "ci_lo", "ci_hi",
                 "classification"]].round(3).to_string(index=False))
    ac_l2.to_csv(DATA / "section8_acoustic_classified.csv", index=False)

    rows = []
    for _, r in neural_ci.iterrows():
        lo, hi = ropes_by_layer[r["layer"]]
        rows.append({**r.to_dict(),
                     "rope_lo": lo, "rope_hi": hi,
                     "classification": classify(r["ci_lo"], r["ci_hi"], lo, hi)})
    neural_classified = pd.DataFrame(rows)
    neural_classified.to_csv(DATA / "section8_neural_classified.csv", index=False)
    print("\nNeural L1/L2 contrasts:")
    print(neural_classified[["layer", "phoneme", "estimate", "ci_lo", "ci_hi",
                             "classification"]].round(4).to_string(index=False))

    print("\n=== Forest plots ===")
    plot_forest_acoustic(ac_ci, rope_ac_lo, rope_ac_hi,
                         FIG / "fig_8_forest_acoustic.png")
    plot_forest_neural(neural_ci, ropes_by_layer,
                       FIG / "fig_8_forest_neural.png")
    print("Saved fig_8_forest_acoustic.png and fig_8_forest_neural.png")


if __name__ == "__main__":
    main()