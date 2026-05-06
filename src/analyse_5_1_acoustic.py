"""Section 5.1: Descriptive statistics on acoustic features."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ORAL_VOWELS = ["i", "e", "ɛ", "a", "ɑ", "o", "u", "y", "ø", "œ", "ə"]
NASALS = ["ɑ̃", "ɛ̃", "œ̃", "ɔ̃"]
ALL_VOWELS = ORAL_VOWELS + NASALS


def add_group_column(df):
    df = df.copy()
    df["group"] = df["L1"].str.upper() + "/" + df["gender"].str.upper()
    return df


def summary_table(df):
    rows = []
    for vowel in ORAL_VOWELS:
        sub = df[df["phoneme"] == vowel]
        if len(sub) < 5:
            continue
        for group, gsub in sub.groupby("group"):
            for col in ["F1_lobanov", "F2_lobanov"]:
                vals = gsub[col].dropna()
                if len(vals) < 3:
                    continue
                q25, q75 = vals.quantile([0.25, 0.75])
                rows.append({
                    "vowel": vowel,
                    "group": group,
                    "feature": col,
                    "n": len(vals),
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "sd": vals.std(ddof=1),
                    "iqr": q75 - q25,
                    "cv": vals.std(ddof=1) / vals.mean() if vals.mean() != 0 else np.nan,
                })
    return pd.DataFrame(rows)


def variance_decomposition(df):
    rows = []
    for vowel in ORAL_VOWELS:
        sub = df[(df["phoneme"] == vowel) & df["F1_mid"].notna()].copy()
        if len(sub) < 30:
            continue
        spk_mean = sub.groupby("speaker")["F1_mid"].mean()
        grand = sub["F1_mid"].mean()
        n_per_spk = sub.groupby("speaker").size()
        inter = np.average((spk_mean - grand) ** 2, weights=n_per_spk)
        intra = sub.groupby("speaker")["F1_mid"].var(ddof=1).mean()
        total = sub["F1_mid"].var(ddof=1)
        residual = max(total - inter - intra, 0)
        rows.append({
            "vowel": vowel,
            "n": len(sub),
            "var_total": total,
            "var_inter": inter,
            "var_intra": intra,
            "var_residual": residual,
            "pct_inter": inter / total * 100,
            "pct_intra": intra / total * 100,
        })
    return pd.DataFrame(rows)


def plot_vowel_chart(df, out_path):
    """One panel per group, vowel centroids with 95% ellipses,
    arranged like the IPA vowel trapezoid (F2 reversed, F1 reversed)."""
    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    palette = sns.color_palette("husl", n_colors=len(ORAL_VOWELS))
    color_map = dict(zip(ORAL_VOWELS, palette))

    for ax, group in zip(axes.flat, groups):
        gsub_all = df[df["group"] == group]
        for vowel in ORAL_VOWELS:
            sub = gsub_all[gsub_all["phoneme"] == vowel].dropna(
                subset=["F1_lobanov", "F2_lobanov"])
            if len(sub) < 5:
                continue
            mu_F1, mu_F2 = sub["F1_lobanov"].mean(), sub["F2_lobanov"].mean()
            ax.scatter(mu_F2, mu_F1, color=color_map[vowel], s=80,
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.text(mu_F2, mu_F1 - 0.12, vowel, fontsize=12, ha="center",
                    fontweight="bold", zorder=4)
            cov = np.cov(sub[["F2_lobanov", "F1_lobanov"]].T)
            if not np.isfinite(cov).all():
                continue
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            chi2_95 = 5.991
            w, h = 2 * np.sqrt(vals * chi2_95)
            e = Ellipse(xy=(mu_F2, mu_F1), width=w, height=h, angle=angle,
                        edgecolor=color_map[vowel], facecolor="none",
                        linewidth=1.0, alpha=0.7)
            ax.add_patch(e)
        ax.set_title(group)
        sns.despine(ax=ax)

    for ax in axes[1, :]:
        ax.set_xlabel("F2 (Lobanov z-score)")
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 (Lobanov z-score)")
    axes[0, 0].invert_xaxis()
    axes[0, 0].invert_yaxis()
    fig.suptitle("Vowel chart per speaker group (95% ellipses, IPA orientation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplots(df, out_path):
    sub = df[df["phoneme"].isin(ORAL_VOWELS)].copy()
    sub = sub.dropna(subset=["F1_lobanov", "F2_lobanov"])
    sub["phoneme"] = pd.Categorical(sub["phoneme"], categories=ORAL_VOWELS, ordered=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, col, label in zip(axes, ["F1_lobanov", "F2_lobanov"],
                              ["F1 (Lobanov z-score)", "F2 (Lobanov z-score)"]):
        sns.boxplot(data=sub, x="phoneme", y=col, hue="group", ax=ax,
                    palette="Set2", showfliers=False)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.legend(title="Group", loc="upper right", ncol=4)
        sns.despine(ax=ax)
    axes[1].set_xlabel("Vowel")
    fig.suptitle("F1 and F2 distributions per vowel × group", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intraspeaker(df, out_path, vowels=("a", "i", "u", "e", "ɛ")):
    """Strip plots of intra-speaker F1 across repetitions, clipped to ±3 SD."""
    sub = df[df["phoneme"].isin(vowels)].copy()
    sub = sub.dropna(subset=["F1_lobanov"])
    sub = sub[sub["F1_lobanov"].abs() <= 3]
    sub["phoneme"] = pd.Categorical(sub["phoneme"], categories=vowels, ordered=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.stripplot(data=sub, x="speaker", y="F1_lobanov", hue="phoneme",
                  palette="Set2", size=4, alpha=0.7, dodge=True,
                  jitter=0.15, ax=ax)
    ax.set_title("Intra-speaker F1 across repetitions (clipped to ±3 SD)")
    ax.set_xlabel("Speaker")
    ax.set_ylabel("F1 (Lobanov z-score)")
    ax.legend(title="Vowel", loc="upper right")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)
    sns.despine(ax=ax)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA / "features_acoustic_norm.csv")
    df = add_group_column(df)
    print(f"Loaded {len(df)} tokens. Groups: {sorted(df['group'].unique())}")

    print("\n=== Summary table ===")
    summary = summary_table(df)
    summary.to_csv(DATA / "section5_summary.csv", index=False)
    print(f"Wrote section5_summary.csv  ({len(summary)} rows)")

    print("\n=== Variance decomposition (F1) ===")
    var_table = variance_decomposition(df)
    var_table.to_csv(DATA / "section5_variance.csv", index=False)
    print(var_table[["vowel", "n", "pct_inter", "pct_intra"]].round(1).to_string(index=False))

    print("\n=== Figures ===")
    plot_vowel_chart(df, FIG / "fig_5_1_vowel_chart.png")
    print("  fig_5_1_vowel_chart.png")
    plot_boxplots(df, FIG / "fig_5_1_boxplots.png")
    print("  fig_5_1_boxplots.png")
    plot_intraspeaker(df, FIG / "fig_5_1_intraspeaker.png")
    print("  fig_5_1_intraspeaker.png")


if __name__ == "__main__":
    main()
