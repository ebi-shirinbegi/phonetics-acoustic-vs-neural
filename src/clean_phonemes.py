"""Stage 1b: Clean the phoneme manifest.

I drop annotation artifacts ('ding...' labels, ~21 tokens) and fold
diacritic-marked variants (devoicing, creaky voice, length) into their
base phoneme. This keeps the phonemic categories defined by the lab
intact and avoids splitting tokens across cosmetic variants.
"""
import unicodedata
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "features_phonemes.csv"
OUT_PATH = ROOT / "data" / "features_phonemes_clean.csv"

# Diacritics I strip (combining characters, length marks, palatalisation, aspiration)
STRIP_CODEPOINTS = {
    0x0303,  # combining tilde — but I keep nasal vowels via a special rule below
    0x0325,  # combining ring below (devoicing)
    0x0330,  # combining tilde below (creaky)
    0x031D,  # combining up tack (raised)
    0x031E,  # combining down tack (lowered)
    0x02D0,  # length mark ː
    0x02B2,  # palatalisation ʲ
    0x02B0,  # aspiration ʰ
    0x003A,  # plain colon used as length in a few cases
}

# Phonemes where I want to KEEP the tilde because it marks a distinct
# phonemic nasal vowel in French
KEEP_TILDE = {"ɑ̃", "ɛ̃", "œ̃", "ɔ̃"}


def fold(label):
    if label in KEEP_TILDE:
        return label
    return "".join(ch for ch in label if ord(ch) not in STRIP_CODEPOINTS)


def main():
    df = pd.read_csv(IN_PATH)
    n0 = len(df)

    # Drop ding... artifacts
    is_ding = df["phoneme"].str.startswith("ding")
    df = df[~is_ding].copy()
    print(f"Dropped {is_ding.sum()} 'ding...' tokens.")

    # Fold diacritics
    df["phoneme_raw"] = df["phoneme"]
    df["phoneme"] = df["phoneme"].map(fold)

    # Drop empty after folding (defensive)
    empty = df["phoneme"] == ""
    if empty.any():
        df = df[~empty].copy()
        print(f"Dropped {empty.sum()} tokens that became empty after folding.")

    # Reorder columns for downstream stages
    cols = ["speaker", "L1", "gender", "sentence_id", "repetition",
            "phoneme", "phoneme_raw", "onset", "offset",
            "duration_ms", "wav_path"]
    df = df[cols]

    df.to_csv(OUT_PATH, index=False)

    print()
    print("=" * 60)
    print(f"Clean manifest: {OUT_PATH.relative_to(ROOT)}")
    print(f"Tokens: {n0} -> {len(df)}")
    print(f"Distinct phonemes after folding: {df['phoneme'].nunique()}")
    print()
    print("Top 25 phonemes (clean):")
    for p, c in df["phoneme"].value_counts().head(25).items():
        codepoints = " ".join(f"U+{ord(ch):04X}" for ch in p)
        print(f"  {repr(p):8s}  n={c:4d}  [{codepoints}]")


if __name__ == "__main__":
    main()