"""Stage 1: Build a phoneme-level manifest from TextGrid alignments.

Reads:
  - corpus_raw/.../RUFRcorr.csv      (file_number -> target word)
  - corpus_raw/.../metadata_RUFR.csv (speaker -> L1, gender)
  - corpus_raw/.../*/*.TextGrid      (word + phones tiers)

Writes:
  - data/features_phonemes.csv with columns:
      speaker, L1, gender, sentence_id, repetition,
      phoneme, onset, offset, duration_ms, wav_path
"""
import csv
from collections import Counter
from pathlib import Path

import parselmouth

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus_raw" / "ru-fr_interference" / "2"
SPEAKERS_DIR = CORPUS / "wav_et_textgrids" / "FRcorp_textgrids_only"
OUT_PATH = ROOT / "data" / "features_phonemes.csv"


def load_word_to_files(path):
    """RUFRcorr.csv -> file_num -> (word, occurrence_index in 1..6)."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            word = row[0].strip()
            for i, occ in enumerate(row[2:8], start=1):
                mapping[int(occ.strip())] = (word, i)
    return mapping


def load_speaker_metadata(path):
    """metadata_RUFR.csv -> speaker -> (L1, gender)."""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            spk = row[1].strip()
            l1 = row[2].strip()
            gender = row[4].strip()
            mapping[spk] = (l1, gender)
    return mapping


def extract_target_phonemes(tg_path, target_word):
    """Open the TextGrid, locate the target word in tier 'words' (between
    'dis' and 'trois'), then return all phoneme intervals from tier 'phones'
    that fall inside the target word's time span.

    Returns a list of (phoneme, onset, offset).
    """
    tg = parselmouth.Data.read(str(tg_path))
    # Tier indices in Praat/parselmouth are 1-based
    n_tiers = parselmouth.praat.call(tg, "Get number of tiers")
    word_tier_idx = phone_tier_idx = None
    for i in range(1, n_tiers + 1):
        name = parselmouth.praat.call(tg, "Get tier name", i)
        if name == "words":
            word_tier_idx = i
        elif name == "phones":
            phone_tier_idx = i
    if word_tier_idx is None or phone_tier_idx is None:
        return []

    # Find 'dis' and 'trois' boundaries on the word tier
    n_words = parselmouth.praat.call(tg, "Get number of intervals", word_tier_idx)
    dis_end = None
    trois_start = None
    for i in range(1, n_words + 1):
        label = parselmouth.praat.call(tg, "Get label of interval", word_tier_idx, i).strip().lower()
        if label == "dis" and dis_end is None:
            dis_end = parselmouth.praat.call(tg, "Get end point", word_tier_idx, i)
        elif label == "trois" and dis_end is not None:
            trois_start = parselmouth.praat.call(tg, "Get starting point", word_tier_idx, i)
            break
    if dis_end is None or trois_start is None:
        return []

    # Collect all non-empty phoneme intervals strictly between dis_end and trois_start
    n_phones = parselmouth.praat.call(tg, "Get number of intervals", phone_tier_idx)
    out = []
    for i in range(1, n_phones + 1):
        label = parselmouth.praat.call(tg, "Get label of interval", phone_tier_idx, i).strip()
        if not label:
            continue
        on = parselmouth.praat.call(tg, "Get starting point", phone_tier_idx, i)
        off = parselmouth.praat.call(tg, "Get end point", phone_tier_idx, i)
        # Check the phoneme is inside the target-word span (with tiny tolerance)
        if on >= dis_end - 1e-3 and off <= trois_start + 1e-3:
            out.append((label, on, off))
    return out


def main():
    word_to_files = load_word_to_files(CORPUS / "RUFRcorr.csv")
    speaker_meta = load_speaker_metadata(CORPUS / "metadata_RUFR.csv")

    rows_out = []
    skipped_files = 0
    skipped_phonemes = 0

    speaker_dirs = sorted(d for d in SPEAKERS_DIR.iterdir() if d.is_dir())
    print(f"Processing {len(speaker_dirs)} speakers...")

    for spk_dir in speaker_dirs:
        spk = spk_dir.name
        if spk not in speaker_meta:
            print(f"  WARN: speaker {spk} not in metadata, skipping")
            continue
        l1, gender = speaker_meta[spk]

        for tg_path in sorted(spk_dir.glob("*.TextGrid")):
            base = tg_path.stem
            try:
                file_num = int(base.split("FRcorp")[-1])
            except ValueError:
                skipped_files += 1
                continue
            if file_num not in word_to_files:
                continue  # filler / warmup file, not a target

            target_word, repetition = word_to_files[file_num]
            wav_path = tg_path.with_suffix(".wav")
            if not wav_path.exists():
                skipped_files += 1
                continue

            phonemes = extract_target_phonemes(tg_path, target_word)
            if not phonemes:
                skipped_phonemes += 1
                continue

            for phon, on, off in phonemes:
                rows_out.append({
                    "speaker": spk,
                    "L1": l1,
                    "gender": gender,
                    "sentence_id": target_word,
                    "repetition": repetition,
                    "phoneme": phon,
                    "onset": round(on, 6),
                    "offset": round(off, 6),
                    "duration_ms": round((off - on) * 1000, 3),
                    "wav_path": str(wav_path.relative_to(ROOT)),
                })

        print(f"  {spk} ({l1}/{gender}) done")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    # Summary
    print()
    print("=" * 60)
    print(f"Manifest written to {OUT_PATH.relative_to(ROOT)}")
    print(f"Total phoneme tokens : {len(rows_out)}")
    print(f"Speakers             : {len(set(r['speaker'] for r in rows_out))}")
    print(f"Distinct phonemes    : {len(set(r['phoneme'] for r in rows_out))}")
    print(f"Words covered        : {len(set(r['sentence_id'] for r in rows_out))}")
    print(f"Skipped files        : {skipped_files}")
    print(f"Skipped (no phonemes): {skipped_phonemes}")

    print("\nSpeaker breakdown (L1 × Gender):")
    counts = Counter((r["L1"], r["gender"]) for r in rows_out
                     if r["repetition"] == 1 and r["sentence_id"] == "tsarine")
    for (l1, g), c in sorted(counts.items()):
        print(f"  L1={l1!s:3s}  gender={g!s:2s}  -> {c} speakers")

    print("\nTop 15 phonemes by frequency:")
    for phon, c in Counter(r["phoneme"] for r in rows_out).most_common(15):
        print(f"  {phon!r:8s}  {c}")


if __name__ == "__main__":
    main()