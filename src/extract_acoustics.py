"""Stage 2: Acoustic features per phoneme token using Praat (parselmouth).

I extract for each token:
  - F1, F2 at the midpoint (always)
  - F3 at the midpoint (always; will mostly be useful for vowels)
  - F1, F2 at 25% and 75% (only if duration > 80 ms, for trajectory)
  - mean f0 over voiced frames
  - spectral centre of gravity (useful for fricatives)

LPC settings follow the project spec: max_formant 5000 Hz (female) /
4500 Hz (male), n_formants = 5. Failed measurements are stored as NaN.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "features_phonemes_clean.csv"
OUT_PATH = ROOT / "data" / "features_acoustic.csv"

MAX_FORMANT_F = 5000.0
MAX_FORMANT_M = 4500.0
N_FORMANTS = 5
TRAJECTORY_THRESHOLD_S = 0.080


def safe_call(func, *args, **kwargs):
    """Run a parselmouth/praat call and return NaN on failure."""
    try:
        v = func(*args, **kwargs)
        if v is None:
            return np.nan
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return np.nan
        return v
    except Exception:
        return np.nan


def measure_token(sound_segment, gender):
    """Return a dict with all acoustic measurements for one phoneme segment."""
    max_formant = MAX_FORMANT_F if gender == "f" else MAX_FORMANT_M
    duration = sound_segment.get_total_duration()
    midpoint = duration / 2.0

    out = {
        "F1_mid": np.nan, "F2_mid": np.nan, "F3_mid": np.nan,
        "F1_25": np.nan, "F2_25": np.nan,
        "F1_75": np.nan, "F2_75": np.nan,
        "f0_mean": np.nan, "scg_mean": np.nan,
    }

    # Formants via Burg LPC
    try:
        formants = sound_segment.to_formant_burg(max_number_of_formants=N_FORMANTS,
                                                 maximum_formant=max_formant)
        out["F1_mid"] = safe_call(formants.get_value_at_time, 1, midpoint)
        out["F2_mid"] = safe_call(formants.get_value_at_time, 2, midpoint)
        out["F3_mid"] = safe_call(formants.get_value_at_time, 3, midpoint)

        if duration > TRAJECTORY_THRESHOLD_S:
            t25 = duration * 0.25
            t75 = duration * 0.75
            out["F1_25"] = safe_call(formants.get_value_at_time, 1, t25)
            out["F2_25"] = safe_call(formants.get_value_at_time, 2, t25)
            out["F1_75"] = safe_call(formants.get_value_at_time, 1, t75)
            out["F2_75"] = safe_call(formants.get_value_at_time, 2, t75)
    except Exception:
        pass

    # f0 via autocorrelation, only voiced frames
    try:
        pitch = sound_segment.to_pitch_ac()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if len(f0) > 0:
            out["f0_mean"] = float(np.mean(f0))
    except Exception:
        pass

    # Spectral centre of gravity
    try:
        spectrum = sound_segment.to_spectrum()
        out["scg_mean"] = safe_call(spectrum.get_centre_of_gravity)
    except Exception:
        pass

    return out


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} tokens.")

    # I cache the loaded WAV per file so I do not re-read it for every phoneme
    cache = {}
    rows = []

    for r in tqdm(df.to_dict("records"), desc="Extracting"):
        wav_path = ROOT / r["wav_path"]
        if wav_path not in cache:
            cache[wav_path] = parselmouth.Sound(str(wav_path))
        full_sound = cache[wav_path]

        try:
            seg = full_sound.extract_part(from_time=r["onset"],
                                          to_time=r["offset"],
                                          preserve_times=False)
        except Exception:
            seg = None

        if seg is None or seg.get_total_duration() <= 0:
            measurements = {k: np.nan for k in
                            ["F1_mid", "F2_mid", "F3_mid",
                             "F1_25", "F2_25", "F1_75", "F2_75",
                             "f0_mean", "scg_mean"]}
        else:
            measurements = measure_token(seg, r["gender"])

        rows.append({**r, **measurements})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")

    # Quick missingness summary
    print("\n=== Missing rate per measurement ===")
    for col in ["F1_mid", "F2_mid", "F3_mid", "f0_mean", "scg_mean"]:
        miss = out[col].isna().mean() * 100
        print(f"  {col:10s}  {miss:5.1f}%")

    print("\n=== Missing F1_mid by phoneme (top 10) ===")
    miss_by_phon = (out.groupby("phoneme")["F1_mid"]
                       .apply(lambda s: s.isna().mean() * 100)
                       .sort_values(ascending=False)
                       .head(10))
    print(miss_by_phon.round(1).to_string())


if __name__ == "__main__":
    main()