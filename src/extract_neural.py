"""Stage 3: Neural representations per phoneme token (MPS-accelerated).

For each WAV I run Whisper-medium and XLS-R once on the M4 GPU (MPS),
then for each phoneme token I slice the hidden states along the time
axis to the [onset, offset] interval and mean-pool.

Layers I keep:
  - Whisper-medium: 6 (lower half), 20 (upper half)
  - XLS-R:           3 (lower), 12 (middle), 20 (upper)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from transformers import (
    WhisperFeatureExtractor, WhisperModel,
    AutoFeatureExtractor, Wav2Vec2Model,
)

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "features_phonemes_clean.csv"
OUT_DIR = ROOT / "data"

WHISPER_NAME = "openai/whisper-medium"
XLSR_NAME = "facebook/wav2vec2-large-xlsr-53"

WHISPER_LAYERS = [6, 20]
XLSR_LAYERS = [3, 12, 20]
TARGET_SR = 16000

WHISPER_FRAME_RATE = 50.0
XLSR_FRAME_RATE = 49.95

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def slice_pool_batch(hidden, onsets, offsets, frame_rate):
    """hidden: (T, D) tensor on device. Returns (N, D) numpy array of
    mean-pooled vectors, one per (onset, offset) pair."""
    T, D = hidden.shape
    out = np.zeros((len(onsets), D), dtype=np.float32)
    for i, (on, off) in enumerate(zip(onsets, offsets)):
        t0 = int(on * frame_rate)
        t1 = max(t0 + 1, int(off * frame_rate))
        t1 = min(t1, T)
        t0 = min(t0, t1 - 1)
        out[i] = hidden[t0:t1].mean(dim=0).float().cpu().numpy()
    return out


def main():
    df = pd.read_csv(IN_PATH)
    n = len(df)
    print(f"Loaded {n} phoneme tokens.")
    print(f"Device: {DEVICE}  dtype: {DTYPE}")

    print(f"Loading Whisper: {WHISPER_NAME}")
    whisper_fe = WhisperFeatureExtractor.from_pretrained(WHISPER_NAME)
    whisper = WhisperModel.from_pretrained(WHISPER_NAME, torch_dtype=DTYPE).to(DEVICE)
    whisper.eval()
    whisper_dim = whisper.config.d_model

    print(f"Loading XLS-R: {XLSR_NAME}")
    xlsr_fe = AutoFeatureExtractor.from_pretrained(XLSR_NAME)
    xlsr = Wav2Vec2Model.from_pretrained(XLSR_NAME, torch_dtype=DTYPE).to(DEVICE)
    xlsr.eval()
    xlsr_dim = xlsr.config.hidden_size

    out_whisper = {l: np.zeros((n, whisper_dim), dtype=np.float32) for l in WHISPER_LAYERS}
    out_xlsr    = {l: np.zeros((n, xlsr_dim),    dtype=np.float32) for l in XLSR_LAYERS}

    # Pre-extract numpy arrays I need inside the inner loop (no iterrows)
    df["row_idx"] = np.arange(n)
    onsets_all  = df["onset"].to_numpy()
    offsets_all = df["offset"].to_numpy()
    row_idx_all = df["row_idx"].to_numpy()
    wav_all     = df["wav_path"].to_numpy()

    distinct_wavs = pd.unique(wav_all)
    print(f"Distinct WAV files: {len(distinct_wavs)}")

    t0 = time.time()
    with torch.no_grad():
        for wav_rel in tqdm(distinct_wavs, desc="Per-WAV pass"):
            mask = wav_all == wav_rel
            on  = onsets_all[mask]
            off = offsets_all[mask]
            idx = row_idx_all[mask]

            audio = load_audio(ROOT / wav_rel)

            # ---- Whisper ----
            inputs = whisper_fe(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            feats = inputs.input_features.to(DEVICE, dtype=DTYPE)
            enc_out = whisper.encoder(feats, output_hidden_states=True)
            for layer in WHISPER_LAYERS:
                hidden = enc_out.hidden_states[layer].squeeze(0)
                pooled = slice_pool_batch(hidden, on, off, WHISPER_FRAME_RATE)
                out_whisper[layer][idx] = pooled

            # ---- XLS-R ----
            inputs = xlsr_fe(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(DEVICE, dtype=DTYPE)
            enc_out = xlsr(input_values, output_hidden_states=True)
            for layer in XLSR_LAYERS:
                hidden = enc_out.hidden_states[layer].squeeze(0)
                pooled = slice_pool_batch(hidden, on, off, XLSR_FRAME_RATE)
                out_xlsr[layer][idx] = pooled

    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed/60:.1f} min")

    for layer in WHISPER_LAYERS:
        path = OUT_DIR / f"features_whisper_L{layer:02d}.npz"
        np.savez(path, embeddings=out_whisper[layer])
        print(f"Saved {path.name}  shape={out_whisper[layer].shape}")
    for layer in XLSR_LAYERS:
        path = OUT_DIR / f"features_xlsr_L{layer:02d}.npz"
        np.savez(path, embeddings=out_xlsr[layer])
        print(f"Saved {path.name}  shape={out_xlsr[layer].shape}")

    index_path = OUT_DIR / "features_neural_index.csv"
    df.drop(columns=["row_idx"]).to_csv(index_path, index=False)
    print(f"Saved {index_path.name}")


if __name__ == "__main__":
    main()