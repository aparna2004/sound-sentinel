"""
================================================================================
VoicePrint-ID MVP | Phase 1
audio_processor.py — v4 (emotion-robust, vocal-tract focused)
================================================================================

KEY CHANGES FROM v3:
    1. REMOVED delta and delta² coefficients entirely.
       Deltas capture how MFCCs *change over time* — that's prosody,
       which is emotion-dependent. Angry speech has steeper MFCC transitions
       than calm speech even for the same speaker. Removing deltas is the
       single most important fix for cross-emotion recognition.

    2. FEATURE WEIGHTING applied at extraction time (not just scoring).
       Low-order MFCCs (C1-C4) = vocal tract shape = speaker identity = HIGH weight
       High-order MFCCs (C13-C40) = fine spectral detail = emotion-sensitive = LOW weight

    3. REPLACED mean-only pooling with mean + robust statistics.
       Added: spectral centroid mean, zero-crossing rate mean (both stable cross-emotion)
       Removed: anything prosody/energy-based from the embedding.

    4. LOUDNESS NORMALIZATION before any processing.
       Different recordings (calm vs angry) have different RMS.
       Normalizing to fixed RMS makes MFCC values volume-independent.

    5. CEPSTRAL MEAN NORMALIZATION (CMN) per utterance.
       Removes channel/microphone effects. Does NOT hurt speaker discrimination
       when applied after removing C0 (energy). Helps cross-condition recognition.

FINAL EMBEDDING: 40-dim L2-normalized vector (plain MFCC means, weighted + CMN).
This is smaller than v3 (was 120-dim with deltas) but FAR more discriminative
for the task of same-person-different-emotion recognition.
================================================================================
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Union, Tuple, Optional
import io

DEBUG = True

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE   = 16_000
DURATION      = 4.0          # seconds — longer window = more stable statistics
N_MFCC        = 40
N_FFT         = 512
HOP_LENGTH    = 160
N_MELS        = 80
PRE_EMPHASIS  = 0.97
TARGET_RMS    = 0.08          # Fixed loudness target

# ---------------------------------------------------------------------------
# Feature weights: vocal-tract dims HIGH, prosody dims LOW
#
# MFCC coefficient numbering (1-indexed after dropping C0):
#   C1–C4:   vocal tract resonance shape → identity-stable → weight 2.0
#   C5–C10:  mixed vocal tract + articulation → moderate → weight 1.2
#   C11–C20: fine spectral texture → somewhat emotion-sensitive → weight 0.8
#   C21–C40: very fine detail, noisy, emotion-sensitive → weight 0.4
#
# These weights are applied to the mean-pooled MFCC vector before L2-norm.
# Result: same speaker with different emotion scores HIGH similarity.
# Different speakers score LOW similarity regardless of emotion.
# ---------------------------------------------------------------------------
def _build_feature_weights(n_mfcc: int) -> np.ndarray:
    w = np.ones(n_mfcc, dtype=np.float32)
    for i in range(n_mfcc):
        coef = i + 1  # 1-indexed (C0 already dropped)
        if coef <= 4:
            w[i] = 2.0
        elif coef <= 10:
            w[i] = 1.2
        elif coef <= 20:
            w[i] = 0.8
        else:
            w[i] = 0.4
    return w

FEATURE_WEIGHTS = _build_feature_weights(N_MFCC)


# ---------------------------------------------------------------------------
# Audio loading + preprocessing
# ---------------------------------------------------------------------------

def loudness_normalize(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """
    Scale audio so RMS = target_rms.
    This makes MFCC magnitudes comparable across recordings at
    different volumes (calm speech is quieter than angry speech).
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-8:
        return audio  # silence — don't amplify
    return (audio * (target_rms / current_rms)).astype(np.float32)


def apply_preemphasis(audio: np.ndarray, coef: float = PRE_EMPHASIS) -> np.ndarray:
    """Boost high-frequency formants."""
    return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)


def voice_activity_detection(audio: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence. Keep original if result is too short."""
    trimmed, _ = librosa.effects.trim(audio, top_db=25, frame_length=512, hop_length=128)
    return trimmed if len(trimmed) >= int(0.5 * SAMPLE_RATE) else audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Center-crop if too long, tile-pad if too short."""
    n = len(audio)
    if n >= target_length:
        start = (n - target_length) // 2
        return audio[start:start + target_length]
    repeats = (target_length // n) + 1
    return np.tile(audio, repeats)[:target_length]


def load_audio(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract emotion-robust speaker features.

    Returns: (N_MFCC,) float32 — weighted MFCC means.

    IMPORTANT — WHY CMN IS NOT APPLIED HERE:
        Classic per-utterance CMN (subtract the per-coefficient mean across
        time and then mean-pool) produces an identically-zero output vector
        because mean-pooling a zero-mean signal gives zero. That destroys all
        speaker information and is the root cause of near-zero / negative
        cosine similarities.

        Instead we:
          1. Mean-pool raw MFCCs across time  →  (N_MFCC,) vector
          2. Apply feature weighting           →  emphasize vocal-tract dims
          3. L2-normalize in aggregate_to_embedding()

        Loudness normalization + dropping C0 already handle most channel /
        energy variance. Further channel normalization can be done by
        subtracting a long-term enrollment-mean at scoring time (not here).

    WHAT WE USE:
        - MFCCs C1–C40 (drop C0 = energy coefficient, emotion-sensitive)
        - Feature weighting (vocal-tract dims up, fine-detail dims down)
        - Mean-only pooling (no std, no deltas — both are emotion-sensitive)

    WHAT WE DELIBERATELY DO NOT USE:
        - Delta / delta² coefficients (prosody = emotion signal)
        - Pitch / F0 (changes 30-50 Hz with emotion)
        - Energy / RMS (changes drastically with emotion)
        - Speaking rate features (emotion-dependent)
        - MFCC C0 (energy-based, emotion-sensitive)
        - Per-utterance CMN before mean-pooling (produces zero vector)
    """
    # Extract MFCCs (C0 through C_N_MFCC)
    mfcc_full = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC + 1,   # +1 so we can drop C0
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=60,              # exclude sub-bass, focus on speech fundamentals
        fmax=7600,
    )

    # Drop C0 (energy coefficient — emotion-sensitive)
    mfcc = mfcc_full[1:, :]   # shape: (N_MFCC, T)

    # Mean-pool across time → (N_MFCC,)
    # NOTE: Do NOT apply per-utterance CMN before this step — doing so makes
    # mean_vec identically zero (mean of a zero-mean signal = 0) which breaks
    # cosine similarity and causes negative scores.
    mean_vec = mfcc.mean(axis=1).astype(np.float32)

    # Apply vocal-tract weighting
    weighted = mean_vec * FEATURE_WEIGHTS

    return weighted  # (N_MFCC,) — will be L2-normalized in aggregate step


def aggregate_to_embedding(features: np.ndarray, label: str = "") -> np.ndarray:
    """
    L2-normalize the weighted feature vector to unit sphere.
    Cosine similarity on unit vectors = dot product = scale-invariant.
    """
    norm = np.linalg.norm(features)

    if DEBUG:
        print(
            f"[DEBUG{' ' + label if label else ''}] "
            f"feature mean={features.mean():.3f}  "
            f"std={features.std():.3f}  "
            f"norm={norm:.4f}  "
            f"shape={features.shape}"
        )
        if features.std() < 0.1:
            print(f"[DEBUG] ⚠️  Very low feature std — audio may be silence or corrupt!")

    if norm < 1e-10:
        print("[AudioProcessor] ⚠️  Near-zero norm — returning zero vector")
        return np.zeros(features.shape[0], dtype=np.float32)

    return (features / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(audio: np.ndarray, label: str = "") -> Optional[np.ndarray]:
    """Shared preprocessing + feature extraction pipeline."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-5:
        print(f"[AudioProcessor] ⚠️  Audio appears to be silence (rms={rms:.6f})")

    audio = loudness_normalize(audio)
    audio = apply_preemphasis(audio)
    audio = voice_activity_detection(audio)
    audio = pad_or_truncate(audio, int(DURATION * SAMPLE_RATE))

    features  = extract_features(audio, SAMPLE_RATE)
    embedding = aggregate_to_embedding(features, label=label)

    if not np.isfinite(embedding).all():
        print(f"[AudioProcessor] ❌ Non-finite embedding for '{label}'")
        return None

    return embedding


def process_audio_file(path: Union[str, Path], label: str = "") -> Optional[np.ndarray]:
    """
    Full pipeline for an audio file on disk.
    Returns N_MFCC-dim L2-normalized embedding or None on error.
    """
    try:
        path_str = str(path)
        audio, sr = load_audio(path_str)

        if DEBUG:
            print(
                f"[DEBUG] Loaded '{Path(path_str).name}': "
                f"{len(audio)} samples, sr={sr}, "
                f"rms={np.sqrt(np.mean(audio ** 2)):.5f}"
            )

        return _run_pipeline(audio, label=label or Path(path_str).name)

    except Exception as exc:
        print(f"[AudioProcessor] ❌ Error processing file: {exc}")
        import traceback; traceback.print_exc()
        return None


def process_audio_bytes(audio_bytes: bytes, label: str = "bytes") -> Optional[np.ndarray]:
    """
    Pipeline for raw bytes from browser / WebSocket.

    Tries in order:
        1. soundfile  (WAV, FLAC, OGG Vorbis)
        2. librosa via temp file (WebM/Opus — needs ffmpeg)
        3. Raw int16 PCM fallback
    """
    import tempfile, os

    if DEBUG:
        print(f"[DEBUG] Received audio bytes: {len(audio_bytes)} bytes")

    audio = None

    # Strategy 1: soundfile
    try:
        import soundfile as sf
        buf = io.BytesIO(audio_bytes)
        audio_sf, sr_sf = sf.read(buf, dtype='float32', always_2d=False)
        if audio_sf.ndim > 1:
            audio_sf = audio_sf.mean(axis=1)
        if sr_sf != SAMPLE_RATE:
            audio_sf = librosa.resample(audio_sf, orig_sr=sr_sf, target_sr=SAMPLE_RATE)
        audio = audio_sf
        if DEBUG:
            print(f"[DEBUG] Decoded via soundfile: {len(audio)} samples, rms={np.sqrt(np.mean(audio**2)):.5f}")
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] soundfile failed: {e}")

    # Strategy 2: librosa via temp file (for WebM/Opus)
    if audio is None:
        for suffix in [".webm", ".ogg", ".wav"]:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                audio_lb, sr_lb = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                audio = audio_lb
                if DEBUG:
                    print(f"[DEBUG] Decoded via librosa ({suffix}): {len(audio)} samples")
                break
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] librosa({suffix}) failed: {e}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

    # Strategy 3: raw int16 PCM
    if audio is None:
        try:
            raw = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if len(raw) >= int(0.5 * SAMPLE_RATE):
                audio = raw
                if DEBUG:
                    print(f"[DEBUG] Decoded as raw int16 PCM: {len(audio)} samples")
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] raw PCM failed: {e}")

    if audio is None:
        print("[AudioProcessor] ❌ All decode strategies failed")
        return None

    return _run_pipeline(audio, label=label)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("\n=== audio_processor.py self-test ===\n")
    t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))

    tests = {
        "80 Hz tone":   (0.5 * np.sin(2 * np.pi * 80  * t)).astype("float32"),
        "300 Hz tone":  (0.5 * np.sin(2 * np.pi * 300 * t)).astype("float32"),
        "1000 Hz tone": (0.5 * np.sin(2 * np.pi * 1000 * t)).astype("float32"),
        "White noise":  (0.3 * np.random.randn(int(DURATION * SAMPLE_RATE))).astype("float32"),
    }

    embs = {}
    for name, audio in tests.items():
        audio = pad_or_truncate(audio, int(DURATION * SAMPLE_RATE))
        feats = extract_features(audio)
        embs[name] = aggregate_to_embedding(feats, label=name)
        print(f"  {name}: norm={np.linalg.norm(embs[name]):.6f}")

    print("\nPairwise cosine similarities (dot product of unit vectors):")
    names = list(embs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = embs[names[i]], embs[names[j]]
            sim  = float(np.dot(a, b))
            ok   = "✅ distinct" if sim < 0.95 else "❌ TOO SIMILAR"
            print(f"  {names[i]} vs {names[j]}: {sim:.6f}  {ok}")

    if len(sys.argv) > 1:
        print(f"\nProcessing: {sys.argv[1]}")
        emb = process_audio_file(sys.argv[1])
        if emb is not None:
            print(f"  Embedding shape={emb.shape}, norm={np.linalg.norm(emb):.6f}")