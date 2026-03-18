"""
================================================================================
VoicePrint-ID MVP | Phase 1
audio_processor.py — v3 (debug + format-aware)
================================================================================

ROOT CAUSE OF 100% BUG (confirmed v3):
    The browser's MediaRecorder produces WebM/Opus blobs, not WAV.
    When librosa fails to decode these (no ffmpeg), it either:
      a) returns silence → all-zero MFCC → same embedding for everyone
      b) raises an exception → we return None → API fails silently

    The fix here:
      1. Detect and handle browser audio blobs properly via soundfile fallback
      2. Add DEBUG mode that prints raw MFCC stats so you can see if features
         are actually different between speakers
      3. Removed CMVN (destroys discrimination) — kept from v2
      4. Use mean-only pooling + L2 norm — kept from v2
      5. Uses librosa mel filterbank with explicit fmin/fmax for speech range

SET DEBUG = True below to see per-request feature stats in your server console.
================================================================================
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Union, Tuple, Optional
import io

# ── Set True to print feature stats for every processed audio clip ──
DEBUG = True

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
SAMPLE_RATE  = 16_000
DURATION     = 3.0
N_MFCC       = 40
N_FFT        = 512
HOP_LENGTH   = 160
N_MELS       = 80
PRE_EMPHASIS = 0.97


def load_audio(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Load and resample audio file to SAMPLE_RATE mono float32."""
    audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), sr


def apply_preemphasis(audio: np.ndarray, coef: float = PRE_EMPHASIS) -> np.ndarray:
    """Boost high-frequency formants: y[n] = x[n] - coef * x[n-1]"""
    return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)


def voice_activity_detection(audio: np.ndarray) -> np.ndarray:
    """Trim silence. Return original if result is too short (<0.5s)."""
    trimmed, _ = librosa.effects.trim(audio, top_db=20, frame_length=512, hop_length=128)
    return trimmed if len(trimmed) >= int(0.5 * SAMPLE_RATE) else audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Fix length to target_length: center-crop if long, tile-pad if short."""
    n = len(audio)
    if n >= target_length:
        start = (n - target_length) // 2
        return audio[start:start + target_length]
    repeats = (target_length // n) + 1
    return np.tile(audio, repeats)[:target_length]


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract MFCC + delta + delta² feature matrix.
    Shape: (N_MFCC * 3, T) = (120, ~300)

    Key parameters for speech:
        fmin=20, fmax=7600 — covers full human speech range
        n_fft=512  — ~32ms window at 16kHz (standard for speech)
        hop_length=160 — 10ms frame shift (standard)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=20, fmax=7600,
    )
    delta  = librosa.feature.delta(mfcc, order=1, width=5)
    delta2 = librosa.feature.delta(mfcc, order=2, width=5)
    return np.vstack([mfcc, delta, delta2])   # (120, T)


def aggregate_to_embedding(features: np.ndarray, label: str = "") -> np.ndarray:
    """
    Mean-pool (120, T) → (120,) then L2-normalize to unit vector.

    WHY mean-only (not mean+std):
        std captures within-utterance variability which depends more on
        what words were spoken than who is speaking → hurts discrimination.

    WHY L2-normalize:
        Cosine similarity on unit vectors = dot product.
        Removes recording volume as a factor.

    DEBUG output shows the raw MFCC stats so you can verify
    different speakers produce genuinely different feature values.
    """
    mean_vec  = features.mean(axis=1)   # (120,)
    norm      = np.linalg.norm(mean_vec)

    if DEBUG:
        # Print per-coefficient stats — if all speakers show same values,
        # the audio decoding is failing (silence or corrupt data)
        raw_mfcc_mean = features[:N_MFCC, :].mean()
        raw_mfcc_std  = features[:N_MFCC, :].std()
        print(f"[DEBUG{' '+label if label else ''}] "
              f"MFCC mean={raw_mfcc_mean:.3f}  std={raw_mfcc_std:.3f}  "
              f"embedding norm={norm:.4f}  "
              f"shape={features.shape}")
        if raw_mfcc_std < 0.5:
            print(f"[DEBUG] ⚠️  Very low MFCC std — audio may be silence or corrupt!")

    if norm < 1e-10:
        print("[AudioProcessor] ⚠️  Near-zero norm — returning zero vector")
        return np.zeros(features.shape[0], dtype=np.float32)

    return (mean_vec / norm).astype(np.float32)


def process_audio_file(path: Union[str, Path], label: str = "") -> Optional[np.ndarray]:
    """
    Full pipeline for an audio file on disk.
    Returns 120-dim L2-normalized embedding or None on error.
    """
    try:
        path_str = str(path)
        audio, sr = load_audio(path_str)

        if DEBUG:
            print(f"[DEBUG] Loaded '{Path(path_str).name}': "
                  f"{len(audio)} samples, sr={sr}, "
                  f"rms={np.sqrt(np.mean(audio**2)):.5f}")

        # Check for silence / near-silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-4:
            print(f"[AudioProcessor] ⚠️  Audio appears to be silence (rms={rms:.6f})")

        audio = apply_preemphasis(audio)
        audio = voice_activity_detection(audio)
        audio = pad_or_truncate(audio, int(DURATION * SAMPLE_RATE))

        features  = extract_mfcc(audio, SAMPLE_RATE)
        embedding = aggregate_to_embedding(features, label=label or Path(path_str).name)

        if not np.isfinite(embedding).all():
            print(f"[AudioProcessor] ❌ Non-finite embedding for {path}")
            return None

        return embedding

    except Exception as exc:
        print(f"[AudioProcessor] ❌ Error: {exc}")
        import traceback; traceback.print_exc()
        return None


def process_audio_bytes(audio_bytes: bytes, label: str = "bytes") -> Optional[np.ndarray]:
    """
    Pipeline for raw bytes from browser / WebSocket.

    Strategy:
        1. Try soundfile (handles WAV, FLAC, OGG)
        2. Try librosa via temp file (handles WebM/Opus if ffmpeg present)
        3. Try raw int16 PCM interpretation as last resort
    """
    import tempfile, os

    if DEBUG:
        print(f"[DEBUG] Received audio bytes: {len(audio_bytes)} bytes")

    audio = None
    sr    = SAMPLE_RATE

    # ── Strategy 1: soundfile (WAV, FLAC, OGG Vorbis) ──
    try:
        import soundfile as sf
        buf         = io.BytesIO(audio_bytes)
        audio_sf, sr_sf = sf.read(buf, dtype='float32', always_2d=False)
        if audio_sf.ndim > 1:
            audio_sf = audio_sf.mean(axis=1)
        if sr_sf != SAMPLE_RATE:
            audio_sf = librosa.resample(audio_sf, orig_sr=sr_sf, target_sr=SAMPLE_RATE)
        audio = audio_sf
        if DEBUG:
            print(f"[DEBUG] Decoded via soundfile: {len(audio)} samples, "
                  f"rms={np.sqrt(np.mean(audio**2)):.5f}")
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] soundfile failed: {e}")

    # ── Strategy 2: temp file + librosa (WebM/Opus needs ffmpeg) ──
    if audio is None:
        for suffix in [".webm", ".ogg", ".wav"]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                audio_lb, sr_lb = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                os.unlink(tmp_path)
                audio = audio_lb
                if DEBUG:
                    print(f"[DEBUG] Decoded via librosa ({suffix}): "
                          f"{len(audio)} samples, rms={np.sqrt(np.mean(audio**2)):.5f}")
                break
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] librosa({suffix}) failed: {e}")
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    # ── Strategy 3: raw int16 PCM ──
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

    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-4:
        print(f"[AudioProcessor] ⚠️  Decoded audio is silence (rms={rms:.6f}) — check mic/format")

    audio     = apply_preemphasis(audio)
    audio     = voice_activity_detection(audio)
    audio     = pad_or_truncate(audio, int(DURATION * SAMPLE_RATE))
    features  = extract_mfcc(audio, SAMPLE_RATE)
    embedding = aggregate_to_embedding(features, label=label)

    if not np.isfinite(embedding).all():
        print("[AudioProcessor] ❌ Non-finite embedding")
        return None

    return embedding


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("\n=== audio_processor.py self-test ===\n")
    t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))

    tests = {
        "80 Hz tone":  (0.5 * np.sin(2 * np.pi * 80  * t)).astype("float32"),
        "300 Hz tone": (0.5 * np.sin(2 * np.pi * 300 * t)).astype("float32"),
        "White noise": (0.3 * np.random.randn(int(DURATION * SAMPLE_RATE))).astype("float32"),
    }

    embs = {}
    for name, audio in tests.items():
        audio = pad_or_truncate(audio, int(DURATION * SAMPLE_RATE))
        feats = extract_mfcc(audio)
        embs[name] = aggregate_to_embedding(feats, label=name)
        print(f"  {name}: norm={np.linalg.norm(embs[name]):.6f}")

    print("\nPairwise cosine similarities:")
    names = list(embs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = embs[names[i]], embs[names[j]]
            sim  = float(np.dot(a, b))
            ok   = "✅" if sim < 0.99 else "❌ TOO SIMILAR"
            print(f"  {names[i]} vs {names[j]}: {sim:.6f}  {ok}")

    if len(sys.argv) > 1:
        print(f"\nProcessing: {sys.argv[1]}")
        emb = process_audio_file(sys.argv[1])
        if emb is not None:
            print(f"  Embedding: shape={emb.shape}, norm={np.linalg.norm(emb):.6f}")
