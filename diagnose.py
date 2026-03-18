"""
================================================================================
diagnose.py — Run this to see exactly what your embeddings look like
================================================================================
Usage:
    python diagnose.py path/to/speaker1.wav path/to/speaker2.wav

If you don't have files, it generates synthetic tones and tests those.
Run this BEFORE enrolling to confirm the pipeline is working correctly.
================================================================================
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audio_processor import (
    load_audio, apply_preemphasis, voice_activity_detection,
    pad_or_truncate, extract_mfcc, aggregate_to_embedding,
    process_audio_file, SAMPLE_RATE, DURATION
)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def print_sep(char="─", n=60):
    print(char * n)

print()
print_sep("═")
print("  VoicePrint-ID — Embedding Diagnostic")
print_sep("═")

# ── TEST 1: Synthetic tones (controlled test) ────────────────────
print("\n[TEST 1] Synthetic tones — should have LOW similarity\n")
t = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))

voices = {
    "Low tone  (80 Hz)":  (0.5 * np.sin(2 * np.pi * 80  * t)).astype("float32"),
    "Mid tone  (200 Hz)": (0.5 * np.sin(2 * np.pi * 200 * t)).astype("float32"),
    "High tone (600 Hz)": (0.5 * np.sin(2 * np.pi * 600 * t)).astype("float32"),
    "Noise":              (0.3 * np.random.randn(int(DURATION * SAMPLE_RATE))).astype("float32"),
}

embeddings = {}
for name, audio in voices.items():
    audio = pad_or_truncate(audio, int(DURATION * SAMPLE_RATE))
    feats = extract_mfcc(audio)
    emb   = aggregate_to_embedding(feats)
    embeddings[name] = emb
    print(f"  {name}: norm={np.linalg.norm(emb):.6f}  mean={emb.mean():.4f}  std={emb.std():.4f}")

print()
print("  Pairwise cosine similarities:")
names = list(embeddings.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        sim = cosine(embeddings[names[i]], embeddings[names[j]])
        flag = "⚠️  TOO HIGH" if sim > 0.95 else "✅"
        print(f"    {names[i]} vs {names[j]}: {sim:.6f}  {flag}")

# ── TEST 2: Same audio twice (should be ~1.0) ────────────────────
print()
print_sep()
print("\n[TEST 2] Same audio processed twice — should be ~1.000\n")
t2   = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE))
same = (0.4 * np.sin(2 * np.pi * 180 * t2) + 0.1 * np.random.randn(len(t2))).astype("float32")
same = pad_or_truncate(same, int(DURATION * SAMPLE_RATE))

e1 = aggregate_to_embedding(extract_mfcc(same))
e2 = aggregate_to_embedding(extract_mfcc(same))
sim = cosine(e1, e2)
flag = "✅" if sim > 0.999 else "⚠️  PROBLEM"
print(f"  Same audio twice: {sim:.8f}  {flag}")

# ── TEST 3: Real audio files (if provided) ───────────────────────
if len(sys.argv) >= 3:
    print()
    print_sep()
    print("\n[TEST 3] Real audio files\n")
    paths = sys.argv[1:]
    real_embeddings = {}
    for p in paths:
        emb = process_audio_file(p)
        if emb is not None:
            real_embeddings[p] = emb
            print(f"  {Path(p).name}: norm={np.linalg.norm(emb):.6f}  "
                  f"mean={emb.mean():.4f}  std={emb.std():.4f}")
        else:
            print(f"  ❌ Could not process: {p}")

    if len(real_embeddings) >= 2:
        print()
        print("  Pairwise cosine similarities:")
        rnames = list(real_embeddings.keys())
        for i in range(len(rnames)):
            for j in range(i+1, len(rnames)):
                sim = cosine(real_embeddings[rnames[i]], real_embeddings[rnames[j]])
                label = Path(rnames[i]).name + " vs " + Path(rnames[j]).name
                flag = "⚠️  TOO HIGH (same speaker?)" if sim > 0.92 else "✅  discriminative"
                print(f"    {label}: {sim:.6f}  {flag}")

# ── TEST 4: Check if profiles.json exists and show raw scores ────
print()
print_sep()
print("\n[TEST 4] profiles.json check\n")
db_path = Path("data") / "speakers" / "profiles.json"
if db_path.exists():
    import json
    with open(db_path) as f:
        profiles = json.load(f)
    print(f"  Found {len(profiles)} enrolled speaker(s):")
    for sid, p in profiles.items():
        centroid = np.array(p["centroid"])
        print(f"    {sid} ({p['name']}): "
              f"centroid norm={np.linalg.norm(centroid):.6f}  "
              f"n_samples={p['n_samples']}")

    if len(profiles) >= 2 and len(sys.argv) >= 2:
        print()
        print("  Query file vs all enrolled speakers (RAW cosine, no rescaling):")
        query = process_audio_file(sys.argv[1])
        if query is not None:
            for sid, p in profiles.items():
                c   = np.array(p["centroid"])
                sim = cosine(query, c)
                print(f"    vs {p['name']} ({sid}): {sim:.6f}")
else:
    print("  No profiles.json found — enroll speakers first, then re-run this script.")

print()
print_sep("═")
print("  Diagnostic complete.")
print_sep("═")
print()
