"""
================================================================================
VoicePrint-ID MVP | Phase 1
speaker_recognizer.py — v5 (cosine similarity, no whitening, correct thresholds)
================================================================================

WHY WE REMOVED WHITENING:
    Whitening (z-score normalization) requires a stable global distribution
    computed across many speakers. With 1–5 speakers it is numerically
    unstable — the global std is dominated by within-speaker variance, so
    the whitened query vector is distorted relative to the whitened centroid
    even when they come from the SAME recording. This is the root cause of
    the 65% confidence bug on enrollment files.

THE CORRECT APPROACH:
    audio_processor.py now returns L2-normalized unit vectors.
    Cosine similarity of two unit vectors = their dot product.
    Same speaker from same file → dot product ≈ 1.000 (by definition).
    Same speaker different file → dot product ≈ 0.75–0.95 (good cross-condition).
    Different speaker → dot product ≈ 0.10–0.55.

    The threshold is set at 0.70 — confidently above the inter-speaker ceiling
    and below the intra-speaker floor for well-recorded speech.

SCORING STRATEGY:
    For IDENTIFICATION (who is this person?):
        Score against each speaker's centroid (mean of all enrollment embeddings,
        re-normalized to unit sphere). Fast and works well with 3+ samples.

    For VERIFICATION (is this person X?):
        Best-of-N: score against every enrollment embedding and take the max.
        More robust when one enrollment sample is an outlier.
        Falls back to centroid if no individual samples stored.

    Both paths then apply a fused score:
        0.5 * centroid_score + 0.5 * best_sample_score
    which is more stable than either alone.

CONFIDENCE REPORTING:
    Returned confidence is the raw cosine similarity [0, 1].
    1.000 = identical embedding (same audio, same processing).
    0.85+ = strong same-speaker match.
    0.70–0.84 = likely same speaker.
    <0.70 = reject.
================================================================================
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from audio_processor import process_audio_file, process_audio_bytes


DEFAULT_DB_PATH = Path(__file__).parent / "data" / "speakers" / "profiles.json"

# Cosine similarity threshold on L2-normalized embeddings.
# Same speaker (cross-condition): 0.70 – 0.99
# Different speaker:              0.05 – 0.55
DEFAULT_THRESHOLD = 0.70

# Minimum gap between best and second-best speaker score (identification only).
# Prevents ambiguous decisions when two speakers score similarly.
GAP_MARGIN = 0.05

DEBUG = True


class SpeakerRecognizer:

    def __init__(self, db_path=None, threshold: float = DEFAULT_THRESHOLD):
        self.db_path   = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.threshold = threshold
        self.profiles: Dict[str, dict] = {}
        self._load_profiles()

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def _load_profiles(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for sid, p in raw.items():
                    p["centroid"] = np.array(p["centroid"], dtype=np.float32)
                    p["all_embeddings"] = [
                        np.array(e, dtype=np.float32)
                        for e in p.get("all_embeddings", [])
                    ]
                self.profiles = raw
                print(f"[SpeakerRecognizer] Loaded {len(self.profiles)} speaker(s)")
            except Exception as exc:
                print(f"[SpeakerRecognizer] Load failed: {exc}")
                self.profiles = {}
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.profiles = {}

    def _save_profiles(self) -> None:
        serializable = {}
        for sid, p in self.profiles.items():
            serializable[sid] = {
                "name": p["name"],
                "n_samples": p["n_samples"],
                "centroid": p["centroid"].tolist(),
                "all_embeddings": [
                    e.tolist() for e in p.get("all_embeddings", [])
                ],
            }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    # ------------------------------------------------------------------
    # SCORING — cosine similarity on unit vectors
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two vectors.
        If both are already L2-normalized (unit vectors), this equals dot(a, b).
        We re-normalize here defensively in case of any numerical drift.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a / norm_a, b / norm_b))

    def _score_against_profile(self, query: np.ndarray, profile: dict) -> dict:
        """
        Compute a fused similarity score against a speaker profile.

        Returns a dict with:
            score         — final fused score [0, 1]
            centroid_score — cosine sim vs centroid
            best_sample   — best cosine sim vs any individual sample
            n_samples     — how many samples contributed

        Fusion: 0.5 * centroid + 0.5 * best_sample
        Why? Centroid smooths out outlier samples; best_sample catches
        cases where the query is very close to one specific recording.
        """
        centroid = np.array(profile["centroid"], dtype=np.float32)
        centroid_score = self._cosine(query, centroid)

        samples = [
            np.array(e, dtype=np.float32)
            for e in profile.get("all_embeddings", [])
        ]

        if not samples:
            return {
                "score": centroid_score,
                "centroid_score": centroid_score,
                "best_sample": centroid_score,
                "n_samples": 0,
            }

        sample_scores = [self._cosine(query, s) for s in samples]
        best_sample = max(sample_scores)

        # Fused score: equal weight on centroid and best sample
        fused = 0.5 * centroid_score + 0.5 * best_sample

        return {
            "score": round(float(fused), 4),
            "centroid_score": round(float(centroid_score), 4),
            "best_sample": round(float(best_sample), 4),
            "n_samples": len(samples),
        }

    # ------------------------------------------------------------------
    # ENROLLMENT
    # ------------------------------------------------------------------

    def enroll(
        self,
        speaker_id: str,
        speaker_name: str,
        audio_paths: Optional[List[str]] = None,
        audio_bytes_list: Optional[List[bytes]] = None,
    ) -> dict:
        embeddings = []

        if audio_paths:
            for path in audio_paths:
                vec = process_audio_file(path, label=f"enroll:{speaker_name}")
                if vec is not None:
                    embeddings.append(vec)
                elif DEBUG:
                    print(f"[Enroll] Skipped invalid sample: {path}")

        if audio_bytes_list:
            for idx, raw in enumerate(audio_bytes_list):
                vec = process_audio_bytes(raw, label=f"enroll:{speaker_name}")
                if vec is not None:
                    embeddings.append(vec)
                elif DEBUG:
                    print(f"[Enroll] Skipped invalid byte sample #{idx + 1}")

        if not embeddings:
            return {
                "success": False,
                "speaker_id": speaker_id,
                "name": speaker_name,
                "n_samples": 0,
                "message": "No valid audio samples could be processed.",
            }

        if speaker_id in self.profiles:
            existing = list(self.profiles[speaker_id].get("all_embeddings", []))
            all_embs = existing + embeddings
        else:
            all_embs = embeddings

        # Centroid = mean of all embeddings, re-normalized to unit sphere
        stacked  = np.stack([np.array(e, dtype=np.float32) for e in all_embs])
        centroid = stacked.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-10:
            centroid = (centroid / centroid_norm).astype(np.float32)

        n_samples = len(all_embs)

        self.profiles[speaker_id] = {
            "name": speaker_name,
            "n_samples": n_samples,
            "centroid": centroid,
            "all_embeddings": all_embs,
        }

        self._save_profiles()

        if DEBUG:
            # Quality check: intra-speaker similarity of new samples
            if len(embeddings) >= 2:
                sims = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sims.append(self._cosine(embeddings[i], embeddings[j]))
                avg_sim = np.mean(sims)
                print(
                    f"[Enroll] {speaker_name} ({speaker_id}): "
                    f"{n_samples} total sample(s), "
                    f"avg intra-speaker cosine={avg_sim:.4f}"
                )
                if avg_sim < 0.60:
                    print(
                        f"[Enroll] ⚠️  Low intra-speaker similarity ({avg_sim:.3f}). "
                        f"Check audio quality or mic consistency."
                    )
            else:
                print(
                    f"[Enroll] {speaker_name} ({speaker_id}): "
                    f"{n_samples} sample(s) enrolled."
                )

        return {
            "success": True,
            "speaker_id": speaker_id,
            "name": speaker_name,
            "n_samples": n_samples,
            "message": f"Enrolled {speaker_name} with {len(embeddings)} new sample(s).",
        }

    # ------------------------------------------------------------------
    # IDENTIFICATION
    # ------------------------------------------------------------------

    def identify(
        self,
        audio_path: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
    ) -> dict:
        _empty = {
            "identified": False,
            "speaker_id": "unknown",
            "speaker_name": "Unknown",
            "confidence": 0.0,
            "all_scores": {},
        }

        if not self.profiles:
            return {**_empty, "message": "No speakers enrolled."}

        if audio_path:
            query = process_audio_file(audio_path, label="identify")
        elif audio_bytes:
            query = process_audio_bytes(audio_bytes, label="identify")
        else:
            return {**_empty, "message": "No audio provided."}

        if query is None:
            return {**_empty, "message": "Failed to process audio."}

        # Score against all enrolled speakers
        results = {}
        for sid, p in self.profiles.items():
            result = self._score_against_profile(query, p)
            results[sid] = result

        # Sort by fused score descending
        sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
        best_id, best_result = sorted_results[0]
        best_score = best_result["score"]

        gap = (
            best_score - sorted_results[1][1]["score"]
            if len(sorted_results) > 1
            else 1.0
        )

        if DEBUG:
            print(f"[Identify] All scores:")
            for sid, r in sorted_results:
                name = self.profiles[sid]["name"]
                print(
                    f"  {name} ({sid}): fused={r['score']:.4f}  "
                    f"centroid={r['centroid_score']:.4f}  "
                    f"best_sample={r['best_sample']:.4f}"
                )
            print(f"  → best={best_score:.4f}  gap={gap:.4f}  threshold={self.threshold}")

        all_scores_simple = {sid: r["score"] for sid, r in results.items()}

        if best_score >= self.threshold and gap >= GAP_MARGIN:
            return {
                "identified": True,
                "speaker_id": best_id,
                "speaker_name": self.profiles[best_id]["name"],
                "confidence": best_score,
                "all_scores": all_scores_simple,
                "detail": best_result,
            }
        else:
            msg = (
                f"Score {best_score:.3f} below threshold {self.threshold}"
                if best_score < self.threshold
                else f"Gap {gap:.3f} too small (need {GAP_MARGIN}) — ambiguous match"
            )
            return {
                **_empty,
                "confidence": best_score,
                "all_scores": all_scores_simple,
                "message": msg,
            }

    # ------------------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------------------

    def verify(
        self,
        claimed_speaker_id: str,
        audio_path: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
    ) -> dict:
        _empty = {
            "verified": False,
            "speaker_id": claimed_speaker_id,
            "speaker_name": "Unknown",
            "confidence": 0.0,
        }

        if claimed_speaker_id not in self.profiles:
            return {**_empty, "message": f"Speaker '{claimed_speaker_id}' not enrolled."}

        if audio_path:
            query = process_audio_file(audio_path, label=f"verify:{claimed_speaker_id}")
        elif audio_bytes:
            query = process_audio_bytes(audio_bytes, label=f"verify:{claimed_speaker_id}")
        else:
            return {**_empty, "message": "No audio provided."}

        if query is None:
            return {**_empty, "message": "Failed to process audio."}

        profile = self.profiles[claimed_speaker_id]
        result  = self._score_against_profile(query, profile)
        score   = result["score"]
        name    = profile["name"]
        verified = score >= self.threshold

        if DEBUG:
            print(
                f"[Verify] {name} ({claimed_speaker_id}): "
                f"fused={score:.4f}  "
                f"centroid={result['centroid_score']:.4f}  "
                f"best_sample={result['best_sample']:.4f}  "
                f"threshold={self.threshold}  "
                f"→ {'✅ VERIFIED' if verified else '❌ REJECTED'}"
            )

        return {
            "verified": verified,
            "speaker_id": claimed_speaker_id,
            "speaker_name": name,
            "confidence": score,
            "detail": result,
        }

    # ------------------------------------------------------------------
    # DATABASE MANAGEMENT
    # ------------------------------------------------------------------

    def list_speakers(self) -> List[dict]:
        return [
            {"speaker_id": sid, "name": p["name"], "n_samples": p["n_samples"]}
            for sid, p in self.profiles.items()
        ]

    def delete_speaker(self, speaker_id: str) -> dict:
        if speaker_id not in self.profiles:
            return {"success": False, "message": f"Speaker '{speaker_id}' not found."}
        name = self.profiles.pop(speaker_id)["name"]
        self._save_profiles()
        return {"success": True, "message": f"Deleted '{name}' ({speaker_id})."}

    def get_speaker(self, speaker_id: str) -> Optional[dict]:
        if speaker_id not in self.profiles:
            return None
        p = self.profiles[speaker_id]
        return {"speaker_id": speaker_id, "name": p["name"], "n_samples": p["n_samples"]}


if __name__ == "__main__":
    rec = SpeakerRecognizer()
    print(f"Enrolled speakers: {rec.list_speakers()}")