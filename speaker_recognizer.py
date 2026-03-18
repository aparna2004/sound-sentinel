"""
================================================================================
VoicePrint-ID MVP | Phase 1
speaker_recognizer.py — v4.1 (whitened scoring + centroid/sample fusion)
================================================================================

WHY COSINE SIMILARITY ON MFCC MEANS FAILS:
    MFCC mean vectors for different speakers cluster very tightly in cosine
    space. The first few MFCC coefficients (especially C0/energy and C1)
    dominate the vector and are similar across speakers. Cosine similarity
    measures the ANGLE between vectors — but all speech MFCC means point
    in roughly the same direction regardless of speaker identity.

    Result: cosine similarity between ANY two speech clips is always 0.97–1.00.

THE FIX — Use Euclidean distance on whitened features:
    1. Collect all enrollment embeddings (not just the centroid)
    2. Compute per-dimension mean and std across ALL enrolled speakers
    3. Whiten (standardize) all embeddings using these global statistics
    4. Use EUCLIDEAN DISTANCE between whitened query and whitened centroids
    5. Convert distance to a score: score = 1 / (1 + distance)

ADDED IMPROVEMENT — Fuse centroid + sample-level scores:
    Centroid-only scoring can blur speaker variations when there are only a few
    enrollment samples. So now we score against:
        - the centroid
        - the best individual sample
        - the mean of top-2 samples
    and fuse them into one final score.

    This makes identification and verification more robust for small datasets.
================================================================================
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from audio_processor import process_audio_file, process_audio_bytes


DEFAULT_DB_PATH = Path(__file__).parent / "data" / "speakers" / "profiles.json"

# Distance threshold in whitened space.
# Same speaker:      distance ≈ 0.5 – 3.0  → score ≈ 0.25 – 0.67
# Different speaker: distance ≈ 4.0 – 15.0 → score ≈ 0.06 – 0.20
DEFAULT_THRESHOLD = 0.35

# Minimum score gap between best and second-best match
GAP_MARGIN = 0.05

# Debug: print scores to server console
DEBUG = True


class SpeakerRecognizer:

    def __init__(self, db_path=None, threshold: float = DEFAULT_THRESHOLD):
        self.db_path   = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.threshold = threshold
        self.profiles: Dict[str, dict] = {}
        # Global whitening statistics computed across all enrolled speakers
        self._global_mean: Optional[np.ndarray] = None
        self._global_std:  Optional[np.ndarray] = None
        self._load_profiles()

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def _load_profiles(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    raw = json.load(f)
                for sid, p in raw.items():
                    p["centroid"] = np.array(p["centroid"], dtype=np.float32)
                    p["all_embeddings"] = [
                        np.array(e, dtype=np.float32)
                        for e in p.get("all_embeddings", [])
                    ]
                self.profiles = raw
                self._recompute_whitening()
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
                "centroid": p["centroid"].tolist() if isinstance(p["centroid"], np.ndarray) else p["centroid"],
                "all_embeddings": [
                    e.tolist() if isinstance(e, np.ndarray) else e
                    for e in p.get("all_embeddings", [])
                ],
            }
        with open(self.db_path, "w") as f:
            json.dump(serializable, f, indent=2)

    # ------------------------------------------------------------------
    # WHITENING
    # ------------------------------------------------------------------

    def _recompute_whitening(self) -> None:
        """
        Compute global mean and std across ALL enrollment embeddings
        from ALL speakers. Used to whiten query embeddings before scoring.
        """
        all_vecs = []
        for p in self.profiles.values():
            for e in p.get("all_embeddings", []):
                all_vecs.append(np.array(e, dtype=np.float32))

        if len(all_vecs) < 2:
            self._global_mean = None
            self._global_std = None
            return

        matrix = np.stack(all_vecs)
        self._global_mean = matrix.mean(axis=0)
        self._global_std = matrix.std(axis=0) + 1e-8

        if DEBUG:
            print(
                f"[Whitening] Computed from {len(all_vecs)} embeddings across "
                f"{len(self.profiles)} speaker(s). "
                f"std range: {self._global_std.min():.4f} – {self._global_std.max():.4f}"
            )

    def _whiten(self, vec: np.ndarray) -> np.ndarray:
        """Apply global whitening: (vec - global_mean) / global_std"""
        if self._global_mean is None:
            return vec
        return (vec - self._global_mean) / self._global_std

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------

    def _score(self, query: np.ndarray, centroid: np.ndarray) -> float:
        """
        Compute similarity score between query and centroid.

        Uses Euclidean distance in whitened space, converted to [0,1] score:
            score = 1 / (1 + euclidean_distance)
        """
        q = self._whiten(query)
        c = self._whiten(centroid)
        dist = float(np.linalg.norm(q - c))
        score = 1.0 / (1.0 + dist)
        return round(score, 4)

    def _score_against_profile(self, query: np.ndarray, profile: dict) -> float:
        """
        More robust scoring against a full speaker profile.

        Combines:
        - centroid score
        - best individual sample score
        - mean of top-2 individual sample scores

        This helps when centroid-only scoring is too averaged for small
        enrollment sets.
        """
        centroid = np.array(profile["centroid"], dtype=np.float32)
        centroid_score = self._score(query, centroid)

        sample_embeddings = [
            np.array(e, dtype=np.float32)
            for e in profile.get("all_embeddings", [])
        ]

        if not sample_embeddings:
            return centroid_score

        sample_scores = [self._score(query, emb) for emb in sample_embeddings]
        sample_scores_sorted = sorted(sample_scores, reverse=True)

        best_sample_score = sample_scores_sorted[0]
        top2_mean = (
            float(np.mean(sample_scores_sorted[:2]))
            if len(sample_scores_sorted) >= 2
            else best_sample_score
        )

        fused_score = (
            0.4 * centroid_score +
            0.3 * best_sample_score +
            0.3 * top2_mean
        )

        return round(float(fused_score), 4)

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

        new_centroid = np.mean(np.stack(embeddings), axis=0)

        if speaker_id in self.profiles:
            existing_embs = self.profiles[speaker_id].get("all_embeddings", [])
            all_embs = existing_embs + embeddings
            n_samples = len(all_embs)
            centroid = np.mean(np.stack(all_embs), axis=0)
        else:
            all_embs = embeddings
            n_samples = len(embeddings)
            centroid = new_centroid

        self.profiles[speaker_id] = {
            "name": speaker_name,
            "n_samples": n_samples,
            "centroid": centroid,
            "all_embeddings": all_embs,
        }

        self._recompute_whitening()
        self._save_profiles()

        if DEBUG:
            print(
                f"[Enroll] {speaker_name} ({speaker_id}): "
                f"{n_samples} total sample(s), "
                f"centroid norm={np.linalg.norm(centroid):.4f}"
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

        scores = {}
        for sid, p in self.profiles.items():
            scores[sid] = self._score_against_profile(query, p)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = sorted_scores[0]
        gap = (best_score - sorted_scores[1][1]) if len(sorted_scores) > 1 else 1.0

        if DEBUG:
            print(f"[Identify] Scores: {sorted_scores}  | gap={gap:.4f} | threshold={self.threshold}")

        if best_score >= self.threshold and gap >= GAP_MARGIN:
            return {
                "identified": True,
                "speaker_id": best_id,
                "speaker_name": self.profiles[best_id]["name"],
                "confidence": best_score,
                "all_scores": scores,
            }
        else:
            msg = (
                f"Score {best_score:.3f} below threshold {self.threshold}"
                if best_score < self.threshold
                else f"Gap {gap:.3f} too small — ambiguous"
            )
            return {
                **_empty,
                "confidence": best_score,
                "all_scores": scores,
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
        score = self._score_against_profile(query, profile)
        name = profile["name"]

        if DEBUG:
            print(
                f"[Verify] {name} ({claimed_speaker_id}): score={score:.4f} "
                f"threshold={self.threshold} → {'VERIFIED' if score >= self.threshold else 'REJECTED'}"
            )

        return {
            "verified": score >= self.threshold,
            "speaker_id": claimed_speaker_id,
            "speaker_name": name,
            "confidence": score,
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
        self._recompute_whitening()
        self._save_profiles()
        return {"success": True, "message": f"Deleted '{name}' ({speaker_id})."}

    def get_speaker(self, speaker_id: str) -> Optional[dict]:
        if speaker_id not in self.profiles:
            return None
        p = self.profiles[speaker_id]
        return {"speaker_id": speaker_id, "name": p["name"], "n_samples": p["n_samples"]}


if __name__ == "__main__":
    rec = SpeakerRecognizer()
    print(f"Enrolled: {rec.list_speakers()}")
    print(f"Whitening ready: {rec._global_mean is not None}")