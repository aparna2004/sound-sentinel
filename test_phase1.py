"""
================================================================================
VoicePrint-ID MVP | Phase 1
tests/test_phase1.py — Unit & Integration Tests
================================================================================

RUN:
    pytest tests/test_phase1.py -v

COVERS:
    1. AudioProcessor — feature extraction pipeline
    2. SpeakerRecognizer — enroll, identify, verify, delete
    3. API endpoints — via FastAPI TestClient (no server needed)
================================================================================
"""

import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import soundfile as sf

from core.audio_processor import (
    apply_preemphasis,
    voice_activity_detection,
    pad_or_truncate,
    extract_mfcc,
    normalize_features,
    aggregate_to_fixed_vector,
    process_audio_file,
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
)
from core.speaker_recognizer import SpeakerRecognizer


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

def make_tone(freq: float = 220.0, duration: float = 3.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a pure sine tone as a synthetic voice substitute."""
    t = np.linspace(0, duration, int(duration * sr))
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def tmp_wav(tmp_path) -> Path:
    """Create a temporary WAV file with a 220 Hz tone."""
    audio = make_tone(220.0)
    path  = tmp_path / "test_220hz.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    return path


@pytest.fixture
def tmp_wav2(tmp_path) -> Path:
    """Create a second temporary WAV file with a different tone (440 Hz)."""
    audio = make_tone(440.0)
    path  = tmp_path / "test_440hz.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    return path


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Temporary speaker database JSON path."""
    return tmp_path / "speakers.json"


# ---------------------------------------------------------------------------
# 1. AUDIO PROCESSOR TESTS
# ---------------------------------------------------------------------------

class TestAudioProcessor:

    def test_preemphasis_shape(self):
        """Pre-emphasis should preserve signal length."""
        audio  = make_tone()
        result = apply_preemphasis(audio)
        assert result.shape == audio.shape

    def test_preemphasis_differs_from_input(self):
        """Pre-emphasis should modify the signal."""
        audio  = make_tone()
        result = apply_preemphasis(audio)
        assert not np.allclose(audio, result), "Pre-emphasis had no effect"

    def test_vad_does_not_crash_on_silence(self):
        """VAD on all-zeros should return an array (possibly empty)."""
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result  = voice_activity_detection(silence)
        assert isinstance(result, np.ndarray)

    def test_pad_shorter_signal(self):
        """Padding a short signal should produce exactly target_length samples."""
        short  = make_tone(duration=1.0)
        target = int(DURATION * SAMPLE_RATE)
        padded = pad_or_truncate(short, target)
        assert len(padded) == target

    def test_truncate_longer_signal(self):
        """Truncating a long signal should produce exactly target_length samples."""
        long_  = make_tone(duration=6.0)
        target = int(DURATION * SAMPLE_RATE)
        cropped = pad_or_truncate(long_, target)
        assert len(cropped) == target

    def test_exact_length_unchanged(self):
        """A signal already at target_length should pass through unchanged."""
        target = int(DURATION * SAMPLE_RATE)
        audio  = make_tone(duration=DURATION)
        result = pad_or_truncate(audio, target)
        assert len(result) == target

    def test_mfcc_output_shape(self):
        """MFCC output should have (N_MFCC*3, T) shape."""
        audio = make_tone()
        mfcc  = extract_mfcc(audio)
        assert mfcc.shape[0] == N_MFCC * 3, f"Expected {N_MFCC * 3} feature rows, got {mfcc.shape[0]}"
        assert mfcc.shape[1] > 0, "Time dimension is empty"

    def test_normalize_mean_approx_zero(self):
        """After CMVN, per-feature mean should be ≈ 0."""
        audio    = make_tone()
        mfcc     = extract_mfcc(audio)
        normed   = normalize_features(mfcc)
        means    = normed.mean(axis=1)
        assert np.allclose(means, 0, atol=1e-6), "Feature means not zeroed after normalization"

    def test_aggregate_vector_size(self):
        """Statistics pooling should produce a (N_MFCC*3*2,) = 240-dim vector."""
        audio    = make_tone()
        mfcc     = extract_mfcc(audio)
        normed   = normalize_features(mfcc)
        vec      = aggregate_to_fixed_vector(normed)
        expected = N_MFCC * 3 * 2
        assert vec.shape == (expected,), f"Expected ({expected},), got {vec.shape}"

    def test_process_audio_file_returns_vector(self, tmp_wav):
        """End-to-end pipeline should return a valid numpy vector."""
        vec = process_audio_file(tmp_wav)
        assert vec is not None, "process_audio_file returned None"
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert not np.any(np.isnan(vec)), "Vector contains NaN values"

    def test_process_audio_file_bad_path(self):
        """A non-existent file should return None gracefully."""
        vec = process_audio_file("/tmp/nonexistent_file_xyz.wav")
        assert vec is None


# ---------------------------------------------------------------------------
# 2. SPEAKER RECOGNIZER TESTS
# ---------------------------------------------------------------------------

class TestSpeakerRecognizer:

    def test_empty_database(self, tmp_db):
        """Freshly initialized recognizer should have no speakers."""
        rec = SpeakerRecognizer(db_path=tmp_db)
        assert len(rec.list_speakers()) == 0

    def test_enroll_single_sample(self, tmp_wav, tmp_db):
        """Enrolling one sample should create a speaker profile."""
        rec    = SpeakerRecognizer(db_path=tmp_db)
        result = rec.enroll("user_001", "Alice", audio_paths=[str(tmp_wav)])
        assert result["success"] is True
        assert result["speaker_id"] == "user_001"
        assert result["n_samples"] == 1
        assert len(rec.list_speakers()) == 1

    def test_enroll_multiple_samples(self, tmp_wav, tmp_wav2, tmp_db):
        """Enrolling multiple samples should merge into one profile."""
        rec    = SpeakerRecognizer(db_path=tmp_db)
        result = rec.enroll("user_002", "Bob", audio_paths=[str(tmp_wav), str(tmp_wav2)])
        assert result["success"] is True
        assert result["n_samples"] == 2

    def test_enroll_invalid_path(self, tmp_db):
        """Enrolling with all-bad paths should return failure."""
        rec    = SpeakerRecognizer(db_path=tmp_db)
        result = rec.enroll("user_bad", "Fail", audio_paths=["/tmp/no_such_file.wav"])
        assert result["success"] is False

    def test_identify_known_speaker(self, tmp_wav, tmp_wav2, tmp_db):
        """After enrollment, identifying a similar audio should return that speaker."""
        rec = SpeakerRecognizer(db_path=tmp_db, threshold=0.0)  # low threshold for tone test
        rec.enroll("user_001", "Alice", audio_paths=[str(tmp_wav)])
        result = rec.identify(audio_path=str(tmp_wav))
        assert result["identified"] is True
        assert result["speaker_id"] == "user_001"

    def test_identify_no_speakers(self, tmp_wav, tmp_db):
        """Identification with empty database should return unknown with a message."""
        rec    = SpeakerRecognizer(db_path=tmp_db)
        result = rec.identify(audio_path=str(tmp_wav))
        assert result["identified"] is False
        assert "message" in result

    def test_verify_correct_speaker(self, tmp_wav, tmp_db):
        """Verifying a speaker against their own enrollment sample should succeed."""
        rec = SpeakerRecognizer(db_path=tmp_db, threshold=0.0)
        rec.enroll("user_001", "Alice", audio_paths=[str(tmp_wav)])
        result = rec.verify("user_001", audio_path=str(tmp_wav))
        assert result["verified"] is True

    def test_verify_nonexistent_speaker(self, tmp_wav, tmp_db):
        """Verifying against a non-enrolled speaker ID should return not-verified."""
        rec    = SpeakerRecognizer(db_path=tmp_db)
        result = rec.verify("ghost_user", audio_path=str(tmp_wav))
        assert result["verified"] is False

    def test_delete_speaker(self, tmp_wav, tmp_db):
        """Deleting an enrolled speaker should remove them from the database."""
        rec = SpeakerRecognizer(db_path=tmp_db)
        rec.enroll("user_001", "Alice", audio_paths=[str(tmp_wav)])
        assert len(rec.list_speakers()) == 1
        rec.delete_speaker("user_001")
        assert len(rec.list_speakers()) == 0

    def test_persistence_across_instances(self, tmp_wav, tmp_db):
        """Profiles saved by one instance should be loadable by another."""
        rec1 = SpeakerRecognizer(db_path=tmp_db)
        rec1.enroll("user_001", "Alice", audio_paths=[str(tmp_wav)])

        rec2 = SpeakerRecognizer(db_path=tmp_db)
        speakers = rec2.list_speakers()
        assert any(s["speaker_id"] == "user_001" for s in speakers)

    def test_cosine_similarity_same_vector(self, tmp_db):
        """Cosine similarity of a vector with itself should be 1.0."""
        rec = SpeakerRecognizer(db_path=tmp_db)
        v   = np.random.rand(240)
        sim = rec._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self, tmp_db):
        """Cosine similarity of orthogonal vectors should be ≈ 0."""
        rec = SpeakerRecognizer(db_path=tmp_db)
        v1  = np.array([1.0, 0.0])
        v2  = np.array([0.0, 1.0])
        sim = rec._cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6


# ---------------------------------------------------------------------------
# 3. API ENDPOINT TESTS (FastAPI TestClient — no server required)
# ---------------------------------------------------------------------------

class TestAPI:

    @pytest.fixture(autouse=True)
    def setup_client(self, tmp_db, monkeypatch):
        """
        Patch the recognizer dependency to use a temp database,
        then create a TestClient for the FastAPI app.
        """
        import api.server as server_module
        from fastapi.testclient import TestClient

        # Reset global singleton
        server_module._recognizer = SpeakerRecognizer(db_path=tmp_db)
        self.client = TestClient(server_module.app)

    def test_health_ok(self):
        res = self.client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"

    def test_list_speakers_empty(self):
        res = self.client.get("/api/v1/speakers")
        assert res.status_code == 200
        assert res.json() == []

    def test_enroll_and_list(self, tmp_wav):
        with open(tmp_wav, "rb") as f:
            res = self.client.post(
                "/api/v1/speakers/enroll",
                data={"speaker_id": "test_001", "speaker_name": "Test User"},
                files=[("files", ("test.wav", f, "audio/wav"))],
            )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True

        # Now list should have 1 speaker
        list_res = self.client.get("/api/v1/speakers")
        assert len(list_res.json()) == 1

    def test_get_speaker_not_found(self):
        res = self.client.get("/api/v1/speakers/nonexistent")
        assert res.status_code == 404

    def test_identify_no_speakers(self, tmp_wav):
        with open(tmp_wav, "rb") as f:
            res = self.client.post(
                "/api/v1/speakers/identify",
                files=[("file", ("test.wav", f, "audio/wav"))],
            )
        assert res.status_code == 200
        assert res.json()["identified"] is False

    def test_delete_speaker(self, tmp_wav):
        # Enroll first
        with open(tmp_wav, "rb") as f:
            self.client.post(
                "/api/v1/speakers/enroll",
                data={"speaker_id": "del_001", "speaker_name": "Delete Me"},
                files=[("files", ("test.wav", f, "audio/wav"))],
            )
        # Delete
        res = self.client.delete("/api/v1/speakers/del_001")
        assert res.status_code == 200
        assert res.json()["success"] is True

    def test_delete_nonexistent_speaker(self):
        res = self.client.delete("/api/v1/speakers/nobody")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Run directly with: python tests/test_phase1.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
