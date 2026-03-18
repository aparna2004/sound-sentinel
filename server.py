"""
================================================================================
VoicePrint-ID MVP | Phase 1
api/server.py — FastAPI REST API Server
================================================================================

PURPOSE:
    Exposes the SpeakerRecognizer as a REST API with the following endpoints:

    POST /api/v1/speakers/enroll     — Register a new speaker
    POST /api/v1/speakers/identify   — Identify a speaker from audio
    POST /api/v1/speakers/verify     — Verify a claimed speaker identity
    GET  /api/v1/speakers            — List all enrolled speakers
    GET  /api/v1/speakers/{id}       — Get a specific speaker's metadata
    DELETE /api/v1/speakers/{id}     — Delete a speaker from the database
    GET  /api/v1/health              — Health check

RUNNING:
    uvicorn api.server:app --reload --port 8000
    Then visit: http://localhost:8000/docs for interactive Swagger UI

PHASE ROADMAP:
    Phase 1 (THIS FILE) → File upload endpoints, REST API
    Phase 2             → WebSocket real-time streaming endpoint
    Phase 3             → Authentication, rate limiting, multi-tenant support
================================================================================
"""
import os

os.environ["PATH"] += os.pathsep + r"C:\Users\Aparna\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
import uuid
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# --- Internal imports ---
from speaker_recognizer import SpeakerRecognizer


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "VoicePrint-ID API",
    description = "Phase 1 MVP — Speaker Enrollment & Identification via MFCC + Cosine Similarity",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow all origins in development; tighten in Phase 3
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve the React/HTML dashboard from the dashboard directory
STATIC_DIR = Path(__file__).parent / "dashboard" / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Dependency: shared SpeakerRecognizer instance (singleton)
# ---------------------------------------------------------------------------

_recognizer: Optional[SpeakerRecognizer] = None

def get_recognizer() -> SpeakerRecognizer:
    """FastAPI dependency injection — returns the shared recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = SpeakerRecognizer()
    return _recognizer


# ---------------------------------------------------------------------------
# Pydantic response models (for clean OpenAPI schema)
# ---------------------------------------------------------------------------

class EnrollResponse(BaseModel):
    success:    bool
    speaker_id: str
    name:       Optional[str]   = None
    n_samples:  Optional[int]   = None
    message:    str

class IdentifyResponse(BaseModel):
    identified:   bool
    speaker_id:   str   = "unknown"
    speaker_name: str   = "Unknown"
    confidence:   float = 0.0
    all_scores:   dict  = Field(default_factory=dict)
    message:      Optional[str] = None

class VerifyResponse(BaseModel):
    verified:     bool
    speaker_id:   str
    speaker_name: str
    confidence:   float
    message:      Optional[str] = None

class SpeakerInfo(BaseModel):
    speaker_id: str
    name:       str
    n_samples:  int

class DeleteResponse(BaseModel):
    success: bool
    message: str

class HealthResponse(BaseModel):
    status:          str
    speakers_enrolled: int
    version:         str


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------

def save_upload_to_tmp(upload: UploadFile) -> Path:
    """
    Save an uploaded file to a temp path using the best-guess original suffix.
    """
    content_type = (upload.content_type or "").lower()
    filename = upload.filename or ""

    if "webm" in content_type:
        suffix = ".webm"
    elif "ogg" in content_type:
        suffix = ".ogg"
    elif "wav" in content_type:
        suffix = ".wav"
    elif "mpeg" in content_type or filename.lower().endswith(".mp3"):
        suffix = ".mp3"
    elif "flac" in content_type or filename.lower().endswith(".flac"):
        suffix = ".flac"
    else:
        suffix = Path(filename).suffix if filename else ".bin"
        if not suffix:
            suffix = ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()

    return Path(tmp.name)


def convert_to_wav(input_path: Path) -> Path:
    """
    Convert any uploaded audio file into a real mono 16 kHz PCM WAV.
    Returns the new temp wav path.
    """
    wav_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_tmp.close()
    output_path = Path(wav_tmp.name)

    ffmpeg_exe = shutil.which("ffmpeg")
    print("FFMPEG FOUND AT:", ffmpeg_exe)

    if not ffmpeg_exe:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            "ffmpeg is not installed or not found in PATH. "
            "Install ffmpeg so browser-recorded audio can be converted to WAV."
        )

    try:
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i", str(input_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return output_path

    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Audio conversion to WAV failed:\n{exc.stderr}")
# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check(rec: SpeakerRecognizer = Depends(get_recognizer)):
    """Health check — confirms the API is running and returns enrolled speaker count."""
    return HealthResponse(
        status            = "ok",
        speakers_enrolled = len(rec.profiles),
        version           = "1.0.0",
    )


@app.get("/api/v1/speakers", response_model=List[SpeakerInfo], tags=["Speakers"])
async def list_speakers(rec: SpeakerRecognizer = Depends(get_recognizer)):
    """List all enrolled speakers with their metadata."""
    return rec.list_speakers()


@app.get("/api/v1/speakers/{speaker_id}", response_model=SpeakerInfo, tags=["Speakers"])
async def get_speaker(speaker_id: str, rec: SpeakerRecognizer = Depends(get_recognizer)):
    """Get metadata for a specific speaker by ID."""
    speaker = rec.get_speaker(speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found.")
    return speaker


@app.post("/api/v1/speakers/enroll", response_model=EnrollResponse, tags=["Speakers"])
async def enroll_speaker(
    speaker_id:   str        = Form(..., description="Unique speaker ID (e.g. user_001)"),
    speaker_name: str        = Form(..., description="Display name (e.g. Alice)"),
    files:        List[UploadFile] = File(..., description="1-5 audio files (WAV/MP3/FLAC)"),
    rec:          SpeakerRecognizer = Depends(get_recognizer),
):
    """
    Enroll a new speaker by uploading 1-5 audio samples.

    **Best practice:** Provide 3-5 samples of natural speech (5-10 seconds each)
    for the most accurate voiceprint.

    If the speaker_id already exists, new samples are merged into the existing profile.
    """
    tmp_paths = []
    try:
        for upload in files:
            original_tmp = save_upload_to_tmp(upload)
            wav_tmp = convert_to_wav(original_tmp)

            tmp_paths.append(str(wav_tmp))

            # remove original uploaded temp after conversion
            original_tmp.unlink(missing_ok=True)

        result = rec.enroll(
            speaker_id   = speaker_id,
            speaker_name = speaker_name,
            audio_paths  = tmp_paths,
        )
        return EnrollResponse(**result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        # Always clean up temp files
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)


@app.post("/api/v1/speakers/identify", response_model=IdentifyResponse, tags=["Speakers"])
async def identify_speaker(
    file: UploadFile = File(..., description="Audio file to identify (WAV/MP3/FLAC)"),
    rec:  SpeakerRecognizer = Depends(get_recognizer),
):
    tmp_path = None
    original_tmp = None
    try:
        original_tmp = save_upload_to_tmp(file)
        tmp_path = convert_to_wav(original_tmp)

        result = rec.identify(audio_path=str(tmp_path))
        return IdentifyResponse(**result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if original_tmp:
            original_tmp.unlink(missing_ok=True)
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

@app.post("/api/v1/speakers/verify", response_model=VerifyResponse, tags=["Speakers"])
async def verify_speaker(
    speaker_id: str        = Form(..., description="The speaker ID being claimed"),
    file:       UploadFile = File(..., description="Audio to verify against claimed identity"),
    rec:        SpeakerRecognizer = Depends(get_recognizer),
):
    tmp_path = None
    original_tmp = None
    try:
        original_tmp = save_upload_to_tmp(file)
        tmp_path = convert_to_wav(original_tmp)

        result = rec.verify(
            claimed_speaker_id=speaker_id,
            audio_path=str(tmp_path),
        )
        return VerifyResponse(**result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if original_tmp:
            original_tmp.unlink(missing_ok=True)
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

@app.delete("/api/v1/speakers/{speaker_id}", response_model=DeleteResponse, tags=["Speakers"])
async def delete_speaker(
    speaker_id: str,
    rec:        SpeakerRecognizer = Depends(get_recognizer),
):
    """Remove a speaker from the database permanently."""
    result = rec.delete_speaker(speaker_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return DeleteResponse(**result)


# ---------------------------------------------------------------------------
# Dashboard redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the dashboard HTML."""
    dashboard_path = Path(__file__).parent / "index.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>VoicePrint-ID API</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host     = "0.0.0.0",
        port     = 8000,
        reload   = True,
        log_level= "info",
    )
