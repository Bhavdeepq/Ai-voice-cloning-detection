"""Minimal FastAPI interface for the finalized voice-clone detector."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .detector import VoiceDetector


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the immutable detector once for the lifetime of the application."""
    app.state.detector = VoiceDetector()
    yield


app = FastAPI(title="Voice Clone Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> dict[str, str]:
    """Return a simple service status response."""
    return {"status": "ok"}


@app.post("/detect")
async def detect_audio(file: UploadFile | None = File(default=None)) -> dict[str, object]:
    """Analyze an uploaded audio file with the finalized VoiceDetector."""
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="An audio file is required.")

    suffix = Path(file.filename).suffix or ".wav"
    temporary_path: Path | None = None
    try:
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(audio_data)
            temporary_path = Path(temporary_file.name)

        return app.state.detector.predict(temporary_path)
    except HTTPException:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Unable to process audio file: {error}") from error
    finally:
        if file is not None:
            await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
