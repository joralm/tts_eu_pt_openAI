#!/usr/bin/env python3
"""Minimal HTTP backend for the web-echo demo: POST text -> WAV bytes.

Browsers can't run the torch model directly, so this ~50-line FastAPI app loads the voice
once and exposes a single /tts endpoint. It also serves the static web-echo page.

    pip install "tts_eu_pt[server]"
    python -m tts_eu_pt.server            # then open http://localhost:8000

This lives inside the package (not in examples/) so it is importable after a plain
``pip install``, not only from a repo checkout.
"""
import io
import logging
import os
import threading
from pathlib import Path

import soundfile as sf
import numpy as np
from fastapi import FastAPI, Body, Request
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from tts_eu_pt import TTS, SAMPLE_RATE

app = FastAPI(title="tts_eu_pt web-echo")
_tts: TTS | None = None
_WEB = Path(__file__).resolve().parent / "web"
_log = logging.getLogger("tts_eu_pt.server")

# One shared engine => serialise inference so concurrent requests never interleave on it.
_tts_lock = threading.Lock()

# OpenAI-compatible API settings. The project ships a single voicepack, so there is exactly
# one real voice; the model id is only a label for clients.
DEFAULT_MODEL_ID = "tts-eu-pt"
VOICE_ID = "pt-PT"
_VOICE_ALIASES = {"pt-pt", "pt_pt", "tts_eu_pt", "tts-eu-pt"}
_FORMATS = {"wav": "audio/wav", "pcm": "audio/pcm"}
_SPEED_MIN, _SPEED_MAX = 0.25, 4.0
_MAX_INPUT_CHARS = 4096


def model_id() -> str:
    return os.environ.get("TTS_EU_PT_MODEL_ID", DEFAULT_MODEL_ID)


def get_tts() -> TTS:
    """Load the voice once, on first request.

    TTS_EU_PT_MODEL / TTS_EU_PT_VOICEPACK let you point the demo at local weights, which
    is the only way to run it until the Hugging Face model repo is published (see
    tts_eu_pt.download).
    """
    global _tts
    if _tts is None:
        _tts = TTS(
            model_path=os.environ.get("TTS_EU_PT_MODEL"),
            voicepack_path=os.environ.get("TTS_EU_PT_VOICEPACK"),
        )
    return _tts


@app.post("/tts")
def tts(text: str = Body(..., embed=True)):
    """Synthesize `text` and return a 24 kHz mono WAV."""
    wav = _synthesize(text)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


def _synthesize(text: str, speed: float = 1.0) -> np.ndarray:
    engine = get_tts()
    with _tts_lock:
        return engine.say(text, speed=speed) if speed != 1.0 else engine.say(text)


def _error(status: int, message: str, param: str | None = None, code: str | None = None,
           etype: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "param": param, "code": code}},
    )


@app.get("/v1/models")
def list_models():
    """Model listing (OpenAI shape) for clients that probe it; only the local model."""
    return {"object": "list",
            "data": [{"id": model_id(), "object": "model", "created": 0, "owned_by": "tts_eu_pt"}]}


@app.post("/v1/audio/speech")
async def openai_speech(request: Request):
    """OpenAI-compatible speech endpoint. Returns raw audio bytes (wav or 16-bit pcm)."""
    try:
        body = await request.json()
    except Exception:
        return _error(400, "Request body must be valid JSON.", code="invalid_json")
    if not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object.", code="invalid_json")

    text = body.get("input")
    if text is None:
        return _error(400, "Missing required field: 'input'.", "input", "missing_input")
    if not isinstance(text, str):
        return _error(400, "'input' must be a string.", "input", "invalid_input")
    if not text.strip():
        return _error(400, "'input' must not be empty.", "input", "empty_input")
    if len(text) > _MAX_INPUT_CHARS:
        return _error(400, f"'input' exceeds {_MAX_INPUT_CHARS} characters.", "input", "input_too_long")

    model = body.get("model")
    if model is not None and model != model_id():
        return _error(404, f"The model '{model}' does not exist. Available: '{model_id()}'.",
                      "model", "model_not_found")

    voice = body.get("voice")
    if voice is not None and (not isinstance(voice, str) or voice.lower() not in _VOICE_ALIASES):
        return _error(400, f"Unsupported voice. Only '{VOICE_ID}' is available.", "voice", "unsupported_voice")

    fmt = body.get("response_format", "wav")
    if not isinstance(fmt, str) or fmt.lower() not in _FORMATS:
        return _error(400, f"Unsupported response_format. Supported: {', '.join(_FORMATS)}.",
                      "response_format", "unsupported_format")
    fmt = fmt.lower()

    speed = body.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) \
            or not (_SPEED_MIN <= speed <= _SPEED_MAX):
        return _error(400, f"'speed' must be a number between {_SPEED_MIN} and {_SPEED_MAX}.",
                      "speed", "invalid_speed")

    if body.get("stream_format", "audio") != "audio":
        return _error(400, "Only stream_format 'audio' is supported.", "stream_format",
                      "unsupported_stream_format")

    try:
        wav = await run_in_threadpool(_synthesize, text, float(speed))
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return _error(422, "The input text produced no audio.", "input", "no_audio")
        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
        if fmt == "pcm":
            return Response(content=pcm.tobytes(), media_type=_FORMATS["pcm"])
        buf = io.BytesIO()
        sf.write(buf, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return Response(content=buf.getvalue(), media_type=_FORMATS["wav"])
    except Exception:
        _log.exception("speech synthesis failed")
        return _error(500, "Speech synthesis failed.", code="synthesis_failed", etype="server_error")


@app.get("/")
def index():
    return FileResponse(_WEB / "index.html")


# static assets (app.js, etc.)
app.mount("/static", StaticFiles(directory=_WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
