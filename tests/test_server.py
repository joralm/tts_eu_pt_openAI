"""Server tests with a fake engine (no weights needed), plus an optional real-engine test."""
import io
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tts_eu_pt import SAMPLE_RATE, server


class FakeTTS:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._l = threading.Lock()

    def say(self, text, speed=1.0):
        with self._l:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        self.calls.append((text, speed))
        with self._l:
            self.active -= 1
        return (0.5 * np.sin(np.linspace(0, 100, 2400))).astype(np.float32)


@pytest.fixture
def fake(monkeypatch):
    f = FakeTTS()
    monkeypatch.setattr(server, "_tts", f)
    return f


@pytest.fixture
def client(fake):
    return TestClient(server.app, raise_server_exceptions=False)


def _speech(client, **kw):
    body = {"model": "tts-eu-pt", "input": "Olá", "voice": "pt-PT", "response_format": "wav"}
    body.update(kw)
    return client.post("/v1/audio/speech", json={k: v for k, v in body.items() if v is not None})


def test_legacy_tts_wav(client, fake):
    r = client.post("/tts", json={"text": "Bom dia"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
    assert fake.calls == [("Bom dia", 1.0)]


def test_speech_wav_header(client):
    r = _speech(client)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    d = r.content
    assert d[:4] == b"RIFF" and d[8:12] == b"WAVE"
    fmt, ch, rate, _, _, bits = struct.unpack("<HHIIHH", d[20:36])
    assert (fmt, ch, rate, bits) == (1, 1, SAMPLE_RATE, 16)


def test_speech_pcm_raw(client):
    r = _speech(client, response_format="pcm")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/pcm"
    assert r.content[:4] != b"RIFF"
    assert len(r.content) == 2400 * 2


def test_defaults_and_speed_forwarded(client, fake):
    r = client.post("/v1/audio/speech", json={"input": "x", "speed": 1.5})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert fake.calls == [("x", 1.5)]


@pytest.mark.parametrize("kw,status,code", [
    ({"input": None}, 400, "missing_input"),
    ({"input": "   "}, 400, "empty_input"),
    ({"input": 5}, 400, "invalid_input"),
    ({"model": "tts-1"}, 404, "model_not_found"),
    ({"voice": "alloy"}, 400, "unsupported_voice"),
    ({"response_format": "mp3"}, 400, "unsupported_format"),
    ({"speed": 10}, 400, "invalid_speed"),
    ({"speed": "fast"}, 400, "invalid_speed"),
    ({"stream_format": "sse"}, 400, "unsupported_stream_format"),
])
def test_validation_errors(client, kw, status, code):
    r = _speech(client, **kw)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code


def test_invalid_json(client):
    r = client.post("/v1/audio/speech", content=b"nope", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_synthesis_failure_hides_details(client, fake):
    def boom(*a, **k):
        raise RuntimeError("/secret/path")
    fake.say = boom
    r = _speech(client)
    assert r.status_code == 500 and "secret" not in r.text


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "list" and j["data"][0]["id"] == "tts-eu-pt"


def test_shared_engine_and_no_reload(client, fake, monkeypatch):
    monkeypatch.setattr(server, "TTS", lambda *a, **k: pytest.fail("engine reloaded"))
    assert server.get_tts() is fake
    client.post("/tts", json={"text": "a"})
    _speech(client)
    assert len(fake.calls) == 2


def test_consecutive_and_concurrent(client, fake):
    for _ in range(3):
        assert _speech(client, response_format="pcm").status_code == 200
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: _speech(client) if i % 2 else client.post("/tts", json={"text": "t"}),
                          range(16)))
    assert all(r.status_code == 200 for r in res)
    assert fake.max_active == 1


def test_dograh_pipecat_contract(client):
    """Pipecat's OpenAITTSService (used by Dograh) requests response_format='pcm' and reads
    24 kHz mono s16le bytes from the body; extra 'instructions' must not break the call."""
    r = client.post("/v1/audio/speech", json={
        "model": "tts-eu-pt", "input": "Olá", "voice": "pt-PT",
        "response_format": "pcm", "instructions": "ignored"})
    assert r.status_code == 200 and len(r.content) % 2 == 0


@pytest.mark.skipif(not os.environ.get("TTS_EU_PT_INTEGRATION"), reason="needs real weights")
def test_real_engine(monkeypatch):
    monkeypatch.setattr(server, "_tts", None)
    c = TestClient(server.app)
    r = c.post("/v1/audio/speech", json={"input": "Bom dia.", "response_format": "pcm"})
    assert r.status_code == 200 and len(r.content) > 1000
