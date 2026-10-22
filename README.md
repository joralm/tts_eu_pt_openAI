# tts_eu_pt - an European Portuguese text-to-speech based on Kokoro TTS

A small, **permissively-licensed** (Apache-2.0) **European Portuguese (pt-PT)** voice,
fine-tuned from [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). It runs on **CPU**,
ships as a tiny Python package.

## Why it's different

The hard part of pt-PT TTS isn't the acoustics — it's reading text the way a Portuguese
speaker actually does. `tts_eu_pt` handles, out of the box:

| Input | Spoken as |
|---|---|
| `25`, `125`, `1139` | *vinte cinco*, *cento e vinte cinco*, *mil cento e trinta e nove* |
| `16:54`, `16:00` | *dezasseis e cinquenta e quatro*, *dezasseis horas* |
| `1139-1185` | *mil cento e trinta e nove **a** mil cento e oitenta e cinco* |
| `D. Afonso I` | *Dom Afonso Primeiro* |
| `IA`, `LLM`, `UTC` | *i-á*, *éle-éle-éme*, *u-tê-cê* |
| `online`, `software` | kept in their English pronunciation (as Portuguese speakers say them) |

The G2P (Grapheme-to-Phoneme) is built on the Apache-2.0 [TugaPhone](https://github.com/TigreGotico/tugaphone)
front-end (Lisbon lect).

## Install

```bash
pip install tts_eu_pt
```

## Quick start

```python
from tts_eu_pt import TTS

tts = TTS()                                       # downloads the voice on first run
wav = tts.say("Olá! São dezasseis horas.")        # numpy float32 @ 24 kHz
tts.save("ola.wav", "Bem-vindo ao tts_eu_pt.")
```

Command line:

```bash
tts-eu-pt "D. Afonso I reinou de 1139 a 1185." reis.wav
tts-eu-pt --text "Bom dia." --speed 1.1 ola.wav      # or python -m tts_eu_pt.cli
```

The voice (~313 MB) downloads once from
[`logus2k/kokoro_tts_eu_pt`](https://huggingface.co/logus2k/kokoro_tts_eu_pt) and is cached
by `huggingface_hub`. To run offline or from a different copy, pass
`TTS(model_path=..., voicepack_path=...)` or set `TTS_EU_PT_REPO`.

## Web echo demo

A one-page demo: type a sentence, press Enter, the page speaks it back in European Portuguese.

```bash
pip install "tts_eu_pt[server]"
python -m tts_eu_pt.server            # open http://localhost:8000
```

A static page ([`tts_eu_pt/web/`](tts_eu_pt/web/)) plus a ~50-line FastAPI backend
([`tts_eu_pt/server.py`](tts_eu_pt/server.py)) — browsers can't run the torch model
directly. To run it against local weights, set `TTS_EU_PT_MODEL` and
`TTS_EU_PT_VOICEPACK`.

## Docker / Portainer

The image is CPU-only, runs as a non-root user and starts `python -m tts_eu_pt.server`
(binds `0.0.0.0`, port 8000).

```bash
docker build -t kokoro-pt:latest .                                   # build
docker run -d --name kokoro-pt -p 8000:8000 \
  -v kokoro_hf_cache:/data/huggingface --restart unless-stopped kokoro-pt:latest   # run
docker ps --filter name=kokoro-pt                                    # STATUS shows (healthy)
git pull && docker build -t kokoro-pt:latest . \
  && docker rm -f kokoro-pt && docker run -d --name kokoro-pt -p 8000:8000 \
  -v kokoro_hf_cache:/data/huggingface --restart unless-stopped kokoro-pt:latest   # update
```

Compose / Portainer stack: paste [`docker-compose.yml`](docker-compose.yml) into
*Stacks > Add stack* (Repository or Web editor), or run `docker compose up -d --build`;
update with `git pull && docker compose up -d --build`.

**Cache persistence.** `huggingface_hub` stores downloads under `$HF_HOME/hub`
(default `~/.cache/huggingface`). The image sets `HF_HOME=/data/huggingface` and the
compose file mounts the named volume `hf_cache` there, so weights are downloaded once on
first synthesis and survive restarts and image updates. The model loads lazily on the
first request.

**Verify the API:**

```bash
curl -s http://localhost:8000/v1/models
curl -s -X POST http://localhost:8000/tts -H 'Content-Type: application/json' \
  -d '{"text":"Olá, são dezasseis horas."}' -o out.wav && file out.wav
curl -s -X POST http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-eu-pt","input":"Olá!","voice":"pt-PT","response_format":"pcm"}' -o out.pcm
```

PCM is signed 16-bit little-endian, mono, 24 kHz.

**GHCR.** The image is ready to be published to `ghcr.io/joralm/tts_eu_pt_openai`
(`docker tag kokoro-pt:latest ghcr.io/joralm/tts_eu_pt_openai:latest` then
`docker push ...` after `docker login ghcr.io`). Nothing is published automatically.

## HTTP API (FastAPI server)

Run: `pip install -e ".[server]" && python -m tts_eu_pt.server` (port 8000).
Point `TTS_EU_PT_MODEL` / `TTS_EU_PT_VOICEPACK` at local weights if needed. One model instance is
loaded once and shared by all endpoints; inference is serialised with a lock.

| Endpoint | Purpose |
|---|---|
| `POST /tts` | Legacy (FreePBX). Body `{"text": "..."}` -> `audio/wav`, mono, 16-bit PCM, 24 kHz. Unchanged. |
| `POST /v1/audio/speech` | OpenAI-compatible speech. Returns raw audio bytes. |
| `GET /v1/models` | OpenAI-style model list (local model only). |
| `GET /` | Web echo page. |

**FreePBX compatibility:** `/tts` URL, payload, WAV response, pt-PT normalisation, model/voice
configuration and startup/port are unchanged. No authentication is required on any endpoint.

### `POST /v1/audio/speech`

| Field | Notes |
|---|---|
| `input` | Required, non-empty string, max 4096 chars. |
| `model` | Optional; must equal the model id (default `tts-eu-pt`, override with `TTS_EU_PT_MODEL_ID`), else 404. |
| `voice` | Optional; the only real voice is `pt-PT` (aliases `tts_eu_pt`, `tts-eu-pt`). Others -> 400. |
| `response_format` | `wav` (default, `audio/wav`) or `pcm` (`audio/pcm`: raw signed 16-bit little-endian, mono, 24 kHz, no header). `mp3`, `flac`, `opus`, `aac` -> 400. |
| `speed` | 0.25-4.0, passed to the engine (>1 is faster). |
| `stream_format` | Only `audio` accepted. |
| other fields (e.g. `instructions`) | Ignored; the engine has no support for them. |

Errors use the OpenAI shape `{"error": {"message", "type", "param", "code"}}` (400 validation,
404 unknown model, 422 no audio, 500 synthesis failure; no tracebacks are exposed).

```bash
curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-eu-pt","input":"Bom dia, são dezasseis horas.","voice":"pt-PT","response_format":"wav"}' \
  -o out.wav
curl -s http://localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-eu-pt","input":"Olá","voice":"pt-PT","response_format":"pcm"}' -o out.pcm
curl -s http://localhost:8000/tts -H 'Content-Type: application/json' -d '{"text":"Olá"}' -o legacy.wav
```

### Dograh

Configure the OpenAI TTS provider with Base URL `http://<host>:<port>/v1`, model `tts-eu-pt`,
voice `pt-PT`, any placeholder API key (it is not checked). Based on Pipecat's
`OpenAITTSService` (which Dograh builds on): it calls `POST {base_url}/audio/speech` with
`response_format="pcm"` and expects 24 kHz mono s16le, which matches this server; it does not call
`/v1/models` (the endpoint is provided only for clients that probe it). This was *not* verified
against a running Dograh instance here; the contract is covered by a test using that request shape.

### Known limitations

- No true streaming: audio is synthesised fully, then returned in one HTTP response (higher
  first-byte latency for long text; client cancellation does not interrupt a running synthesis).
- Requests are processed one at a time on the shared model; concurrent requests queue, so
  client timeouts should allow for that.
- Only `wav` and `pcm`; one voice; `instructions` is ignored.

## How it works

- **Weights** encode the *language* (pt-PT phonology/accent); a small **voicepack** encodes
  the *speaker*. They are separate files.
- Text → `tts_eu_pt.g2p` (normalisation + TugaPhone) → phoneme ids → Kokoro acoustic model
  → 24 kHz audio.
- The Kokoro acoustic model is vendored (`tts_eu_pt/kokoro/`, Apache-2.0). Only the acoustic
  `KModel` is used; the upstream pipeline's GPL-dependent G2P is not.

## Licensing

Apache-2.0.

## Status

Early release: the runtime + SDK + examples.

## Docker image (GHCR)

A GitHub Actions workflow (`.github/workflows/docker-image.yml`) builds the root `Dockerfile`
and pushes it on every push to `main` (or manually via *Run workflow*).

- **Published to:** `ghcr.io/<owner>/<repo>` (lower-cased automatically), e.g. `ghcr.io/joralm/tts_eu_pt_openai`.
- **Tags:** `main` (branch), `sha-<short commit>`, and `latest` (default branch).
- **Settings:** the workflow uses `GITHUB_TOKEN` with `packages: write`; ensure
  *Settings → Actions → General → Workflow permissions* allows it. New packages are private by
  default: change visibility under the package's *Package settings* to pull without login, or
  `docker login ghcr.io` with a token that has `read:packages`.
