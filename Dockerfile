FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/huggingface

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/huggingface \
    && chown -R app:app /data

WORKDIR /app

# CPU-only torch (no CUDA wheels) keeps the image small.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"

COPY pyproject.toml README.md MANIFEST.in LICENSE ./
COPY tts_eu_pt ./tts_eu_pt
RUN pip install ".[server]"

USER app
VOLUME ["/data/huggingface"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/v1/models', timeout=4)" || exit 1

CMD ["python", "-m", "tts_eu_pt.server"]
