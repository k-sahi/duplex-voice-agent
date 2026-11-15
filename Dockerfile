# Container for the live demo (Google Cloud Run, or any Docker host).
FROM python:3.11-slim

# libgomp: OpenMP runtime used by the speech models' CPU kernels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch first (the default download includes ~2 GB of GPU libraries)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the Whisper model at build time so the first visitor doesn't wait for it
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

COPY --chown=user . .

# Hugging Face expects port 7860. Cloud Run overrides $PORT with its own value.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT}"]