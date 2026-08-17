# Container image for src/server.py, meant to be deployed to Cloud Run and
# fronted by Firebase Hosting. See FIREBASE_DEPLOY.md for the actual deploy
# steps -- this file only builds the image.

FROM python:3.11-slim

# libsndfile1: what soundfile/librosa need to decode audio.
# ffmpeg: broader format support (kept for future file-upload mode; wav/flac
# alone don't strictly need it, but it's cheap insurance).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# matplotlib is imported (unused, see requirements.txt) by spectogram.py at
# module load time -- Agg is the headless/no-display backend, which avoids
# it trying (and failing) to find a GUI toolkit inside the container.
ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Mirrors the repo's own layout (src/ and checkpoints/ as siblings) --
# server.py resolves the checkpoint path relative to its own file location,
# not the working directory, so this layout is what it expects.
COPY src/ /app/src/
COPY checkpoints/ /app/checkpoints/

WORKDIR /app/src

# Cloud Run sets $PORT at runtime (usually 8080) and expects the container
# to listen on it -- don't hardcode 8080 here, read the env var.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
