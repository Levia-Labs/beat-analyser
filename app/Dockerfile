# Built from the repository root (docker-compose.yml sets the context), because the image
# needs both app/ and the beatcore crate.
#
# Four stages. The web stage builds the SPA with Node. The Rust toolchain comes from the
# official image and is used only to build the beatcore wheel; the runtime stage copies the
# finished virtualenv and the SPA's dist/, so neither Node, Rust nor a compiler ships in the
# image.

# The SPA in web/, built once; only its dist/ reaches the runtime image, so Node does not.
FROM node:26.11.1-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
# The About page shows this; after the COPY so a new commit rebuilds only the SPA build.
ARG VITE_COMMIT=dev
ENV VITE_COMMIT=$VITE_COMMIT
RUN npm run build


FROM rust:1.98-slim AS rust


FROM python:3.14.7-slim AS build

# gcc is the linker rustc calls. The base image tag is pinned; an unpinned apt package takes
# Debian security updates on rebuild.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/opt/venv/bin:/usr/local/cargo/bin:$PATH

RUN python -m venv /opt/venv
COPY app/requirements.txt /tmp/requirements.txt
# pip is upgraded on purpose; the requirements themselves are pinned in app/requirements.txt.
# hadolint ignore=DL3013
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Last, so an edit to the crate rebuilds only this layer.
COPY beatcore /tmp/beatcore
RUN pip install --no-cache-dir /tmp/beatcore


FROM python:3.14.7-slim

# The base image tag is pinned; an unpinned apt package takes Debian security updates on rebuild.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UPLOAD_ROOT=/app/uploads \
    WEB_DIST=/app/web

# Unprivileged, and uid 1000 so the bind-mounted uploads/ is writable on a typical Linux host.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/uploads \
    && chown app:app /app/uploads

WORKDIR /app
COPY app/*.py ./
COPY app/models ./models
COPY --from=web /web/dist ./web

USER app
EXPOSE 8000

# One worker so the note model loads once; threads for concurrent requests. An analysis can
# run for minutes on a long song, so the worker timeout is 30 minutes, not 30 seconds.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "1800", "--worker-tmp-dir", "/tmp", "app:app"]
