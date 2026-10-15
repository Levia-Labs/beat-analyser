# Two stages. The build stage carries the compiler and git that madmom's Cython extensions
# and its pinned commit need; the runtime stage copies only the finished virtualenv, so
# neither ships in the image.

FROM python:3.14.2-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
# numpy and Cython first: madmom's setup.py imports both, and --no-build-isolation is what
# lets it see them. Quoted pins: an unquoted `Cython>=0.25` in a RUN line is a shell
# redirect, which is how the previous Dockerfile installed an unpinned Cython.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "Cython==3.2.4" "numpy==2.4.4" \
    && pip install --no-cache-dir --no-build-isolation -r /tmp/requirements.txt


FROM python:3.14.2-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UPLOAD_ROOT=/app/uploads

# Unprivileged, and uid 1000 so the bind-mounted uploads/ is writable on a typical Linux host.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/uploads \
    && chown app:app /app/uploads

WORKDIR /app
COPY *.py ./
COPY templates ./templates
COPY static ./static

USER app
EXPOSE 8000

# One worker so the madmom models load once; threads for concurrent requests. An analysis
# runs for minutes, so the worker timeout is 30 minutes rather than gunicorn's 30 seconds.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "1800", "--worker-tmp-dir", "/tmp", "app:app"]
