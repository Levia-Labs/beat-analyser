FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy scipy cython
RUN pip install --no-cache-dir madmom flask

WORKDIR /app

COPY app.py .
COPY templates ./templates

EXPOSE 5000

CMD ["python", "app.py"]