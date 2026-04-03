FROM python:3.14.2-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    libsndfile1 

# Upgrade pip and wheel
RUN pip install --upgrade pip setuptools wheel

# Install ffmpeg for audio processing
RUN apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install \
    blinker==1.9.0 \
    click==8.3.1 \
    colorama==0.4.6 \
    Cython>=0.25 \
    Flask==3.1.3 \
    itsdangerous==2.2.0 \
    Jinja2==3.1.6 \
    MarkupSafe==3.0.3 \
    mido>=1.2.6 \
    numpy==2.4.4 \
    packaging==26.0 \
    pydub==0.25.1 \
    scipy==1.17.1 \
    Werkzeug==3.1.7

# Install git and supporting tools
RUN apt-get update && apt-get install -y \
    git \
    gnupg \
    ca-certificates \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# Install madmom from GitHub
RUN pip install "madmom@git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"

# additional installs
RUN pip install librosa==0.10.2
RUN pip install yt-dlp

# Create directory for uploaded audio
RUN mkdir -p uploads

# Copy Flask app
COPY app.py . 
COPY templates ./templates

# Expose Flask port
EXPOSE 80

# Run Flask
CMD ["python", "app.py"]