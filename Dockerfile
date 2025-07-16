FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

COPY app.py ./
COPY assets/ ./
COPY backend/ ./
COPY html/ ./
COPY langchain_mlx/ ./
COPY models/ ./
COPY ComfyUI/ ./
COPY notebook/ ./
COPY outputs/ ./
COPY presets/ ./
COPY requirements/ ./
COPY src/ ./
COPY translations/ ./
COPY tts_outputs/ ./
COPY util/ ./
COPY workflows/ ./
COPY --chmod=777 installer_linux_amd64_cuda.sh ./

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    tesseract-ocr \
    tesseract-ocr-all \
    curl \
    wget \
    gpg \
    gcc \
    g++ \
    make \
    ffmpeg \
    sqlite3 \
    openjdk \
    openssl \
    ruby-full \
    git

ADD https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6.tar.gz /download/cmake-3.31.6/

WORKDIR /download/cmake-3.31.6
RUN ./bootstrap --prefix=/usr/local && \
    make && \
    make install

WORKDIR /app

RUN uv venv --python 3.12 && \
    source .venv/bin/activate && \
    ./installer_linux_amd64_cuda.sh

SHELL ["/bin/bash", "--login", "-c"]
CMD ["uv", "run", "app.py", "--mcp_server"]