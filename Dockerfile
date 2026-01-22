# Triton Inference Server with CosyVoice Python Backend
# Based on Triton 24.07 with Python backend support
FROM nvcr.io/nvidia/tritonserver:24.07-py3

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        ffmpeg \
        sox \
        libsox-dev \
        curl \
        net-tools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy the FastCosyVoice codebase (contains pyproject.toml)
COPY FastCosyVoice /workspace/FastCosyVoice

# Copy Triton model repository
COPY models /triton_models

# Set PYTHONPATH for CosyVoice modules
ENV PYTHONPATH="${PYTHONPATH}:/workspace/FastCosyVoice"

# Install all dependencies using uv sync (same as the main Dockerfile)
# This ensures all deps including TensorRT, hyperpyyaml, etc are installed
WORKDIR /workspace/FastCosyVoice
RUN uv sync --no-dev

# Install Triton-specific dependencies (not in FastCosyVoice requirements)
RUN uv pip install pymongo

# Create models directory for CosyVoice model weights
RUN mkdir -p /models

# Set model directory environment variable
ENV COSYVOICE_MODEL_DIR="/models/Fun-CosyVoice3-0.5B"

# Download models from HuggingFace
RUN . .venv/bin/activate && python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/models/Fun-CosyVoice3-0.5B')"

# Expose Triton ports
EXPOSE 8000 8001 8002

# MongoDB configuration (set via environment variables at runtime)
# These are empty defaults - must be provided when running the container
ENV MONGO_URI=""
ENV MONGO_DB=""
ENV MONGO_COLLECTION=""

# TensorRT settings (same as main service)
ENV USE_TRT_FLOW=true
ENV USE_TRT_LLM=true
ENV TRT_LLM_DTYPE=bfloat16

# Set virtual environment for Triton to use
ENV VIRTUAL_ENV=/workspace/FastCosyVoice/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Start Triton server
CMD ["tritonserver", "--model-repository=/triton_models", "--log-verbose=1"]
