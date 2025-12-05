FROM paddlepaddle/paddle:3.2.0-gpu-cuda11.8-cudnn8.9

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install TensorRT runtime/dev libraries (Ubuntu 22.04 repo for CUDA 11.8)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libnvonnxparsers8 \
    libnvparsers8 \
    libnvinfer-bin \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    libnvonnxparsers-dev \
    libnvparsers-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Default to TensorRT enabled; override with USE_TENSORRT=0 to disable
ENV USE_TENSORRT=1

WORKDIR /app

# Copy and install Python dependencies (pinned to production versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY scripts/ scripts/
COPY onstart.sh .

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["bash", "/app/onstart.sh"]
