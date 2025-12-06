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

# Install TensorRT runtime/dev libraries (Ubuntu 20.04 ML repo for CUDA 11.x)
RUN wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb && \
    dpkg -i nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb && \
    rm nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb && \
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
