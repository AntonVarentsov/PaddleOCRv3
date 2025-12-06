FROM paddlepaddle/paddle:3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Default to TensorRT enabled; override with USE_TENSORRT=0 to disable (TensorRT already in base image)
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
