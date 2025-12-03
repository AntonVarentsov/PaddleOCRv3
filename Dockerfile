FROM paddlepaddle/paddle:3.2.0-gpu-cuda11.8-cudnn8.9

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

WORKDIR /app

# Copy and install Python dependencies (pinned to production versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY scripts/ scripts/
COPY ocr_with_tables.py .

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "ocr_with_tables:app", "--host", "0.0.0.0", "--port", "8000"]
