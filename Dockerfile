FROM paddlepaddle/paddle:3.0.0b2-gpu-cuda11.8-cudnn8.6-trt8.5

# Install system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage cache
COPY api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY api/ api/
# We will copy scripts if they exist, but for now let's assume we copy everything in context if needed
# Or just copy scripts explicitly. 
# To avoid error if scripts dir doesn't exist yet (though I will create it), 
# I'll add it.
COPY scripts/ scripts/

# Expose port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
