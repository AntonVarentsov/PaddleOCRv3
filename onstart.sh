#!/bin/bash
set -e
cd /app

# Allow overriding port via env, default to 8000
PORT="${PORT:-8000}"

echo "Starting API on port ${PORT}..."
exec uvicorn api.ocr_with_tables:app --host 0.0.0.0 --port "${PORT}"
