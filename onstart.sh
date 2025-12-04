#!/bin/bash
set -e
cd /app
echo "Starting API on port 8000..."
exec uvicorn api.ocr_with_tables:app --host 0.0.0.0 --port 8000
