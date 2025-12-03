#!/bin/bash
cd /app
nohup uvicorn api.ocr_with_tables:app --host 0.0.0.0 --port 8000 > /var/log/api.log 2>&1 &
echo "API started on port 8000"
