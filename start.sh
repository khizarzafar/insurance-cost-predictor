#!/bin/bash

# start.sh — Startup Script for Docker Container
# ================================================
# This script runs inside the container when it starts.
# It launches BOTH services together:
#   1. FastAPI  on port 8000 (the backend API)
#   2. Streamlit on port 7860 (the frontend UI — required by Hugging Face Spaces)
#
# The '&' at the end of a command means "run in the background"
# The 'wait' at the end means "keep this script alive until all background jobs finish"
# Without 'wait', the script would exit immediately and Docker would stop the container.

echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit frontend on port 7860..."
streamlit run streamlit_app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false &

# Wait for both background processes to stay alive
# If either crashes, the container will also stop (which is correct behaviour)
wait
