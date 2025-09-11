#!/bin/bash
# run-inference.sh - Script to run the inference application

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Activate virtual environment and run the inference application
echo "Starting inference application..."
source .venv/bin/activate
uvicorn src.inference.inference_service:app --host 0.0.0.0 --port 8001 --reload