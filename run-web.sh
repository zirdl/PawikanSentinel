#!/bin/bash
# run-web.sh - Script to run the web application

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

# Build Tailwind CSS
./build-css.sh

# Activate virtual environment and run the web application
echo "Starting web application..."
source .venv/bin/activate
uvicorn src.core.main:app --host 0.0.0.0 --port 8000 --reload
