#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Pawikan Sentinel installation..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null
then
    echo "Error: python3.11 is not installed. Please install it and try again."
    exit 1
fi

# Create a Python 3.11 virtual environment
echo "Creating virtual environment..."
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install uv within the virtual environment
echo "Installing uv..."
python -m pip install uv

# Install dependencies using uv
echo "Installing project dependencies..."
./.venv/bin/uv pip install tflite-runtime
./.venv/bin/uv pip install -r requirements.txt

echo "Installation complete. To activate the virtual environment, run: source .venv/bin/activate"
echo "You can now run the application using: python src/main.py"
echo "Remember to configure your config.ini file and set up the systemd service as per DEPLOYMENT_RPI.md."
