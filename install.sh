#!/bin/bash

# This script sets up the Pawikan Sentinel application on a Raspberry Pi.
# It creates a virtual environment, installs dependencies, and prepares the application for execution.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Pawikan Sentinel Installation ---"

# Ensure the script is being run from the project root directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: This script must be run from the Pawikan Sentinel project root directory."
    exit 1
fi

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null
then
    echo "Error: python3.11 is not installed. Please install it and try again."
    exit 1
fi

# 1. Create a Python 3.11 virtual environment
echo "[1/4] Creating Python 3.11 virtual environment in ./.venv..."
python3.11 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install uv for faster package management
echo "[2/4] Installing 'uv' package manager..."
pip install uv

# 4. Install Python dependencies
echo "[3/4] Installing project dependencies..."
# Install tflite-runtime for ARM architecture (Raspberry Pi)
uv pip install tflite-runtime
# Install other dependencies from requirements.txt
uv pip install -r requirements.txt

# 5. Final instructions
echo "[4/4] Installation complete."
echo
echo "--- Next Steps ---"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo
echo "2. Configure your settings in 'config.ini':"
echo "   - Set the RTSP_URL for your camera."
echo "   - Add your Twilio credentials (ACCOUNT_SID, AUTH_TOKEN, PHONE_NUMBER, RECIPIENT_NUMBER, MESSAGING_SERVICE_SID)."
echo
echo "3. Run the application:"
echo "   python src/main.py"
echo
echo "For automatic startup, refer to DEPLOYMENT_RPI.md to set up the systemd service."
echo "-------------------------------------------------"

