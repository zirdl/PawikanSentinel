#!/bin/bash
# Pawikan Sentinel Pi Deployment Script

set -e  # Exit on any error

echo "Starting Pawikan Sentinel Pi deployment..."

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
echo "Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git curl wget libgl1 libglib2.0-0

# Create project directory
echo "Creating project directory..."
mkdir -p /home/pi/pawikan-sentinel
cd /home/pi/pawikan-sentinel

# Clone or copy the project files
echo "Setting up project files..."
# In a real deployment, you would clone from your repository:
# git clone https://github.com/yourusername/pawikan-sentinel.git .
# For now, we assume files are already copied

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-pi.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p detections
mkdir -p logs

# Set up environment variables
echo "Setting up environment variables..."
cp .env.pi.example .env
echo "Please edit .env with your actual configuration values"

# Create systemd service file
echo "Creating systemd service..."
sudo tee /etc/systemd/system/pawikan-sentinel.service > /dev/null <<EOF
[Unit]
Description=Pawikan Sentinel Inference Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pawikan-sentinel
ExecStart=/home/pi/pawikan-sentinel/venv/bin/python -m pi-inference.inference_service_pi
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
echo "Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable pawikan-sentinel.service

echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Edit /home/pi/pawikan-sentinel/.env with your actual configuration"
echo "2. Start the service with: sudo systemctl start pawikan-sentinel.service"
echo "3. Check status with: sudo systemctl status pawikan-sentinel.service"
echo "4. View logs with: sudo journalctl -u pawikan-sentinel.service -f"