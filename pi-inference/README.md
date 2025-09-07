# Pawikan Sentinel Pi Inference Package

This directory contains all the files needed to deploy and run the Pawikan Sentinel inference service on a Raspberry Pi.

## Files

- `inference_pi.py` - Contains the Pi-optimized inference worker
- `inference_service_pi.py` - FastAPI service for managing Pi inference
- `deploy_pi.sh` - Deployment script for setting up the service on a Pi
- `manage_pi.sh` - Management script for controlling the service
- `requirements-pi.txt` - Pi-specific Python dependencies
- `.env.pi.example` - Example environment configuration file
- `monitor_pi.py` - System monitoring utilities

## Deployment

1. Copy this entire directory to your Raspberry Pi:
   ```bash
   scp -r pi-inference pi@your-pi-ip:/home/pi/pawikan-sentinel/
   ```

2. Run the deployment script:
   ```bash
   cd /home/pi/pawikan-sentinel/pi-inference
   chmod +x deploy_pi.sh
   ./deploy_pi.sh
   ```

3. Configure environment variables:
   Copy `.env.pi.example` to `.env` and edit with your actual values:
   ```bash
   cp .env.pi.example .env
   nano .env
   ```

## Management

Use the management script for common tasks:
```bash
cd /home/pi/pawikan-sentinel/pi-inference
./manage_pi.sh status    # Check service status
./manage_pi.sh start     # Start service
./manage_pi.sh stop      # Stop service
./manage_pi.sh restart   # Restart service
./manage_pi.sh logs      # View recent logs
./manage_pi.sh follow    # Follow logs
./manage_pi.sh resources # Check system resources
./manage_pi.sh update    # Update software
```

## Pi-Specific Optimizations

- Limited to 2 concurrent camera workers
- Processes every 10th frame to reduce CPU load
- Resizes frames to 320x240 for faster processing
- Automatically pauses when CPU usage exceeds 80%