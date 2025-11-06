# Pawikan Sentinel Raspberry Pi Deployment Guide

## Overview
This guide explains how to deploy the Pawikan Sentinel inference service on a Raspberry Pi 4B for turtle monitoring in the field.

## Hardware Requirements
- Raspberry Pi 4B (4GB RAM recommended)
- microSD card (32GB or larger, Class 10 recommended)
- Power supply (USB-C, 5V/3A recommended)
- Case with cooling (fan or heatsinks recommended)
- Ethernet cable or WiFi for network connectivity

## Software Requirements
- Raspberry Pi OS (Bullseye or later recommended)
- Python 3.8 or later
- Docker (optional, for running Roboflow inference server)

## Deployment Steps

### 1. Prepare the Raspberry Pi
1. Install Raspberry Pi OS on the microSD card
2. Enable SSH and configure WiFi/Ethernet
3. Update the system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

### 2. Deploy the Inference Service
1. Copy the deployment files to the Pi:
   ```bash
   scp -r pawikan-sentinel pi@your-pi-ip:/home/pi/
   ```

2. Run the deployment script:
   ```bash
   cd /home/pi/pawikan-sentinel
   chmod +x pi-inference/deploy_pi.sh
   ./pi-inference/deploy_pi.sh
   ```

3. Configure environment variables:
   Edit `/home/pi/pawikan-sentinel/.env` with your actual values:
   - Roboflow API key
   - iprog credentials (if using SMS notifications)
   - Camera RTSP URLs

### 3. Start the Service
```bash
sudo systemctl start pawikan-sentinel.service
```

### 4. Monitor the Service
```bash
# Check status
sudo systemctl status pawikan-sentinel.service

# View logs
sudo journalctl -u pawikan-sentinel.service -f
```

## Pi-Specific Optimizations

### Resource Management
- Limited to 2 concurrent camera workers
- Processes every 10th frame to reduce CPU load
- Resizes frames to 320x240 for faster processing
- Automatically pauses when CPU usage exceeds 80%

### SMS Notifications
When a turtle is detected with high confidence (>80%), the system sends SMS notifications to emergency contacts using iprog (cost-effective for Philippine deployments).

## Management Commands
Use the management script for common tasks:
```bash
cd /home/pi/pawikan-sentinel
./pi-inference/manage_pi.sh status    # Check service status
./pi-inference/manage_pi.sh start     # Start service
./pi-inference/manage_pi.sh stop      # Stop service
./pi-inference/manage_pi.sh restart   # Restart service
./pi-inference/manage_pi.sh logs      # View recent logs
./pi-inference/manage_pi.sh follow    # Follow logs
./pi-inference/manage_pi.sh resources # Check system resources
./pi-inference/manage_pi.sh update    # Update software
```

## Remote Monitoring
The system includes monitoring capabilities that can report:
- CPU and memory usage
- Disk space
- Temperature
- Active camera workers

## Troubleshooting

### Common Issues
1. **High CPU Usage**: Check if too many cameras are active
2. **RTSP Connection Issues**: Verify camera URLs and network connectivity
3. **SMS Not Sending**: Check iprog credentials and account status

### Logs
Check logs for detailed error information:
```bash
sudo journalctl -u pawikan-sentinel.service -f
```

## Security Considerations
- Change default Pi password
- Use strong passwords for all services
- Keep system updated
- Restrict network access to necessary ports only
- Regularly backup configuration and detection data

## Maintenance
- Regularly clean detection images to save disk space
- Monitor system health and temperature
- Update software periodically
- Check camera connections and positioning