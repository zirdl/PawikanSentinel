#!/bin/bash
# Pawikan Sentinel Management Script

# Source the environment
source /home/pi/pawikan-sentinel/venv/bin/activate

# Function to check service status
check_status() {
    echo "=== Service Status ==="
    sudo systemctl status pawikan-sentinel.service --no-pager -l
}

# Function to start service
start_service() {
    echo "Starting Pawikan Sentinel service..."
    sudo systemctl start pawikan-sentinel.service
    echo "Service started"
}

# Function to stop service
stop_service() {
    echo "Stopping Pawikan Sentinel service..."
    sudo systemctl stop pawikan-sentinel.service
    echo "Service stopped"
}

# Function to restart service
restart_service() {
    echo "Restarting Pawikan Sentinel service..."
    sudo systemctl restart pawikan-sentinel.service
    echo "Service restarted"
}

# Function to view logs
view_logs() {
    echo "=== Recent Logs ==="
    sudo journalctl -u pawikan-sentinel.service --no-pager -n 50
}

# Function to follow logs
follow_logs() {
    echo "=== Following Logs (Ctrl+C to exit) ==="
    sudo journalctl -u pawikan-sentinel.service -f
}

# Function to check system resources
check_resources() {
    echo "=== System Resources ==="
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    echo "Memory Usage:"
    free -h | grep Mem
    echo "Disk Usage:"
    df -h /home/pi
    echo "Temperature:"
    vcgencmd measure_temp
}

# Function to update software
update_software() {
    echo "Updating Pawikan Sentinel software..."
    cd /home/pi/pawikan-sentinel
    git pull
    pip install -r requirements-pi.txt
    sudo systemctl restart pawikan-sentinel.service
    echo "Software updated and service restarted"
}

# Main menu
case "$1" in
    status)
        check_status
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    logs)
        view_logs
        ;;
    follow)
        follow_logs
        ;;
    resources)
        check_resources
        ;;
    update)
        update_software
        ;;
    *)
        echo "Pawikan Sentinel Management Script"
        echo "Usage: $0 {status|start|stop|restart|logs|follow|resources|update}"
        echo ""
        echo "Commands:"
        echo "  status    - Check service status"
        echo "  start     - Start the service"
        echo "  stop      - Stop the service"
        echo "  restart   - Restart the service"
        echo "  logs      - View recent logs"
        echo "  follow    - Follow logs in real-time"
        echo "  resources - Check system resources"
        echo "  update    - Update software and restart service"
        exit 1
        ;;
esac