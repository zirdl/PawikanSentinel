import os
import json
import psutil
import requests
from datetime import datetime

def get_system_info():
    """Get system information for monitoring"""
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Memory info
    memory = psutil.virtual_memory()
    
    # Disk info
    disk = psutil.disk_usage('/')
    
    # Temperature (Raspberry Pi specific)
    temperature = None
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temperature = int(f.read()) / 1000.0
    except:
        pass
    
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "percent": cpu_percent
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": (disk.used / disk.total) * 100
        },
        "temperature": temperature
    }

def send_monitoring_data():
    """Send monitoring data to the main web application"""
    try:
        system_info = get_system_info()
        
        # In a real implementation, you would send this to your main web app
        # For now, we'll just print it
        print(json.dumps(system_info, indent=2))
        
        # Example of how to send to web app:
        # response = requests.post(
        #     "http://your-main-server.com/api/monitoring/pi-status",
        #     json=system_info,
        #     headers={"Authorization": "Bearer your-token"}
        # )
        # return response.status_code == 200
        
        return True
    except Exception as e:
        print(f"Error sending monitoring data: {e}")
        return False

if __name__ == "__main__":
    send_monitoring_data()