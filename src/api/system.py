from fastapi import APIRouter
import psutil
import time
import os

router = APIRouter(prefix="/api/system", tags=["System"])

START_TIME = time.time()

@router.get("/health")
def get_health():
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    
    temp = None
    try:
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read().strip()) / 1000.0
    except Exception:
        pass
        
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - START_TIME,
        "cpu_percent": cpu_percent,
        "ram_percent": ram.percent,
        "ram_used_mb": ram.used / (1024 * 1024),
        "disk_percent": disk.percent,
        "temperature_c": temp
    }
