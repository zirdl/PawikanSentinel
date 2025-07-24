"""
System Monitor Component
"""

import psutil

class SystemMonitor:
    """
    Monitors system health, performance, and resource usage.
    """
    def get_cpu_usage(self) -> float:
        """
        Returns the current CPU usage as a percentage.
        """
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self) -> dict:
        """
        Returns memory usage statistics.
        """
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "used": mem.used,
        }

    def get_temperature(self) -> float:
        """
        Returns the CPU temperature in Celsius.
        """
        # This is a placeholder. Actual implementation depends on the system.
        # For Raspberry Pi, you might read from /sys/class/thermal/thermal_zone0/temp
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0
                return temp
        except FileNotFoundError:
            return 0.0 # Not on a Raspberry Pi

