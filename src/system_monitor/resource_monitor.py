import psutil
import time
from typing import Optional

class ResourceMonitor:
    """
    Monitors system resources like CPU, memory, and temperature.
    """

    def get_cpu_usage(self) -> float:
        """
        Gets the current CPU usage as a percentage.

        Returns:
            float: The CPU usage percentage.
        """
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self) -> float:
        """
        Gets the current memory usage as a percentage.

        Returns:
            float: The memory usage percentage.
        """
        return psutil.virtual_memory().percent

    def get_cpu_temperature(self) -> Optional[float]:
        """
        Gets the CPU temperature in Celsius.

        Returns:
            float: The CPU temperature, or None if not available.
        """
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        return None

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Initialize the resource monitor
    resource_monitor = ResourceMonitor()

    # 2. Monitor the system for 10 seconds
    print("Monitoring system resources for 10 seconds...")
    for _ in range(10):
        cpu_usage = resource_monitor.get_cpu_usage()
        memory_usage = resource_monitor.get_memory_usage()
        cpu_temp = resource_monitor.get_cpu_temperature()

        print(f"- CPU: {cpu_usage}%, Memory: {memory_usage}%, Temp: {cpu_temp}°C")
        time.sleep(1)
