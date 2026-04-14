import time
import psutil
import logging

try:
    from gpiozero import PWMOutputDevice
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False
    logging.warning("gpiozero not found. Fan Manager running in Mock mode.")

# Default BCM GPIO pin for the PWM fan 
FAN_PIN = 14

class FanManager:
    def __init__(self, target_temp_low=45.0, target_temp_high=70.0):
        self.target_temp_low = target_temp_low
        self.target_temp_high = target_temp_high
        self.fan = None
        
        if HAS_GPIO:
            self.fan = PWMOutputDevice(FAN_PIN)
            self.fan.value = 0

    def _get_cpu_temp(self):
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current
            elif "coretemp" in temps:
                return temps["coretemp"][0].current
        return 45.0 # Default if unknown

    def loop_tick(self):
        """Called periodically to adjust fan speed based on temp."""
        temp = self._get_cpu_temp()
        
        # Calculate expected PWM duty cycle (0.0 to 1.0)
        if temp <= self.target_temp_low:
            speed = 0.0
        elif temp >= self.target_temp_high:
            speed = 1.0
        else:
            # Linear scaling
            speed = (temp - self.target_temp_low) / (self.target_temp_high - self.target_temp_low)
            
        if not HAS_GPIO:
            logging.debug(f"[Fan Mock] Temp: {temp:.1f}C -> Setting fan speed to {speed*100:.0f}%")
        else:
            self.fan.value = speed
            
        # Fan doesn't need to be updated instantly, wait 5 seconds
        time.sleep(5)
