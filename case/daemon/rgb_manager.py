import time
import psutil
import logging
import colorsys

try:
    from rpi_ws281x import PixelStrip, Color
    HAS_RGB = True
except ImportError:
    HAS_RGB = False
    logging.warning("rpi_ws281x not found. RGB Manager running in Mock mode.")

# --- LED Strip Settings ---
LED_COUNT = 3        # Depends on standard Ice Tower RGB fans, usually small number per ring, assuming 3 for demo or adjust. Many generic case fans have 3 or more.
LED_PIN = 18         # Default GPIO pin connected to the pixels
LED_FREQ_HZ = 800000 
LED_DMA = 10         
LED_BRIGHTNESS = 100 
LED_INVERT = False   
LED_CHANNEL = 0      

class RGBManager:
    def __init__(self, mode="breathing"):
        self.mode = mode
        self.strip = None
        self.breathing_step = 0
        self.breathing_dir = 1
        
        if HAS_RGB:
            self.strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
            self.strip.begin()

    def set_color(self, red, green, blue):
        if not HAS_RGB:
            logging.debug(f"[RGB Mock] Setting lights to R:{red} G:{green} B:{blue}")
            return
        
        color = Color(int(red), int(green), int(blue))
        for i in range(self.strip.numPixels()):
             self.strip.setPixelColor(i, color)
        self.strip.show()

    def _get_cpu_temp(self):
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current
            elif "coretemp" in temps:
                return temps["coretemp"][0].current
        return 45.0 # Default if unknown

    def loop_tick(self):
        """Called constantly in the main RGB thread to update the requested pattern"""
        if self.mode == "breathing":
             # Sentinel Breathing (Default) - Cyan / Deep Blue hue
             # Adjust brightness sinusoidally
             # Let's run a simple triangle wave on brightness
             brightness = 30 + (self.breathing_step * 2) # between 30 and 130
             
             # Color base: Deep Cyan
             r = 0
             g = int(brightness * 0.7)
             b = brightness
             self.set_color(r, g, b)

             self.breathing_step += self.breathing_dir
             if self.breathing_step >= 50 or self.breathing_step <= 0:
                 self.breathing_dir *= -1
                 
             time.sleep(0.05)
             
        elif self.mode == "thermal":
             # Thermal Gradient Map
             # <40 Blue, 50 Green, >65 Red
             temp = self._get_cpu_temp()
             
             if temp < 45:
                 self.set_color(0, 0, 255) # Blue
             elif temp < 60:
                 self.set_color(0, 255, 0) # Green
             else:
                 self.set_color(255, 0, 0) # Red
                 
             time.sleep(1)

        elif self.mode == "inference":
             # Flash green rapidly
             self.set_color(0, 255, 0)
             time.sleep(0.2)
             self.set_color(0, 0, 0)
             time.sleep(0.2)
             
        else:
             time.sleep(1)
