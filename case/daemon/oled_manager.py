import os
import psutil
import socket
import logging
from PIL import Image, ImageDraw, ImageFont, ImageSequence

try:
    from luma.core.interface.serial import i2c
    from luma.core.render import canvas
    from luma.oled.device import ssd1306
    HAS_OLED = True
except ImportError:
    HAS_OLED = False
    logging.warning("luma or specialized dependencies not found. OLED Manager running in Mock mode.")

class OLEDManager:
    def __init__(self, media_path=None, width=128, height=32, i2c_port=1, i2c_address=0x3C):
        self.width = width
        self.height = height
        self.device = None

        if HAS_OLED:
            serial = i2c(port=i2c_port, address=i2c_address)
            self.device = ssd1306(serial, width=self.width, height=self.height)
        
        self.font = ImageFont.load_default()
        
        self.view_index = 0
        self.views = [self._render_sys_info, self._render_media_brand]
        
        self.tick_count = 0
        
        # Pre-process media frames (GIF or static image)
        self.media_frames = []
        if media_path and os.path.exists(media_path):
            logging.info(f"Loading custom media from {media_path}")
            try:
                img = Image.open(media_path)
                # Ensure the image fits alongside the text (e.g. 32x32 bounding box)
                max_size = (self.height, self.height)
                
                for frame in ImageSequence.Iterator(img):
                    # Convert to RGBA first to handle transparency/backgrounds, then to "1" (monochrome)
                    f = frame.copy().convert("RGBA")
                    
                    # Create a white background bounding box to handle transparent gifs properly
                    bg = Image.new("RGBA", max_size, (255, 255, 255, 255))
                    f.thumbnail(max_size, Image.LANCZOS)
                    
                    # Center the frame inside the 32x32 box
                    offset_x = (max_size[0] - f.width) // 2
                    offset_y = (max_size[1] - f.height) // 2
                    bg.paste(f, (offset_x, offset_y), f)
                    
                    # Convert to standard 1-bit mode for luma OLED
                    self.media_frames.append(bg.convert("1"))
                    
            except Exception as e:
                logging.error(f"Failed to load media {media_path}: {e}")
        else:
            if media_path:
                logging.warning(f"Media file not found at {media_path}")

    def _get_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _get_cpu_temp(self):
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            for key in ["cpu_thermal", "cpu-thermal", "coretemp"]:
                if key in temps and temps[key]:
                    return temps[key][0].current
        return 0.0

    def next_view(self):
        self.view_index = (self.view_index + 1) % len(self.views)

    def _render_sys_info(self):
        # Create a fresh black canvas
        img = Image.new("1", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)
        
        ip_addr = self._get_ip()
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        cpu_temp = self._get_cpu_temp()
        
        draw.text((0, -2), f"IP: {ip_addr}", font=self.font, fill="white")
        draw.text((0, 8), f"CPU: {cpu_usage:.1f}% T: {cpu_temp:.1f}C", font=self.font, fill="white")
        draw.text((0, 18), f"RAM: {ram.percent}%", font=self.font, fill="white")
        return img

    def _render_media_brand(self):
        img = Image.new("1", (self.width, self.height), "black")
        
        # Paste the current animation frame if available
        if self.media_frames:
            # Swap frame every 1 tick (assuming 10fps loop, so 10fps playback)
            frame_idx = self.tick_count % len(self.media_frames)
            current_frame = self.media_frames[frame_idx]
            
            # Since the device is monochrome, we swap the black bg with the image
            img.paste(current_frame, (0, 0))
            
        draw = ImageDraw.Draw(img)
        
        # Draw Title
        draw.text((40, 4), "PAWIKAN", font=self.font, fill="white")
        draw.text((40, 16), "SENTINEL", font=self.font, fill="white")
        return img

    def update(self):
        self.tick_count += 1
        
        # Render the current view to an Image object
        img = self.views[self.view_index]()
        
        if not HAS_OLED:
            if self.tick_count % 50 == 0:
                logging.debug(f"[OLED Mock] Rendered view {self.view_index} (Tick {self.tick_count})")
        else:
            self.device.display(img)
