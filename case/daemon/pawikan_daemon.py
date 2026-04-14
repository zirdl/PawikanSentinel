#!/usr/bin/env python3
import time
import threading
import argparse
import logging
import signal
import sys

from oled_manager import OLEDManager
from rgb_manager import RGBManager
from fan_manager import FanManager

# Configure root logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

running = True

def signal_handler(sig, frame):
    global running
    logging.info("Shutting down daemon...")
    running = False
    sys.exit(0)

def rgb_worker(rgb_manager):
    while running:
        rgb_manager.loop_tick()

def fan_worker(fan_manager):
    while running:
        fan_manager.loop_tick()

def main():
    parser = argparse.ArgumentParser(description="Pawikan Sentinel Case Daemon")
    parser.add_argument("--rgb-mode", choices=["breathing", "thermal", "inference"], default="breathing", help="Initial RGB led mode")
    parser.add_argument("--media", type=str, default="", help="Path to a custom image or gif to display alongside text")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("Starting Pawikan Case Daemon...")

    oled = OLEDManager(media_path=args.media)
    rgb = RGBManager(mode=args.rgb_mode)
    fan = FanManager()

    threading.Thread(target=rgb_worker, args=(rgb,), daemon=True).start()
    threading.Thread(target=fan_worker, args=(fan,), daemon=True).start()

    logging.info("Entering main loop...")
    
    last_view_switch = time.time()
    
    while running:
        # Every 5 seconds, change the view
        if time.time() - last_view_switch > 5.0:
            oled.next_view()
            last_view_switch = time.time()
            
        # Update OLED at 10 FPS to support smooth animations
        oled.update()
        time.sleep(0.1)

if __name__ == "__main__":
    main()
