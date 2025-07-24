"""
Alert Manager Component
"""

import time
from typing import List, Dict

class AlertManager:
    """
    Manages alert generation, deduplication, and delivery.
    """
    def __init__(self, deduplication_window: int = 600): # 10 minutes
        self.deduplication_window = deduplication_window
        self.last_alert_time = 0.0

    def generate_alert(self, detections: List[Dict]):
        """
        Generates an alert if new, significant detections are found.
        """
        if not detections:
            return

        current_time = time.time()
        if current_time - self.last_alert_time > self.deduplication_window:
            self.last_alert_time = current_time
            message = f"Turtle detected! {len(detections)} turtles found."
            print(f"ALERT: {message}")
            # In a real implementation, this would trigger the SIM interface
        else:
            print("Detections found, but within deduplication window. No alert sent.")

