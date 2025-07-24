"""
Frame Processor Component
"""

import time
import cv2
from typing import Optional, Tuple

class FrameProcessor:
    """
    Captures and preprocesses video frames from an RTSP camera.
    """
    def __init__(self, rtsp_url: str, frame_rate: int = 10, resolution: Optional[Tuple[int, int]] = (640, 480)):
        self.rtsp_url = rtsp_url
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.cap = None
        self.running = False

    def start(self):
        """
        Starts the frame capture process.
        """
        self.running = True
        self._connect()
        print("Frame processor started.")

    def stop(self):
        """
        Stops the frame capture process.
        """
        self.running = False
        if self.cap:
            self.cap.release()
        print("Frame processor stopped.")

    def _connect(self):
        """
        Connects to the RTSP stream.
        """
        while self.running and not (self.cap and self.cap.isOpened()):
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if not self.cap.isOpened():
                    print(f"Error opening RTSP stream: {self.rtsp_url}. Retrying in 5 seconds...")
                    time.sleep(5)
            except cv2.error as e:
                print(f"OpenCV error: {e}. Retrying in 5 seconds...")
                time.sleep(5)

    def get_frame(self):
        """
        Retrieves and preprocesses a single frame.
        """
        if not (self.cap and self.cap.isOpened()):
            self._connect()
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            self._connect()
            return None

        if self.resolution:
            frame = cv2.resize(frame, self.resolution)
        
        return frame

