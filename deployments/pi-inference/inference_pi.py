import os
import cv2
import time
from datetime import datetime
import threading
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
import psutil
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from src.inference.sms_sender import IprogSMSSender

from src.core.database import get_db_connection

# Load environment variables
load_dotenv()

# Configuration optimized for Raspberry Pi
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "pawikansentinel-era7l/2")
DETECTIONS_DIR = os.getenv("DETECTIONS_DIR", "detections")
# iprog configuration (replaces Semaphore)
IPROG_API_TOKEN = os.getenv("IPROG_API_TOKEN")
IPROG_SENDER_NAME = os.getenv("IPROG_SENDER_NAME", "PawikanSentinel")
SMS_NOTIFICATION_COOLDOWN = int(os.getenv("SMS_NOTIFICATION_COOLDOWN", "10"))  # Cooldown in minutes

# Pi-specific optimizations
MAX_WORKERS = int(os.getenv("MAX_INFERENCE_WORKERS", "2"))  # Reduced for Pi
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "10"))  # Process every 10th frame
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", "320"))  # Lower resolution
RESIZE_HEIGHT = int(os.getenv("RESIZE_HEIGHT", "240"))
CPU_THRESHOLD = int(os.getenv("CPU_THRESHOLD", "80"))  # Max CPU usage percentage
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))  # Confidence threshold (80%)

# Ensure directories exist
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    camera_id: int
    timestamp: str
    class_name: str
    confidence: float
    image_path: str

class PiOptimizedWorker(threading.Thread):
    def __init__(self, camera_id: int, rtsp_url: str, model_id: str, api_url: str, api_key: str):
        super().__init__(name=f"CameraWorker-{camera_id}")
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.model_id = model_id
        self.api_url = api_url
        self.api_key = api_key
        
        # Initialize inference client
        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key
        )
        
        # Initialize iprog SMS sender
        self.sms_sender = IprogSMSSender()
        
        # Worker state
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.detection_buffer = []
        self.last_sms_time = {}  # Track last SMS time per contact

    def _check_system_resources(self) -> bool:
        """Check if system resources are within acceptable limits"""
        current_time = time.time()
        # Check CPU usage every 5 seconds
        if current_time - self.last_cpu_check > 5:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            logger.debug(f"Pi Resources - CPU: {cpu_percent}%, Memory: {memory_percent}%")
            
            self.last_cpu_check = current_time
            
            # Pause if CPU usage is too high
            if cpu_percent > CPU_THRESHOLD:
                logger.warning(f"High CPU usage ({cpu_percent}%), pausing worker")
                return False
                
        return True

    def _send_sms_notification(self, detection: DetectionResult):
        """Send SMS notification for significant detections with cooldown"""
        if not self.sms_sender.is_enabled():
            logger.warning("SMS sender not enabled - skipping SMS notification")
            return
            
        try:
            # Get emergency contacts from database
            conn = get_db_connection()
            contacts = conn.execute("SELECT phone FROM contacts LIMIT 3").fetchall()
            conn.close()
            
            if not contacts:
                logger.warning("No contacts found for SMS notifications")
                return
            
            # Extract phone numbers from contacts
            phone_numbers = [contact["phone"] for contact in contacts if contact["phone"]]
            
            if not phone_numbers:
                logger.warning("No valid phone numbers found for SMS notifications")
                return
            
            # Send SMS notifications using Semaphore
            results = self.sms_sender.send_detailed_notification(
                phone_numbers=phone_numbers,
                class_name=detection.class_name,
                confidence=detection.confidence,
                timestamp=detection.timestamp
            )
            
            # Log the results
            for phone, success in results.items():
                if success:
                    logger.info(f"SMS sent successfully to {phone}")
                else:
                    logger.error(f"Failed to send SMS to {phone}")
                    
        except Exception as e:
            logger.error(f"Error sending SMS notifications: {e}")

    def _flush_detections(self):
        """Flush detection buffer to database"""
        if not self.detection_buffer:
            return

        conn = None
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            db_records = [
                (d.camera_id, d.timestamp, d.class_name, d.confidence, d.image_path)
                for d in self.detection_buffer
            ]
            
            c.executemany(
                "INSERT INTO detections (camera_id, timestamp, class, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
                db_records
            )
            conn.commit()
            logger.info(f"Camera {self.camera_id}: Flushed {len(self.detection_buffer)} detections to DB.")
            
            # Clear buffer after successful flush
            self.detection_buffer.clear()
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error flushing detections to DB: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

    def run(self):
        """Main worker loop optimized for Raspberry Pi"""
        self.running = True
        logger.info(f"Camera {self.camera_id}: Starting Pi-optimized inference worker")
        
        # Setup video capture
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error(f"Camera {self.camera_id}: Cannot open RTSP stream")
            self.running = False
            return

        try:
            while self.running:
                # Handle pause state
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Check system resources
                if not self._check_system_resources():
                    time.sleep(2)
                    continue

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Camera {self.camera_id}: Failed to grab frame")
                    time.sleep(2)
                    continue

                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % FRAME_SKIP != 0:
                    continue

                # Resize for faster processing
                resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
                
                # Perform inference
                try:
                    results = self.client.infer(resized, model_id=self.model_id)
                    
                    if "predictions" in results and results["predictions"]:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        for pred in results["predictions"]:
                            class_name = pred["class"]
                            confidence = pred["confidence"]
                            
                            # Save annotated frame
                            image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{self.camera_id}.jpg"
                            image_path = os.path.join(DETECTIONS_DIR, image_filename)
                            
                            # Save image
                            cv2.imwrite(image_path, resized)
                            
                            # Create detection result
                            detection = DetectionResult(
                                camera_id=self.camera_id,
                                timestamp=timestamp,
                                class_name=class_name,
                                confidence=confidence,
                                image_path=image_path
                            )
                            
                            self.detection_buffer.append(detection)
                            
                            # Send SMS notification for high-confidence detections
                            if confidence > CONFIDENCE_THRESHOLD:  # Configurable confidence threshold
                                self._send_sms_notification(detection)
                            
                            # Flush buffer if it reaches small batch size
                            if len(self.detection_buffer) >= 3:  # Small batch for Pi
                                self._flush_detections()
                
                except Exception as e:
                    logger.error(f"Camera {self.camera_id}: Inference error: {e}", exc_info=True)
                
                # Small delay to prevent overwhelming the Pi
                time.sleep(0.2)
                
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Worker error: {e}", exc_info=True)
        finally:
            # Cleanup
            cap.release()
            self._flush_detections()
            logger.info(f"Camera {self.camera_id}: Worker stopped")

    def stop(self):
        """Stop the worker"""
        logger.info(f"Camera {self.camera_id}: Stopping worker")
        self.running = False

    def pause(self):
        """Pause the worker"""
        self.paused = True
        logger.info(f"Camera {self.camera_id}: Paused worker")

    def resume(self):
        """Resume the worker"""
        self.paused = False
        logger.info(f"Camera {self.camera_id}: Resumed worker")