import os
import cv2
import time
from datetime import datetime
import threading
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import queue
import json

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

# Import WebSocket manager for real-time updates from dedicated module (will be imported locally when needed)

from ..inference.sms_sender import IprogSMSSender

from ..core.database import get_db_connection

# Load environment variables from .env file
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "pawikansentinel-era7l/2") # Default model ID
DETECTIONS_DIR = os.getenv("DETECTIONS_DIR", "detections")

# iprog configuration for SMS notifications (replaces Semaphore)
IPROG_API_TOKEN = os.getenv("IPROG_API_TOKEN")
IPROG_SENDER_NAME = os.getenv("IPROG_SENDER_NAME", "PawikanSentinel")
SMS_NOTIFICATION_COOLDOWN = int(os.getenv("SMS_NOTIFICATION_COOLDOWN", "10"))  # Default 10 minutes cooldown

# Detection configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))/100.0  # Default 80% confidence threshold
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "5"))  # Default frame skip of 5

# Ensure the detections directory exists
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# --- Logging Setup ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# Initialize iprog SMS sender
sms_sender = IprogSMSSender()

if not ROBOFLOW_API_KEY:
    logger.warning("ROBOFLOW_API_KEY not set in .env. Inference will not work.")

@dataclass
class DetectionResult:
    camera_id: int
    timestamp: str
    class_name: str
    confidence: float
    image_path: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e

    def on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RTSPInferenceWorker(threading.Thread):
    def __init__(self, camera_id: int, rtsp_url: str, model_id: str, api_url: str, api_key: str, 
                 frame_skip: int = FRAME_SKIP, batch_size: int = 10, max_retries: int = 3):
        super().__init__(name=f"CameraWorker-{camera_id}")
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.model_id = model_id
        self.api_url = api_url
        self.api_key = api_key
        self.frame_skip = frame_skip
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialize inference client with circuit breaker
        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key
        )
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        # Threading and state management
        self.running = False
        self.paused = False
        self.detection_buffer: List[DetectionResult] = []
        self.stats = {
            "frames_processed": 0,
            "detections_made": 0,
            "errors": 0,
            "last_error": None,
            "last_success": None
        }
        
        # SMS notification tracking
        self.sms_sender = sms_sender
        
        # Connection management
        self.cap: Optional[cv2.VideoCapture] = None
        self.reconnect_delay = 1  # Start with 1 second delay

    def _setup_capture(self) -> bool:
        """Setup RTSP capture with proper error handling"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                logger.error(f"Camera {self.camera_id}: Cannot open RTSP stream at {self.rtsp_url}")
                return False
                
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"Camera {self.camera_id}: Successfully connected to RTSP stream")
            return True
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error setting up capture: {e}", exc_info=True)
            return False

    def _release_capture(self):
        """Safely release the video capture"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                logger.debug(f"Camera {self.camera_id}: Released video capture")
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error releasing capture: {e}", exc_info=True)

    @contextmanager
    def capture_context(self):
        """Context manager for video capture"""
        if not self._setup_capture():
            raise Exception(f"Failed to setup capture for camera {self.camera_id}")
        try:
            yield self.cap
        finally:
            self._release_capture()

    def _flush_detections(self):
        """Flush detection buffer to database with error handling"""
        if not self.detection_buffer:
            return

        conn = None
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Convert DetectionResult objects to tuples for DB insertion
            db_records = [
                (d.camera_id, d.timestamp, d.class_name, d.confidence, d.image_path)
                for d in self.detection_buffer
            ]
            
            # Get the last detection ID before insertion to identify new records
            c.execute("SELECT MAX(id) FROM detections WHERE camera_id = ?", (self.camera_id,))
            last_id_before = c.fetchone()[0] or 0
            
            # Insert each detection individually to get proper IDs
            for detection in self.detection_buffer:
                c.execute("INSERT INTO detections (camera_id, timestamp, class, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
                          (detection.camera_id, detection.timestamp, detection.class_name, detection.confidence, detection.image_path))
            
            conn.commit()
            logger.info(f"Camera {self.camera_id}: Flushed {len(self.detection_buffer)} detections to DB.")
            

            
            # Send SMS notifications for new detections if Semaphore is configured
            if self.sms_sender.is_enabled():
                self._send_sms_notifications()
            
            # Clear buffer after successful flush
            self.detection_buffer.clear()
            
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error flushing detections to DB: {e}", exc_info=True)
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
        finally:
            if conn:
                conn.close()

    def _send_sms_notifications(self):
        """Send SMS notifications for new detections with cooldown"""
        try:
            # Get camera name
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT name FROM cameras WHERE id = ?", (self.camera_id,))
            camera_result = c.fetchone()
            camera_name = camera_result["name"] if camera_result else f"Camera {self.camera_id}"
            
            # Get contacts
            c.execute("SELECT phone FROM contacts")
            contacts = c.fetchall()
            conn.close()
            
            if not contacts:
                logger.debug("No contacts found for SMS notifications")
                return
            
            # Extract phone numbers from contacts
            phone_numbers = [contact["phone"] for contact in contacts if contact["phone"]]
            
            if not phone_numbers:
                logger.warning("No valid phone numbers found for SMS notifications")
                return
            
            if not self.sms_sender.is_enabled():
                logger.warning("SMS sender not enabled - skipping SMS notifications")
                return
            
            # Create message with detection details
            confidence_percent = self.detection_buffer[-1].confidence * 100
            message_body = f"Pawikan Sentinel Alert: {len(self.detection_buffer)} detections of {self.detection_buffer[-1].class_name} on {camera_name}. Last detection at {self.detection_buffer[-1].timestamp} with {confidence_percent:.1f}% confidence."
            
            # Send SMS notifications using Semaphore
            results = self.sms_sender.send_sms_notification(
                phone_numbers=phone_numbers,
                message_body=message_body
            )
            
            # Log the results
            for phone, success in results.items():
                if success:
                    logger.info(f"SMS sent successfully to {phone}")
                else:
                    logger.error(f"Failed to send SMS to {phone}")
            
        except Exception as e:
            logger.error(f"Error sending SMS notifications: {e}", exc_info=True)

    def _save_annotated_frame(self, frame, detections: List[Dict], timestamp: str) -> Optional[str]:
        """Save annotated frame with bounding boxes"""
        try:
            # Create a copy for annotation
            annotated_frame = frame.copy()
            
            # Draw bounding boxes for each detection
            for pred in detections:
                if "x" in pred and "y" in pred and "width" in pred and "height" in pred:
                    x, y = int(pred["x"]), int(pred["y"])
                    w, h = int(pred["width"]), int(pred["height"])
                    class_name = pred.get("class", "unknown")
                    confidence = pred.get("confidence", 0.0)
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x - w//2, y - h//2 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Generate filename and save
            image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{self.camera_id}.jpg"
            image_path = os.path.join(DETECTIONS_DIR, image_filename)
            
            success = cv2.imwrite(image_path, annotated_frame)
            if success:
                logger.debug(f"Camera {self.camera_id}: Saved annotated image to {image_path}")
                return image_path
            else:
                logger.warning(f"Camera {self.camera_id}: Failed to save annotated image")
                return None
                
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error saving annotated frame: {e}", exc_info=True)
            return None

    def _perform_inference(self, frame) -> Optional[Dict]:
        """Perform inference with circuit breaker and retry logic"""
        def inference_call():
            return self.client.infer(frame, model_id=self.model_id)
        
        try:
            results = self.circuit_breaker.call(inference_call)
            self.stats["last_success"] = datetime.now().isoformat()
            return results
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Inference failed: {e}", exc_info=True)
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return None

    def run(self):
        """Main worker loop with improved error handling and recovery"""
        self.running = True
        frame_count = 0
        
        logger.info(f"Camera {self.camera_id}: Starting inference worker")
        
        while self.running:
            try:
                # Handle pause state
                if self.paused:
                    time.sleep(1)
                    continue
                
                # Setup capture if not already done
                if self.cap is None or not self.cap.isOpened():
                    if not self._setup_capture():
                        logger.warning(f"Camera {self.camera_id}: Failed to connect to RTSP stream, retrying in {self.reconnect_delay}s")
                        time.sleep(self.reconnect_delay)
                        self.reconnect_delay = min(self.reconnect_delay * 2, 30)  # Exponential backoff, max 30s
                        continue
                    else:
                        self.reconnect_delay = 1  # Reset on successful connection
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Camera {self.camera_id}: Failed to grab frame, attempting reconnect...")
                    self._release_capture()
                    time.sleep(self.reconnect_delay)
                    continue
                
                frame_count += 1
                self.stats["frames_processed"] += 1
                
                # Skip frames based on frame_skip setting
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Resize for faster inference
                resized = cv2.resize(frame, (640, 480))
                
                # Perform inference with retry logic
                results = None
                for attempt in range(self.max_retries):
                    results = self._perform_inference(resized)
                    if results is not None:
                        break
                    logger.warning(f"Camera {self.camera_id}: Inference attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
                if results is None:
                    logger.error(f"Camera {self.camera_id}: All inference attempts failed")
                    continue
                
                logger.debug(f"Camera {self.camera_id} Detections: {results}")
                
                # Process detections
                if "predictions" in results:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    for pred in results["predictions"]:
                        class_name = pred["class"]
                        confidence = pred["confidence"]
                        
                        # Check confidence threshold
                        if confidence < CONFIDENCE_THRESHOLD:
                            logger.debug(f"Camera {self.camera_id}: Detection below confidence threshold ({confidence:.2f} < {CONFIDENCE_THRESHOLD})")
                            continue
                        
                        # Save annotated frame
                        image_path = self._save_annotated_frame(resized, [pred], timestamp)
                        
                        if image_path:
                            # Create DetectionResult object
                            detection = DetectionResult(
                                camera_id=self.camera_id,
                                timestamp=timestamp,
                                class_name=class_name,
                                confidence=confidence,
                                image_path=image_path,
                                bbox=(int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"]))
                            )
                            
                            self.detection_buffer.append(detection)
                            self.stats["detections_made"] += 1
                            
                            # Send SMS notification for high-confidence detections
                            if confidence > CONFIDENCE_THRESHOLD:
                                # Flush immediately for high-confidence detections
                                self._flush_detections()
                            else:
                                # Flush buffer if it reaches batch size for low-confidence detections
                                if len(self.detection_buffer) >= self.batch_size:
                                    self._flush_detections()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Unexpected error in worker loop: {e}", exc_info=True)
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
                self._release_capture()
                time.sleep(2)  # Brief pause before retrying
        
        # Cleanup on stop
        self._flush_detections()
        self._release_capture()
        logger.info(f"Camera {self.camera_id}: Inference worker stopped.")

    def stop(self):
        """Stop the worker gracefully"""
        logger.info(f"Camera {self.camera_id}: Stopping inference worker")
        self.running = False

    def pause(self):
        """Pause the worker"""
        self.paused = True
        logger.info(f"Camera {self.camera_id}: Paused inference worker")

    def resume(self):
        """Resume the worker"""
        self.paused = False
        logger.info(f"Camera {self.camera_id}: Resumed inference worker")

    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            "camera_id": self.camera_id,
            "running": self.running,
            "paused": self.paused,
            "buffer_size": len(self.detection_buffer),
            "stats": self.stats.copy(),
            "circuit_breaker_state": self.circuit_breaker.state
        }
