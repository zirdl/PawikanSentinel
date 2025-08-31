"""
The main background service for continuous inference and alerting.
This version is a direct adaptation of scripts/test_rtsp.py's main function.
"""
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from datetime import datetime
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Tuple
from sqlalchemy.orm import Session
import logging
import threading
import queue

from ..core import config, annotate, storage, schemas, postprocess
from ..core.database import SessionLocal, init_db
from ..core.logging_utils import setup_logger
from ..core.throttle import CooldownManager
from ..core.notifier.twilio_provider import TwilioNotifier

# --- Database Writer Thread ---
class DbWriterThread(threading.Thread):
    def __init__(self, detection_queue: queue.Queue, session_local_factory):
        super().__init__()
        self.detection_queue = detection_queue
        self.session_local_factory = session_local_factory
        self.stop_event = threading.Event()
        logger.info("DbWriterThread initialized.")

    def run(self):
        logger.info("DbWriterThread started.")
        while not self.stop_event.is_set() or not self.detection_queue.empty():
            try:
                # Use a timeout to allow checking stop_event periodically
                event_data = self.detection_queue.get(timeout=1)
                db = self.session_local_factory()
                try:
                    storage.create_detection_event(db, event_data)
                    logger.debug("Detection event written to DB.", extra={"detection_count": event_data.detection_count})
                except Exception as e:
                    logger.error(f"Error writing detection event to DB: {e}", exc_info=True)
                    db.rollback()
                finally:
                    db.close()
                self.detection_queue.task_done()
            except queue.Empty:
                continue # No items, check stop_event again
            except Exception as e:
                logger.critical(f"Unhandled error in DbWriterThread: {e}", exc_info=True)
                time.sleep(5) # Prevent rapid error looping
        logger.info("DbWriterThread stopped.")

    def stop(self):
        self.stop_event.set()
        logger.info("DbWriterThread received stop signal.")

# --- Setup ---
static_settings = config.settings
logger = setup_logger(Path(static_settings["logging"]["dir"]), static_settings["logging"]["level"])

class InferenceService:
    def __init__(self, db: Session):
        self.db = db
        self.rtsp_url = static_settings["video"]["rtsp_url"]
        self.model_path = static_settings["model"]["path"]
        self.confidence_threshold = static_settings["model"]["confidence_threshold"]
        self.iou_threshold = static_settings["model"]["nms_iou_threshold"]
        self.max_detections = static_settings["model"]["max_detections"]
        self.gallery_dir = Path(static_settings["gallery"]["dir"])

        # --- TFLite Interpreter Setup ---
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- Cooldowns and Notifications ---
        self.gallery_cooldown_sec = int(static_settings["gallery"]["save_interval_seconds"])
        self.gallery_cooldown = CooldownManager("gallery_save", self.gallery_cooldown_sec)
        
        # Notification cooldown from DB, fallback to config if not in DB
        notification_cooldown_obj = storage.get_setting(db, "notification_cooldown_seconds")
        if notification_cooldown_obj:
            notification_cooldown_sec = int(notification_cooldown_obj.value)
        else:
            notification_cooldown_sec = int(static_settings["notify"]["cooldown_seconds"]) if "cooldown_seconds" in static_settings["notify"] else 1200
            logger.warning(f"Notification cooldown not found in DB, using config value: {notification_cooldown_sec} seconds.")

        self.notification_cooldown = CooldownManager("notification", notification_cooldown_sec)
        
        self.notifier = self._setup_notifier()

        # --- Database Writer Thread Setup ---
        self.detection_queue = queue.Queue()
        self.db_writer_thread = DbWriterThread(self.detection_queue, SessionLocal)
        self.db_writer_thread.start()
        logger.info("Detection event queue and DB writer thread started.")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[tuple]]:
        """
        Processes a single frame for inference, draws bounding boxes, and returns detections.
        Directly adapted from scripts/test_rtsp.py.
        """
        input_shape = self.input_details[0]['shape']
        model_input_height, model_input_width = input_shape[1], input_shape[2]
        
        original_height, original_width = frame.shape[:2]
        resized_img = cv2.resize(frame, (model_input_width, model_input_height))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        if self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (img_rgb / input_scale + input_zero_point).astype(np.int8)
        else:
            input_data = img_rgb.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        inference_start_time = time.time()
        self.interpreter.invoke()
        inference_end_time = time.time()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        if self.output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        detections = postprocess.decode_yolov8_output(
            output_data, 
            (model_input_height, model_input_width),
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections
        )
        
        frame_with_boxes = frame.copy()
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            x1_scaled = int(x1 * original_width / model_input_width)
            y1_scaled = int(y1 * original_height / model_input_height)
            x2_scaled = int(x2 * original_width / model_input_width)
            y2_scaled = int(y2 * original_height / model_input_height)
            
            cv2.rectangle(frame_with_boxes, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            label = f"turtle: {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_with_boxes, (x1_scaled, y1_scaled - label_height - baseline), (x1_scaled + label_width, y1_scaled), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame_with_boxes, label, (x1_scaled, y1_scaled - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        fps = 1 / (inference_end_time - inference_start_time)
        cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame_with_boxes, detections

    def _setup_notifier(self):
        provider_obj = storage.get_setting(self.db, "notify_provider")
        provider = provider_obj.value if provider_obj else "none"

        enabled_obj = storage.get_setting(self.db, "notify_enabled")
        enabled = (enabled_obj.value.lower() == "true") if enabled_obj else False # Default to False for safety

        if not enabled or provider != "twilio":
            logger.info(f"Notifications disabled (provider: '{provider}', enabled: {enabled})")
            return None
        
        twilio_sid_obj = storage.get_setting(self.db, "twilio_account_sid")
        twilio_token_obj = storage.get_setting(self.db, "twilio_auth_token")
        twilio_from_obj = storage.get_setting(self.db, "twilio_from_number")

        if not all([twilio_sid_obj, twilio_token_obj, twilio_from_obj]):
            logger.warning("Twilio notifier is enabled but credentials are not fully set in the database.")
            return None
            
        return TwilioNotifier(
            account_sid=twilio_sid_obj.value, 
            auth_token=twilio_token_obj.value, 
            from_number=twilio_from_obj.value
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), before_sleep=lambda rs: logger.warning("RTSP connection failed. Retrying..."))
    def _connect_rtsp(self):
        logger.info("Connecting to RTSP stream.", extra={"url": self.rtsp_url})
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened(): 
            raise ConnectionError(f"Failed to open RTSP stream at {self.rtsp_url}")
        logger.info("RTSP stream connected.")
        return cap

    def run(self):
        cap = None
        try:
            cap = self._connect_rtsp()

            frame_count = 0
            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Lost RTSP stream. Attempting to reconnect...")
                        cap.release()
                        cap = self._connect_rtsp() # Reconnect
                        continue

                    frame_count += 1
                    
                    processed_frame, detections = self._process_frame(frame)

                    if detections:
                        max_confidence = max(d[4] for d in detections) # d[4] is score
                        logger.info(f"Found {len(detections)} detections.", extra={"max_confidence": f"{max_confidence:.2f}"})

                        # Log to analytics DB (asynchronously)
                        avg_confidence = sum(d[4] for d in detections) / len(detections) # d[4] is score
                        self.detection_queue.put(schemas.DetectionEventCreate(detection_count=len(detections), average_confidence=avg_confidence))
                        logger.debug("Detection event added to queue.")

                        # Save to gallery
                        if self.gallery_cooldown.is_ready():
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Convert detections to a list of dicts for JSON serialization
                            detections_for_json = [
                                {"box": [d[0], d[1], d[2], d[3]], "score": d[4], "class_id": d[5]}
                                for d in detections
                            ]

                            annotate.save_annotated_frame_raw(processed_frame, self.gallery_dir, timestamp, detections_for_json)
                            self.gallery_cooldown.trigger()
                            logger.info("Saved detection frame to gallery.")
                        
                        # Send notification
                        if self.notifier and self.notification_cooldown.is_ready():
                            contacts = storage.get_all_contacts(self.db)
                            recipients = [c.phone_number for c in contacts]
                            if recipients:
                                message = f"Pawikan Sentinel Alert: {len(detections)} turtle(s) detected (max confidence: {max_confidence:.2f})."
                                self.notifier.send(message, recipients)
                                self.notification_cooldown.trigger()
                                logger.info("Notification sent.")
                            else:
                                logger.warning("Detection found, but no active contacts to notify.")
                    else:
                        logger.debug("No detections found in frame.")

                except KeyboardInterrupt:
                    logger.info("Stopping service due to KeyboardInterrupt.")
                    break
                except ConnectionError as e:
                    logger.error(f"RTSP connection error during runtime: {e}. Attempting to reconnect...", exc_info=True)
                    if cap: cap.release()
                    cap = self._connect_rtsp() # Attempt to reconnect
                except Exception as e:
                    logger.critical(f"An unhandled error occurred in the main loop: {e}", exc_info=True)
                    time.sleep(5) # Prevent rapid error looping

        except ConnectionError as e:
            logger.critical(f"Initial RTSP connection failed: {e}. Exiting service.", exc_info=True)
            return # Exit if initial connection fails
        finally:
            if cap:
                cap.release()
            if self.db_writer_thread:
                logger.info("Signaling DbWriterThread to stop and waiting for it to finish...")
                self.db_writer_thread.stop()
                self.db_writer_thread.join()
                logger.info("DbWriterThread stopped.")
            logger.info("Inference service stopped.")

def main():
    init_db()
    db = SessionLocal()
    try:
        service = InferenceService(db)
        service.run()
    except Exception as e:
        logger.critical(f"Inference service failed critically: {e}", exc_info=True)
    finally:
        db.close()
