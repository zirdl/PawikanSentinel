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

class StreamProcessor(threading.Thread):
    def __init__(self, rtsp_url: str, interpreter, input_details, output_details, detection_queue: queue.Queue, gallery_cooldown: CooldownManager, notification_cooldown: CooldownManager, notifier, db_session_factory, static_settings):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.detection_queue = detection_queue
        self.gallery_cooldown = gallery_cooldown
        self.notification_cooldown = notification_cooldown
        self.notifier = notifier
        self.db_session_factory = db_session_factory
        self.static_settings = static_settings
        self.stop_event = threading.Event()
        self.model_path = self.static_settings["model"]["path"]
        self.confidence_threshold = self.static_settings["model"]["confidence_threshold"]
        self.iou_threshold = self.static_settings["model"]["nms_iou_threshold"]
        self.max_detections = self.static_settings["model"]["max_detections"]
        self.gallery_dir = Path(self.static_settings["gallery"]["dir"])
        logger.info(f"StreamProcessor for {self.rtsp_url} initialized.")

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[tuple]]:
        """
        Processes a single frame for inference, draws bounding boxes, and returns detections.
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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30), before_sleep=lambda rs: logger.warning(f"RTSP connection to {self.rtsp_url} failed. Retrying..."))
    def _connect_rtsp(self):
        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened(): 
            raise ConnectionError(f"Failed to open RTSP stream at {self.rtsp_url}")
        logger.info(f"RTSP stream connected: {self.rtsp_url}")
        return cap

    def run(self):
        cap = None
        try:
            cap = self._connect_rtsp()
            db = self.db_session_factory()

            while not self.stop_event.is_set():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Lost RTSP stream from {self.rtsp_url}. Attempting to reconnect...")
                        cap.release()
                        cap = self._connect_rtsp()
                        continue

                    processed_frame, detections = self._process_frame(frame)

                    if detections:
                        max_confidence = max(d[4] for d in detections)
                        logger.info(f"Found {len(detections)} detections in stream {self.rtsp_url}.", extra={"max_confidence": f"{max_confidence:.2f}"})

                        avg_confidence = sum(d[4] for d in detections) / len(detections)
                        self.detection_queue.put(schemas.DetectionEventCreate(detection_count=len(detections), average_confidence=avg_confidence))
                        logger.debug("Detection event added to queue.")

                        if self.gallery_cooldown.is_ready():
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            detections_for_json = [{"box": [d[0], d[1], d[2], d[3]], "score": d[4], "class_id": d[5]} for d in detections]
                            annotate.save_annotated_frame_raw(processed_frame, self.gallery_dir, timestamp, detections_for_json)
                            self.gallery_cooldown.trigger()
                            logger.info(f"Saved detection frame to gallery from stream {self.rtsp_url}.")
                        
                        if self.notifier and self.notification_cooldown.is_ready():
                            contacts = storage.get_all_contacts(db)
                            recipients = [c.phone_number for c in contacts]
                            if recipients:
                                message = f"Pawikan Sentinel Alert: {len(detections)} turtle(s) detected in stream {self.rtsp_url} (max confidence: {max_confidence:.2f})."
                                self.notifier.send(message, recipients)
                                self.notification_cooldown.trigger()
                                logger.info("Notification sent.")
                            else:
                                logger.warning("Detection found, but no active contacts to notify.")
                    else:
                        logger.debug(f"No detections found in frame from {self.rtsp_url}.")

                except ConnectionError as e:
                    logger.error(f"RTSP connection error during runtime for {self.rtsp_url}: {e}. Attempting to reconnect...", exc_info=True)
                    if cap: cap.release()
                    cap = self._connect_rtsp()
                except Exception as e:
                    logger.critical(f"An unhandled error occurred in the loop for {self.rtsp_url}: {e}", exc_info=True)
                    time.sleep(5)
        
        except ConnectionError as e:
            logger.critical(f"Initial RTSP connection failed for {self.rtsp_url}: {e}. Exiting thread.", exc_info=True)
        finally:
            if cap:
                cap.release()
            if db:
                db.close()
            logger.info(f"StreamProcessor for {self.rtsp_url} stopped.")

    def stop(self):
        self.stop_event.set()
        logger.info(f"StreamProcessor for {self.rtsp_url} received stop signal.")

class InferenceService:
    def __init__(self, db: Session):
        self.db = db
        self.rtsp_urls = static_settings["video"]["rtsp_urls"]
        self.model_path = static_settings["model"]["path"]
        
        # --- TFLite Interpreter Setup ---
        # This should be thread-safe if each thread has its own interpreter.
        # For this implementation, we create one per stream thread.
        
        # --- Cooldowns and Notifications ---
        self.gallery_cooldown_sec = int(static_settings["gallery"]["save_interval_seconds"])
        self.gallery_cooldown = CooldownManager("gallery_save", self.gallery_cooldown_sec)
        
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
        self.stream_processors = []

    def _setup_notifier(self):
        provider_obj = storage.get_setting(self.db, "notify_provider")
        provider = provider_obj.value if provider_obj else "none"

        enabled_obj = storage.get_setting(self.db, "notify_enabled")
        enabled = (enabled_obj.value.lower() == "true") if enabled_obj else False

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

    def run(self):
        logger.info("Starting inference service with multiple streams.")
        self.db_writer_thread.start()

        for url in self.rtsp_urls:
            # Each thread needs its own interpreter instance
            interpreter = tflite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            processor = StreamProcessor(
                rtsp_url=url,
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                detection_queue=self.detection_queue,
                gallery_cooldown=self.gallery_cooldown,
                notification_cooldown=self.notification_cooldown,
                notifier=self.notifier,
                db_session_factory=SessionLocal,
                static_settings=static_settings
            )
            self.stream_processors.append(processor)
            processor.start()

        try:
            while True:
                # Keep the main thread alive, or handle other service-wide tasks
                time.sleep(10)
                # Optional: Check if threads are alive and restart if needed
                for processor in self.stream_processors:
                    if not processor.is_alive():
                        logger.error(f"Stream processor for {processor.rtsp_url} has died. It will not be automatically restarted in this version.")
                        # In a more robust implementation, you might want to restart it.

        except KeyboardInterrupt:
            logger.info("Stopping service due to KeyboardInterrupt.")
        finally:
            self.stop()

    def stop(self):
        logger.info("Stopping all stream processors...")
        for processor in self.stream_processors:
            processor.stop()
        for processor in self.stream_processors:
            processor.join()
        logger.info("All stream processors stopped.")

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
