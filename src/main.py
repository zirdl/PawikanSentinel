import warnings
import time
import cv2
from datetime import datetime
from src.frame_processor.rtsp_client import RTSPClient

warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float32'> type is zero.")
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.")
from src.frame_processor.preprocessing import preprocess_frame
from src.ml_inference.model_loader import ModelLoader
from src.ml_inference.inference import InferenceEngine
import os
from src.detection_analyzer.post_processing import post_process_detections, draw_detections
from src.detection_analyzer.object_tracker import ObjectTracker
from src.alert_manager.alert_generator import AlertGenerator
from src.alert_manager.alert_delivery import AlertDelivery
from src.notification_service.twilio_client import TwilioClient
from src.system_monitor.resource_monitor import ResourceMonitor
from src.system_monitor.logging_manager import LoggingManager
from src.config import ConfigManager

def main():
    # Load Configuration
    config = ConfigManager()

    RTSP_URL = config.get("APP", "RTSP_URL", "rtsp://your_camera_url")
    MODEL_PATH = config.get("APP", "MODEL_PATH", "./model.tflite")
    CONFIDENCE_THRESHOLD = config.get_float("APP", "CONFIDENCE_THRESHOLD", 0.5)
    IOU_THRESHOLD = config.get_float("APP", "IOU_THRESHOLD", 0.5)
    DEDUPLICATION_WINDOW_MINUTES = config.get_int("APP", "DEDUPLICATION_WINDOW_MINUTES", 10)
    LOG_FILE = config.get("APP", "LOG_FILE", "pawikan_sentinel.log")
    SAVE_ANNOTATED_FRAMES = config.get_boolean("APP", "SAVE_ANNOTATED_FRAMES", False)
    ANNOTATED_FRAMES_DIR = config.get("APP", "ANNOTATED_FRAMES_DIR", "/tmp/pawikan_sentinel/annotated_frames")
    DISPLAY_VIDEO_FEED = config.get_boolean("APP", "DISPLAY_VIDEO_FEED", False)

    if SAVE_ANNOTATED_FRAMES and not os.path.exists(ANNOTATED_FRAMES_DIR):
        os.makedirs(ANNOTATED_FRAMES_DIR)

    # Initialize Logging
    log_manager = LoggingManager(LOG_FILE)
    logger = log_manager.get_logger()
    logger.info("Pawikan Sentinel application started.")

    # Initialize Resource Monitor
    resource_monitor = ResourceMonitor()

    # Initialize RTSP Client
    rtsp_client = RTSPClient(RTSP_URL)
    if not rtsp_client.connect():
        logger.error(f"Failed to connect to RTSP stream at {RTSP_URL}. Exiting.")
        return

    # Initialize ML Inference Engine
    model_loader = ModelLoader(MODEL_PATH)
    if not model_loader.load():
        logger.error(f"Failed to load TFLite model from {MODEL_PATH}. Exiting.")
        rtsp_client.release()
        return
    inference_engine = InferenceEngine(model_loader)

    # Initialize Detection Analyzer
    object_tracker = ObjectTracker()

    # Initialize Alert Manager
    notification_service = TwilioClient(config)
    alert_generator = AlertGenerator(DEDUPLICATION_WINDOW_MINUTES)
    alert_delivery = AlertDelivery(notification_service)

    try:
        while True:
            # Read frame
            success, frame = rtsp_client.read_frame()
            if not success:
                logger.warning("Failed to read frame. Attempting to reconnect...")
                if not rtsp_client.reconnect():
                    logger.error("Failed to reconnect to RTSP stream. Exiting.")
                    break
                continue

            # Get original frame dimensions
            original_frame_height, original_frame_width, _ = frame.shape

            # Preprocess frame
            preprocessed_frame = preprocess_frame(frame, (640, 640))

            # Run inference
            raw_detections = inference_engine.run(preprocessed_frame)

            # Post-process detections
            processed_detections = post_process_detections(
                raw_detections, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, original_frame_width, original_frame_height
            )

            # Update object tracker
            # The object tracker expects a list of bounding boxes in [x1, y1, x2, y2] format
            # Need to convert processed_detections to this format
            bboxes_for_tracker = []
            for det in processed_detections:
                bboxes_for_tracker.append(det["box"])

            tracked_objects = object_tracker.update(bboxes_for_tracker)

            # Generate alert
            alert_message = alert_generator.generate_alert(tracked_objects)
            if alert_message:
                alert_delivery.enqueue_alert(alert_message)

            # Process alert queue
            alert_delivery.process_queue()

            # --- Visualization (Optional) ---
            if SAVE_ANNOTATED_FRAMES and processed_detections:
                # Make a copy of the frame to draw on, to avoid modifying the original for other uses
                annotated_frame = frame.copy()
                draw_detections(annotated_frame, processed_detections)
                
                # Use the detection event ID for the filename
                # The alert_generator logs the event ID, we need to capture it or pass it
                # For now, let's use a timestamp if no event ID is directly available here
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(ANNOTATED_FRAMES_DIR, f"detection_{timestamp_str}.jpg")
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"Saved annotated frame to {filename}")

            # Display the frame (optional, for debugging/monitoring on Pi)
            # This part can be commented out or made configurable if not needed for deployment
            if DISPLAY_VIDEO_FEED:
                cv2.imshow("Pawikan Sentinel", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Display closed by user.")
                    break
            # --- End Visualization ---

            # Monitor system resources
            cpu_usage = resource_monitor.get_cpu_usage()
            memory_usage = resource_monitor.get_memory_usage()
            cpu_temp = resource_monitor.get_cpu_temperature()
            logger.info(f"CPU: {cpu_usage}%, Mem: {memory_usage}%, Temp: {cpu_temp}°C")

            

    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
    finally:
        rtsp_client.release()
        if DISPLAY_VIDEO_FEED:
            cv2.destroyAllWindows() # Destroy all OpenCV windows
        logger.info("Pawikan Sentinel application terminated.")

if __name__ == "__main__":
    main()