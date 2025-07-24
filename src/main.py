"""
Main Application for Pawikan Sentinel
"""

import time
from frame_processor import FrameProcessor
from ml_inference import MLInferenceEngine
from detection_analyzer import DetectionAnalyzer
from alert_manager import AlertManager
from notification_service import NotificationService
from system_monitor import SystemMonitor

def main():
    """
    Main application loop.
    """
    # Configuration
    rtsp_url = "rtsp://your_camera_url"
    model_path = "path/to/your/model.tflite"
    twilio_sid = "your_twilio_sid"
    twilio_token = "your_twilio_token"
    twilio_phone_number = "your_twilio_phone_number"
    recipient_phone_number = "recipient_phone_number"

    # Initialization
    frame_processor = FrameProcessor(rtsp_url=rtsp_url)
    ml_inference = MLInferenceEngine(model_path=model_path)
    detection_analyzer = DetectionAnalyzer()
    alert_manager = AlertManager()
    notification_service = NotificationService(
        sid=twilio_sid,
        token=twilio_token,
        phone_number=twilio_phone_number
    )
    system_monitor = SystemMonitor()

    # Load ML model
    try:
        ml_inference.load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Start frame processor
    frame_processor.start()

    try:
        while True:
            # System monitoring
            cpu_usage = system_monitor.get_cpu_usage()
            mem_usage = system_monitor.get_memory_usage()
            temp = system_monitor.get_temperature()
            print(f"CPU: {cpu_usage}%, Memory: {mem_usage['percent']}%, Temp: {temp}°C")

            # Get frame
            frame = frame_processor.get_frame()
            if frame is None:
                time.sleep(1)
                continue

            # Run inference
            detections = ml_inference.run_inference(frame)

            # Analyze detections
            analyzed_detections = detection_analyzer.analyze(detections)

            # Generate alerts
            alert_manager.generate_alert(analyzed_detections)

            # Optional: Add a delay to control the loop speed
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        frame_processor.stop()

if __name__ == "__main__":
    main()

