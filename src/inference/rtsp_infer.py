import cv2
import time
import os
from inference_sdk import InferenceHTTPClient

# Local inference server (running in Docker)
client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # change if running remotely
    api_key=os.environ.get("ROBOFLOW_API_KEY")  # optional if model is public
)

# Your model ID (from Roboflow dashboard, e.g., "turtledetector/vichai/3")
MODEL_ID = "pawikansentinel-era7l/2"

# RTSP stream (replace with your camera details)
RTSP_URL = "rtsp://localhost:8554/test"

# Open RTSP stream
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open RTSP stream")

print("✅ Connected to RTSP stream. Press Ctrl+C to stop.")

frame_skip = 5  # process 1 out of every 5 frames

try:
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame, retrying...")
            time.sleep(1)
            continue

        count += 1
        if count % frame_skip != 0:
            continue

        # Resize for faster inference
        resized = cv2.resize(frame, (640, 480))

        # Run inference
        results = client.infer(resized, model_id=MODEL_ID)
        print("Detections:", results)

        # OPTIONAL: draw bounding boxes and save annotated frame
        if "predictions" in results:
            for pred in results["predictions"]:
                x, y = int(pred["x"]), int(pred["y"])
                w, h = int(pred["width"]), int(pred["height"])
                label = pred["class"]
                conf = pred["confidence"]

                # Draw box
                cv2.rectangle(resized, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                cv2.putText(resized, f"{label} {conf:.2f}",
                            (x - w//2, y - h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save annotated frame (optional)
            cv2.imwrite("last_detection.jpg", resized)

        # Sleep to avoid hammering the server
        time.sleep(0.5)

except KeyboardInterrupt:
    print("🛑 Stopping...")
finally:
    cap.release()

