"""
Standalone RTSP inference script for quick testing.
Uses local YOLO11 model (no Docker/Roboflow needed).

Usage:
    python rtsp_infer.py --rtsp-url rtsp://your-camera/stream [--input-size 320]
"""

import cv2
import time
import os
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from yolo_detector import load_model, download_model, YOLOModel

def main():
    parser = argparse.ArgumentParser(description="RTSP inference with local YOLO11 model")
    parser.add_argument("--rtsp-url", default="rtsp://localhost:8554/test", help="RTSP stream URL")
    parser.add_argument("--input-size", type=int, default=320, help="Model input size (default: 320)")
    parser.add_argument("--model-dir", default="models", help="Directory for model files")
    args = parser.parse_args()

    # Load model
    model_dir = Path(args.model_dir)
    model_path = model_dir / "turtle_detector.pt"
    if not model_path.exists():
        print("Model not found locally, downloading from HuggingFace...")
        model_path = download_model(args.model_dir)

    model = load_model(str(model_path))
    print(f"Model loaded: {model.class_names}")

    # Open RTSP stream
    cap = cv2.VideoCapture(args.rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RTSP stream: {args.rtsp_url}")

    print(f"Connected to RTSP stream. Press Ctrl+C to stop.")

    frame_skip = 5
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                time.sleep(1)
                continue

            count += 1
            if count % frame_skip != 0:
                continue

            # Resize for faster inference
            resized = cv2.resize(frame, (args.input_size, args.input_size))

            # Run inference
            results = model.infer(resized)

            # Draw bounding boxes
            if "predictions" in results and results["predictions"]:
                for pred in results["predictions"]:
                    x, y = int(pred["x"]), int(pred["y"])
                    w, h = int(pred["width"]), int(pred["height"])
                    label = pred["class"]
                    conf = pred["confidence"]

                    # Scale coordinates back to original frame size
                    orig_h, orig_w = frame.shape[:2]
                    scale_x = orig_w / args.input_size
                    scale_y = orig_h / args.input_size

                    x_orig = int(x * scale_x)
                    y_orig = int(y * scale_y)
                    w_orig = int(w * scale_x)
                    h_orig = int(h * scale_y)

                    # Draw box on original frame
                    cv2.rectangle(frame, (x_orig - w_orig//2, y_orig - h_orig//2),
                                  (x_orig + w_orig//2, y_orig + h_orig//2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x_orig - w_orig//2, y_orig - h_orig//2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save annotated frame
                cv2.imwrite("last_detection.jpg", frame)
                print(f"Saved annotated frame: last_detection.jpg")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
