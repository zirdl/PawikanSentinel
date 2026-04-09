"""
YOLO11 local inference module for sea turtle detection.
Replaces the Roboflow HTTP client with direct local inference
using the BVRA/TurtleDetector model from HuggingFace.

This module handles:
- Auto-downloading the model from HuggingFace on first run
- Loading the model with optimized settings for CPU/Raspberry Pi
- Running inference on numpy arrays (OpenCV frames)
- Returning predictions in the same format as the Roboflow API
  for drop-in compatibility with existing worker code
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

# Model configuration
MODEL_REPO = os.getenv("YOLO_MODEL_REPO", "BVRA/TurtleDetector")
MODEL_FILENAME = os.getenv("YOLO_MODEL_FILENAME", "turtle_detector.pt")
MODEL_DIR = os.getenv("YOLO_MODEL_DIR", "models")
INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "320"))  # Default 320x320 for Pi
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))  # Ultralytics default

# Only track turtle body detections; ignore flipper/head keypoints
TARGET_CLASS = "turtle"


def download_model(model_dir: Optional[str] = None) -> Path:
    """Download the TurtleDetector model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model download. "
            "Install with: pip install huggingface_hub"
        )

    dir_path = Path(model_dir or MODEL_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {MODEL_REPO}/{MODEL_FILENAME} from HuggingFace...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(dir_path)
    )

    model_path = Path(model_path)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model downloaded to {model_path} ({size_mb:.1f} MB)")
    return model_path


def load_model(model_path: Optional[str] = None, device: str = "cpu") -> "YOLOModel":
    """Load the YOLO11 turtle detection model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for inference. "
            "Install with: pip install ultralytics"
        )

    if model_path is None:
        model_dir = Path(MODEL_DIR)
        model_path = model_dir / MODEL_FILENAME
        if not model_path.exists():
            model_path = download_model(MODEL_DIR)
        model_path = str(model_path)

    logger.info(f"Loading YOLO11 model from {model_path} on {device}...")
    yolo = YOLO(model_path)

    # Optimize for CPU inference
    # fuse() merges Conv and BatchNorm layers for faster inference
    yolo.fuse()

    model = YOLOModel(yolo)
    logger.info(f"Model loaded successfully. Classes: {model.class_names}")
    return model


class YOLOModel:
    """
    Wrapper around ultralytics.YOLO that provides the same prediction
    interface as the Roboflow InferenceHTTPClient for drop-in compatibility.
    """

    def __init__(self, yolo_model):
        self.yolo = yolo_model
        # Get class names from model
        self.class_names = yolo_model.names

    def infer(
        self,
        image: np.ndarray,
        model_id: Optional[str] = None,  # Ignored, for API compatibility
        confidence: float = CONFIDENCE_THRESHOLD,
        **kwargs
    ) -> Dict:
        """
        Run inference on a single image (numpy array from OpenCV).

        Returns predictions in Roboflow-compatible format:
        {
            "predictions": [
                {
                    "x": int,       # center x
                    "y": int,       # center y
                    "width": int,
                    "height": int,
                    "class": str,
                    "confidence": float
                },
                ...
            ]
        }
        """
        # Convert BGR (OpenCV) to RGB for ultralytics
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image

        # Run inference
        # imgsz controls input resolution - smaller = faster
        results = self.yolo(
            image_rgb,
            imgsz=INPUT_SIZE,
            conf=confidence,
            verbose=False,  # Suppress ultralytics console output
            device="cpu",
            save=False,
        )

        predictions = []
        result = results[0]  # Single image result

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # 1. Collect all detected parts (heads, flippers, turtles)
            raw_boxes = []
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                raw_boxes.append({
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "conf": float(conf)
                })

            # 2. Merge overlapping boxes (treating all parts as one turtle)
            # We use a pixel margin to group parts that are close but not perfectly touching
            def check_overlap(b1, b2, margin=30):
                return not (
                    b1["x2"] + margin < b2["x1"] - margin or
                    b1["x1"] - margin > b2["x2"] + margin or
                    b1["y2"] + margin < b2["y1"] - margin or
                    b1["y1"] - margin > b2["y2"] + margin
                )

            merged_boxes = []
            while raw_boxes:
                current = raw_boxes.pop(0)
                still_merging = True
                while still_merging:
                    still_merging = False
                    for i in range(len(raw_boxes) - 1, -1, -1):
                        if check_overlap(current, raw_boxes[i]):
                            overlap = raw_boxes.pop(i)
                            current["x1"] = min(current["x1"], overlap["x1"])
                            current["y1"] = min(current["y1"], overlap["y1"])
                            current["x2"] = max(current["x2"], overlap["x2"])
                            current["y2"] = max(current["y2"], overlap["y2"])
                            # Take highest confidence
                            current["conf"] = max(current["conf"], overlap["conf"])
                            still_merging = True
                merged_boxes.append(current)

            # 3. Format merged bounding boxes into the API schema as full "turtles"
            for b in merged_boxes:
                width = int(b["x2"] - b["x1"])
                height = int(b["y2"] - b["y1"])
                center_x = int((b["x1"] + b["x2"]) / 2)
                center_y = int((b["y1"] + b["y2"]) / 2)

                predictions.append({
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                    "class": "turtle",  # Force classification as a whole turtle
                    "confidence": float(b["conf"]),
                })

        return {"predictions": predictions}
