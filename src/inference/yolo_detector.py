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

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Only keep turtle body detections; skip flipper/keypoint classes
                if class_name != TARGET_CLASS:
                    continue

                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)

                predictions.append({
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                    "class": class_name,
                    "confidence": float(conf),
                })

        return {"predictions": predictions}
