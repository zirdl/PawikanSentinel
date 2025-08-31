"""
Functions for annotating images with detection results.
"""
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def draw_bounding_box(
    image: np.ndarray,
    box: List[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draws a single bounding box and label on an image."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        image,
        (x1, y1 - label_height - baseline),
        (x1 + label_width, y1),
        color,
        -1,
    )
    # Draw label text
    cv2.putText(
        image,
        label,
        (x1, y1 - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

def save_annotated_frame_raw(
    frame: np.ndarray,
    gallery_dir: Path,
    timestamp: str,
    detections: List[dict], # For metadata
    model_version: str = "unknown",
) -> None:
    """
    Saves an already annotated frame and a sidecar JSON file with detection metadata.
    This is used when the frame has been annotated externally (e.g., by _process_frame).

    Args:
        frame: The already annotated image frame to save.
        gallery_dir: The directory to save the gallery files.
        timestamp: The timestamp string for the filename.
        detections: A list of detection dictionaries (for metadata).
        model_version: The version of the model used for detection.
    """
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_filename = f"{timestamp}.jpg"
    image_path = gallery_dir / image_filename
    cv2.imwrite(str(image_path), frame) # Save the provided frame directly

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_version": model_version,
        "detections": detections,
    }
    json_filename = f"{timestamp}.json"
    json_path = gallery_dir / json_filename
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

def save_annotated_frame(
    frame: np.ndarray,
    detections: List[dict],
    gallery_dir: Path,
    timestamp: str,
    model_input_shape: Tuple[int, int],
    model_version: str = "unknown",
) -> None:
    """
    Saves an annotated frame and a sidecar JSON file with detection metadata.

    Args:
        frame: The image frame to save.
        detections: A list of detection dictionaries.
        gallery_dir: The directory to save the gallery files.
        timestamp: The timestamp string for the filename.
        model_version: The version of the model used for detection.
    """
    gallery_dir.mkdir(parents=True, exist_ok=True)
    
    annotated_frame = frame.copy()
    original_height, original_width = frame.shape[:2]
    model_input_height, model_input_width = model_input_shape

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        
        x1_scaled = int(x1 * original_width / model_input_width)
        y1_scaled = int(y1 * original_height / model_input_height)
        x2_scaled = int(x2 * original_width / model_input_width)
        y2_scaled = int(y2 * original_height / model_input_height)
        
        scaled_box = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
        label = f'Turtle {det["score"]:.2f}'
        draw_bounding_box(annotated_frame, scaled_box, label)

    # Save image
    image_filename = f"{timestamp}.jpg"
    image_path = gallery_dir / image_filename
    cv2.imwrite(str(image_path), annotated_frame)

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_version": model_version,
        "detections": detections,
    }
    json_filename = f"{timestamp}.json"
    json_path = gallery_dir / json_filename
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
