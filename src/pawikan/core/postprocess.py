import cv2
import numpy as np
from typing import List, Tuple

def decode_yolov8_output(
    output_data: np.ndarray,
    input_shape: Tuple[int, int], # (height, width) of the model input (e.g., 640, 640)
    confidence_threshold: float = 0.8,
    iou_threshold: float = 0.45,
    max_detections: int = 50 # Maximum number of detections to keep after NMS
) -> List[Tuple]:
    predictions = output_data[0] # Remove batch dimension (5, 8400)

    boxes_xywh = predictions[:4, :].T # Extract xywh coordinates and transpose to (8400, 4)
    class_scores_raw = predictions[4:, :].T # Extract raw class probabilities/logits and transpose to (8400, 1)
    
    class_scores = class_scores_raw # Assuming scores are already probabilities (0-1)

    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    keep_indices = np.where(scores >= confidence_threshold)[0]

    filtered_boxes = boxes_xywh[keep_indices]
    filtered_scores = scores[keep_indices]
    filtered_class_ids = class_ids[keep_indices]

    x = filtered_boxes[:, 0]
    y = filtered_boxes[:, 1]
    w = filtered_boxes[:, 2]
    h = filtered_boxes[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

    input_height, input_width = input_shape
    scaled_boxes = boxes_xyxy * np.array([input_width, input_height, input_width, input_height], dtype=np.float32)

    boxes_for_nms = scaled_boxes.tolist()
    scores_for_nms = filtered_scores.tolist()

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms,
        scores_for_nms,
        confidence_threshold,
        iou_threshold
    )

    if len(indices) == 0:
        return []

    if isinstance(indices, tuple):
        indices = indices[0]
    if indices.ndim == 2:
        indices = indices.flatten()

    final_detections = []
    for i in indices:
        box = scaled_boxes[i]
        score = filtered_scores[i]
        class_id = filtered_class_ids[i]
        final_detections.append((int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score), int(class_id)))

    final_detections.sort(key=lambda x: x[4], reverse=True)
    return final_detections[:max_detections]

def postprocess_output(
    output_data: np.ndarray,
    original_shape: Tuple[int, int], # (height, width)
    input_shape: Tuple[int, int],    # (height, width)
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> List[dict]:
    detections_raw = decode_yolov8_output(
        output_data,
        input_shape,
        confidence_threshold,
        iou_threshold,
        max_detections
    )

    original_height, original_width = original_shape
    model_input_height, model_input_width = input_shape

    final_detections = []
    for det in detections_raw:
        x1, y1, x2, y2, score, class_id = det
        x1_scaled = int(x1 * original_width / model_input_width)
        y1_scaled = int(y1 * original_height / model_input_height)
        x2_scaled = int(x2 * original_width / model_input_width)
        y2_scaled = int(y2 * original_height / model_input_height)
        final_detections.append({
            "box": [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
            "score": float(score),
            "class_id": int(class_id),
        })
    return final_detections
