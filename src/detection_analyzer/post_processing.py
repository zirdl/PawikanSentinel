import numpy as np
import tensorflow as tf

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Applies non-maximum suppression to filter overlapping bounding boxes.

    Args:
        boxes (np.ndarray): An array of bounding boxes, shape (N, 4).
        scores (np.ndarray): An array of confidence scores, shape (N,).
        iou_threshold (float): The Intersection over Union (IoU) threshold for suppression.

    Returns:
        np.ndarray: The indices of the bounding boxes to keep.
    """
    return tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size=100,  # Adjust as needed
        iou_threshold=iou_threshold
    ).numpy()

def post_process_detections(raw_output: dict, confidence_threshold: float, iou_threshold: float) -> list:
    """
    Post-processes the raw output from the inference engine.

    Args:
        raw_output (dict): The raw output from the inference engine.
        confidence_threshold (float): The confidence threshold for filtering detections.
        iou_threshold (float): The IoU threshold for non-maximum suppression.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected object.
    """
    # Assuming the output tensor name is known
    # This might need to be adjusted based on the actual model output
    output_tensor = next(iter(raw_output.values()))

    # Squeeze the batch dimension
    detections = np.squeeze(output_tensor)

    # Filter out low-confidence detections
    scores = detections[:, 4]
    confident_detections = detections[scores >= confidence_threshold]

    if len(confident_detections) == 0:
        return []

    # Extract boxes and scores
    boxes = confident_detections[:, :4]  # [x, y, w, h]
    scores = confident_detections[:, 4]

    # Convert boxes from [center_x, center_y, width, height] to [y1, x1, y2, x2]
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    y1 = y - h / 2
    x1 = x - w / 2
    y2 = y + h / 2
    x2 = x + w / 2
    boxes_for_nms = np.stack([y1, x1, y2, x2], axis=1)

    # Apply non-maximum suppression
    selected_indices = non_max_suppression(boxes_for_nms, scores, iou_threshold)

    # Create a list of detected objects
    detected_objects = []
    for index in selected_indices:
        detection = confident_detections[index]
        detected_objects.append({
            "box": detection[:4].tolist(),
            "confidence": detection[4],
            "class_id": np.argmax(detection[5:])
        })

    return detected_objects

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Create dummy raw output from the model
    #    This simulates the output of the InferenceEngine
    dummy_output = {
        'output_1': np.random.rand(1, 10, 85).astype(np.float32)
    }

    # 2. Set confidence and IoU thresholds
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    # 3. Run post-processing
    detected_objects = post_process_detections(
        dummy_output, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
    )

    # 4. Print the results
    print(f"Detected {len(detected_objects)} objects:")
    for obj in detected_objects:
        print(f"  - Box: {obj['box']}, Confidence: {obj['confidence']:.2f}, Class ID: {obj['class_id']}")
