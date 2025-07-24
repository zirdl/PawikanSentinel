import numpy as np
import tensorflow as tf
import cv2

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

def draw_detections(frame, detections, class_names=None):
    """
    Draws bounding boxes and labels on the frame.

    Args:
        frame (np.ndarray): The image frame to draw on.
        detections (list): A list of dictionaries, where each dictionary represents a detected object.
                           Each dict should have 'box', 'confidence', and 'class_id'.
        class_names (list, optional): A list of class names corresponding to class_id. Defaults to None.
    """
    if class_names is None:
        class_names = {0: "turtle"} # Default to turtle if no class names provided

    for det in detections:
        x1, y1, x2, y2 = [int(b) for b in det['box']]
        confidence = det['confidence']
        class_id = det['class_id']
        
        # Get class name, default to "Class X" if not found
        class_name = class_names.get(class_id, f"Class {class_id}")

        color = (0, 255, 0)  # Green color for bounding box
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        label = f"{class_name}: {confidence:.2f}"

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Draw background rectangle for label
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)

        # Put label text
        cv2.putText(frame, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

def post_process_detections(raw_output: dict, confidence_threshold: float, iou_threshold: float, original_frame_width: int, original_frame_height: int) -> list:
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

    # Convert boxes from [center_x, center_y, width, height] to [x1, y1, x2, y2] (absolute pixels)
    x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Denormalize coordinates
    x1 = (x_center - w / 2) * original_frame_width
    y1 = (y_center - h / 2) * original_frame_height
    x2 = (x_center + w / 2) * original_frame_width
    y2 = (y_center + h / 2) * original_frame_height

    # Ensure coordinates are within frame boundaries
    x1 = np.maximum(0, x1)
    y1 = np.maximum(0, y1)
    x2 = np.minimum(original_frame_width, x2)
    y2 = np.minimum(original_frame_height, y2)

    boxes_for_nms = np.stack([y1, x1, y2, x2], axis=1)

    # Apply non-maximum suppression
    selected_indices = non_max_suppression(boxes_for_nms, scores, iou_threshold)

    # Create a list of detected objects
    detected_objects = []
    for index in selected_indices:
        # Use the denormalized [x1, y1, x2, y2] for the output
        box = [x1[index], y1[index], x2[index], y2[index]]
        confidence = confident_detections[index, 4]
        class_id = np.argmax(confident_detections[index, 5:]) # Assuming class probabilities start from index 5

        detected_objects.append({
            "box": box,
            "confidence": confidence,
            "class_id": class_id
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
        dummy_output, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, 640, 480
    )

    # 4. Print the results
    print(f"Detected {len(detected_objects)} objects:")
    for obj in detected_objects:
        print(f"  - Box: {obj['box']}, Confidence: {obj['confidence']:.2f}, Class ID: {obj['class_id']}")
