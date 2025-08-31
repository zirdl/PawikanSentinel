import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import time

def decode_yolov8_output(
    output_data: np.ndarray,
    input_shape: tuple, # (height, width) of the model input (e.g., 640, 640)
    confidence_threshold: float = 0.8,
    iou_threshold: float = 0.45,
    num_classes: int = 1, # Number of classes the model was trained on (1 for turtle)
    max_detections: int = 50 # Maximum number of detections to keep after NMS
) -> list:
    print(f"Raw output_data shape: {output_data.shape}")

    # My model outputs (1, 5, 8400), where 5 is [x, y, w, h, confidence]
    # We need to extract boxes and scores from this.
    predictions = output_data[0] # Remove batch dimension (5, 8400)
    print(f"Predictions shape: {predictions.shape}")

    boxes_xywh = predictions[:4, :].T # Extract xywh coordinates and transpose to (8400, 4)
    class_scores_raw = predictions[4:, :].T # Extract raw class probabilities/logits and transpose to (8400, 1)
    print(f"Class scores raw shape: {class_scores_raw.shape}")
    print(f"Class scores raw (first 5): {class_scores_raw[:5]}")

    # For most exported YOLOv8 TFLite models, the class scores are already probabilities (0-1).
    # If your model's last layer for classification is linear (logits), you might need to apply
    # a sigmoid activation here: `class_scores = 1 / (1 + np.exp(-class_scores_raw))`
    class_scores = class_scores_raw

    # Get the maximum class score for each box and its corresponding class ID.
    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    # Filter by confidence threshold.
    keep_indices = np.where(scores >= confidence_threshold)[0]

    filtered_boxes = boxes_xywh[keep_indices] # Corrected: Use boxes_xywh here
    filtered_scores = scores[keep_indices]
    filtered_class_ids = class_ids[keep_indices]

    # Convert xywh (center_x, center_y, width, height) to x1y1x2y2 (top-left, bottom-right)
    x = filtered_boxes[:, 0]
    y = filtered_boxes[:, 1]
    w = filtered_boxes[:, 2]
    h = filtered_boxes[:, 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # Combine box coordinates.
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)

    # Apply Non-Maximum Suppression (NMS).
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

def run_yolov8_tflite_inference(model_path, image_path, output_image_path=None,
                                confidence_threshold=0.5, iou_threshold=0.45, max_detections=50):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    model_input_height, model_input_width = input_shape[1], input_shape[2]
    
    # Preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_height, original_width = img.shape[:2]
    resized_img = cv2.resize(img, (model_input_width, model_input_height))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Handle quantization for INT8 model
    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (img_rgb / input_scale + input_zero_point).astype(np.int8)
    else:
        input_data = img_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("Running TFLite inference...")
    inference_start_time = time.time()
    interpreter.invoke()
    inference_end_time = time.time()
    print(f"Inference took: {inference_end_time - inference_start_time:.4f} seconds")

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Handle dequantization for INT8 model
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    print("Running post-processing...")
    postprocess_start_time = time.time()
    detections = decode_yolov8_output(
        output_data, 
        (model_input_height, model_input_width),
        confidence_threshold=confidence_threshold, 
        iou_threshold=iou_threshold, 
        max_detections=max_detections
    )
    postprocess_end_time = time.time()
    print(f"Post-processing took: {postprocess_end_time - postprocess_start_time:.4f} seconds")
    
    # Rescale detections to original image size
    final_detections_scaled = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        x1_scaled = int(x1 * original_width / model_input_width)
        y1_scaled = int(y1 * original_height / model_input_height)
        x2_scaled = int(x2 * original_width / model_input_width)
        y2_scaled = int(y2 * original_height / model_input_height)
        final_detections_scaled.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id])

    # Draw bounding boxes and save the image
    img_with_boxes = img.copy()
    for det in final_detections_scaled:
        x1, y1, x2, y2, score, class_id = det
        # Bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label
        label = f"turtle: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Label background
        cv2.rectangle(img_with_boxes, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
        # Label text
        cv2.putText(img_with_boxes, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if output_image_path:
        cv2.imwrite(output_image_path, img_with_boxes)
        print(f"Saved detection image to {output_image_path}")


    return final_detections_scaled

import os
import random

# --- Main execution ---
if __name__ == "__main__":
    model_path = "models/pawikan_int8.tflite"
    image_dir = "/home/gio/Projects/PawikanSentinel/data/processed/images/test/"

    # Get a list of all images in the directory
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"Error: No images found in {image_dir}")
            exit()
    except FileNotFoundError:
        print(f"Error: Directory not found at {image_dir}")
        exit()

    # Pick a random image
    random_image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, random_image_name)
    
    print(f"\n--- Testing on random image: {random_image_name} ---")
    
    detected_objects = run_yolov8_tflite_inference(
        model_path=model_path, 
        image_path=image_path,
        output_image_path='random_test_detected.png',
        confidence_threshold=0.5, # Set to 0.5 as requested
        iou_threshold=0.45,      
        max_detections=50        
    )

    if detected_objects:
        print(f"Detected {len(detected_objects)} objects in {random_image_name}:")
        for det in detected_objects:
            print(f"  Score: {det[4]:.4f}, Box: {det[:4]}")
    else:
        print(f"No objects detected in {random_image_name} with a confidence of 0.5 or higher.")

