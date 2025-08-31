import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import argparse
import os

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

def process_frame(frame, interpreter, input_details, output_details, confidence_threshold, iou_threshold):
    """Processes a single frame for inference."""
    input_shape = input_details[0]['shape']
    model_input_height, model_input_width = input_shape[1], input_shape[2]
    
    original_height, original_width = frame.shape[:2]
    resized_img = cv2.resize(frame, (model_input_width, model_input_height))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    if input_details[0]['dtype'] == np.int8:
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data = (img_rgb / input_scale + input_zero_point).astype(np.int8)
    else:
        input_data = img_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    inference_start_time = time.time()
    interpreter.invoke()
    inference_end_time = time.time()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    detections = decode_yolov8_output(
        output_data, 
        (model_input_height, model_input_width),
        confidence_threshold=confidence_threshold, 
        iou_threshold=iou_threshold
    )
    
    frame_with_boxes = frame.copy()
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        x1_scaled = int(x1 * original_width / model_input_width)
        y1_scaled = int(y1 * original_height / model_input_height)
        x2_scaled = int(x2 * original_width / model_input_width)
        y2_scaled = int(y2 * original_height / model_input_height)
        
        cv2.rectangle(frame_with_boxes, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
        label = f"turtle: {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_with_boxes, (x1_scaled, y1_scaled - label_height - baseline), (x1_scaled + label_width, y1_scaled), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame_with_boxes, label, (x1_scaled, y1_scaled - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    fps = 1 / (inference_end_time - inference_start_time)
    cv2.putText(frame_with_boxes, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame_with_boxes, detections

def main():
    parser = argparse.ArgumentParser(description="Run TFLite inference on an RTSP stream and save detections.")
    parser.add_argument("--rtsp-url", type=str, required=True, help="URL of the RTSP stream.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save detection images.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument("--model-path", type=str, default="pawikan_int8.tflite", help="Path to the TFLite model file.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    interpreter = tflite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(args.rtsp_url)

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {args.rtsp_url}")
        return

    print("Successfully opened RTSP stream. Processing frames... Press Ctrl+C to stop.")

    frame_count = 0
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or could not read frame. Retrying...")
                time.sleep(2)
                cap.open(args.rtsp_url)
                continue

            frame_count += 1

            processed_frame, detections = process_frame(
                frame, interpreter, input_details, output_details,
                args.confidence, args.iou_threshold
            )

            if detections:
                timestamp = int(time.time())
                filename = f"detection_{timestamp}_{frame_count}.jpg"
                output_path = os.path.join(args.output_dir, filename)

                cv2.imwrite(output_path, processed_frame)
                print(f"Saved detection image to {output_path} ({len(detections)} objects)")

        except KeyboardInterrupt:
            print("Stopping script.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)

    cap.release()

if __name__ == "__main__":
    main()
