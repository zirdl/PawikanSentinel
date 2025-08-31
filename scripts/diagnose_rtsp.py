
import numpy as np
import cv2
import ai_edge_litert.interpreter as tflite
import time
import os

def decode_yolov8_output(
    output_data: np.ndarray,
    input_shape: tuple, # (height, width) of the model input (e.g., 640, 640)
    confidence_threshold: float = 0.8,
    iou_threshold: float = 0.45,
    num_classes: int = 1, # Number of classes the model was trained on (1 for turtle)
    max_detections: int = 50 # Maximum number of detections to keep after NMS
) -> list:
    # print(f"Raw output_data shape: {output_data.shape}")

    # My model outputs (1, 5, 8400), where 5 is [x, y, w, h, confidence]
    # We need to extract boxes and scores from this.
    predictions = output_data[0] # Remove batch dimension (5, 8400)
    # print(f"Predictions shape: {predictions.shape}")

    boxes_xywh = predictions[:4, :].T # Extract xywh coordinates and transpose to (8400, 4)
    class_scores_raw = predictions[4:, :].T # Extract raw class probabilities/logits and transpose to (8400, 1)
    # print(f"Class scores raw shape: {class_scores_raw.shape}")
    # print(f"Class scores raw (first 5): {class_scores_raw[:5]}")

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

def run_rtsp_inference(model_path, rtsp_url, gallery_dir, confidence_threshold=0.5, iou_threshold=0.45, max_detections=50):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    model_input_height, model_input_width = input_shape[1], input_shape[2]

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return

    print(f"Successfully connected to RTSP stream: {rtsp_url}")

    os.makedirs(gallery_dir, exist_ok=True)
    print(f"Gallery directory ensured: {gallery_dir}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, attempting to reconnect...")
                cap.release()
                time.sleep(5) # Wait before trying to reconnect
                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    print("Reconnection failed. Exiting.")
                    break
                continue

            original_height, original_width = frame.shape[:2]
            resized_frame = cv2.resize(frame, (model_input_width, model_input_height))
            img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Handle quantization for INT8 model
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
            inference_time = inference_end_time - inference_start_time

            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Handle dequantization for INT8 model
            if output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

            postprocess_start_time = time.time()
            detections = decode_yolov8_output(
                output_data,
                (model_input_height, model_input_width),
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections
            )
            postprocess_end_time = time.time()
            postprocess_time = postprocess_end_time - postprocess_start_time

            # Rescale detections to original frame size
            final_detections_scaled = []
            for det in detections:
                x1, y1, x2, y2, score, class_id = det
                x1_scaled = int(x1 * original_width / model_input_width)
                y1_scaled = int(y1 * original_height / model_input_height)
                x2_scaled = int(x2 * original_width / model_input_width)
                y2_scaled = int(y2 * original_height / model_input_height)
                final_detections_scaled.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, score, class_id])

            # Draw bounding boxes and display the frame
            frame_with_boxes = frame.copy()
            for det in final_detections_scaled:
                x1, y1, x2, y2, score, class_id = det
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"turtle: {score:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_with_boxes, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame_with_boxes, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save annotated frame if detections are made
            if final_detections_scaled:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_filename = os.path.join(gallery_dir, f"detection_{timestamp}.jpg")
                cv2.imwrite(image_filename, frame_with_boxes)
                print(f"Saved detection image to {image_filename}")

            # Display inference and post-processing times
            cv2.putText(frame_with_boxes, f"Inference: {inference_time:.4f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, f"Post-process: {postprocess_time:.4f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # No live display window, only saving frames

    finally:
        cap.release()
        # cv2.destroyAllWindows() # Not needed without imshow

if __name__ == "__main__":
    model_path = "models/pawikan_int8.tflite"
    # IMPORTANT: Replace with your actual RTSP URL
    # For testing, you can use a sample video file path if you don't have a live RTSP stream:
    # rtsp_url = "file:///path/to/your/video.mp4"
    # Or a public RTSP stream (be aware of privacy and terms of service):
    # rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    rtsp_url = "rtsp://localhost:8554/test" # Placeholder, user needs to update this
    gallery_dir = "detections" # Directory to save annotated images

    print(f"--- Starting RTSP Inference with model: {model_path} ---")
    run_rtsp_inference(
        model_path=model_path,
        rtsp_url=rtsp_url,
        gallery_dir=gallery_dir,
        confidence_threshold=0.1,
        iou_threshold=0.45,
        max_detections=50
    )
