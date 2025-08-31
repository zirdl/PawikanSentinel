import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import argparse
import os
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurtleDetector:
    def __init__(self, model_path, confidence_threshold=0.45, iou_threshold=0.3, simulation_mode=False):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.simulation_mode = simulation_mode
        
        # Detection consistency tracking with spatial awareness
        self.detection_history = deque(maxlen=10)
        self.frame_count = 0
        
        # Background subtraction for motion detection (only for real streams)
        if not simulation_mode:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=500)
            logger.info("Real stream mode: Motion detection enabled")
        else:
            self.bg_subtractor = None
            logger.info("Simulation mode: Motion detection disabled")
        
        # Load model
        logger.info(f"Loading model: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Log model details
        input_shape = self.input_details[0]['shape']
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model input dtype: {self.input_details[0]['dtype']}")
        logger.info(f"Model output shape: {self.output_details[0]['shape']}")
        logger.info(f"Model output dtype: {self.output_details[0]['dtype']}")

    def decode_yolov8_output(self, output_data, input_shape):
        """Enhanced YOLOv8 output decoding with aggressive false positive filtering"""
        
        # Remove batch dimension
        predictions = output_data[0]  # (5, 8400)
        
        # Extract components
        boxes_xywh = predictions[:4, :].T  # (8400, 4) - x, y, w, h
        class_scores_raw = predictions[4:, :].T  # (8400, 1) - raw scores
        
        # Apply sigmoid activation to convert logits to probabilities
        class_scores = 1 / (1 + np.exp(-class_scores_raw))
        
        # For single class, we just take the single score
        scores = class_scores[:, 0]  # (8400,)
        class_ids = np.zeros(len(scores), dtype=int)  # All class 0 (turtle)
        
        # Log score statistics every 30 frames for debugging
        if self.frame_count % 30 == 0:
            mode_str = "SIM" if self.simulation_mode else "LIVE"
            logger.info(f"[{mode_str}] Score stats - Min: {scores.min():.4f}, Max: {scores.max():.4f}, "
                       f"Mean: {scores.mean():.4f}, Above {self.confidence_threshold}: {(scores > self.confidence_threshold).sum()}")
        
        # Filter by confidence threshold
        keep_indices = scores >= self.confidence_threshold
        
        if not np.any(keep_indices):
            return []
            
        filtered_boxes = boxes_xywh[keep_indices]
        filtered_scores = scores[keep_indices]
        filtered_class_ids = class_ids[keep_indices]
        
        # Convert xywh to xyxy format
        x = filtered_boxes[:, 0]
        y = filtered_boxes[:, 1]
        w = filtered_boxes[:, 2]
        h = filtered_boxes[:, 3]
        
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
        
        # Scale boxes to input dimensions
        input_height, input_width = input_shape
        scaled_boxes = boxes_xyxy * np.array([input_width, input_height, input_width, input_height])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            scaled_boxes.tolist(),
            filtered_scores.tolist(),
            self.confidence_threshold,
            self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Handle different OpenCV versions
        if isinstance(indices, tuple):
            indices = indices[0]
        if indices.ndim == 2:
            indices = indices.flatten()
        
        # Enhanced filtering for false positives
        final_detections = []
        for i in indices:
            box = scaled_boxes[i]
            score = filtered_scores[i]
            class_id = filtered_class_ids[i]
            
            # Calculate box dimensions
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height
            aspect_ratio = box_width / box_height if box_height > 0 else 0
            
            # Adjust filtering based on mode
            if self.simulation_mode:
                # Slightly more lenient for simulation testing
                min_area = 600
                max_area = input_width * input_height * 0.3
                aspect_min, aspect_max = 0.5, 2.8
                edge_margin = 25
                top_exclusion = 0.25  # Allow detections in top 25%
            else:
                # Stricter for real deployment
                min_area = 800
                max_area = input_width * input_height * 0.25
                aspect_min, aspect_max = 0.6, 2.5
                edge_margin = 30
                top_exclusion = 0.3  # Skip top 30%
            
            # 1. Size filtering
            if box_area < min_area or box_area > max_area:
                continue
            
            # 2. Aspect ratio filtering
            if aspect_ratio < aspect_min or aspect_ratio > aspect_max:
                continue
            
            # 3. Edge detection filtering
            if (box[0] < edge_margin or box[1] < edge_margin or 
                box[2] > input_width - edge_margin or box[3] > input_height - edge_margin):
                continue
            
            # 4. Position filtering
            center_y = (box[1] + box[3]) / 2
            if center_y < input_height * top_exclusion:
                continue
            
            # 5. Confidence-based stricter filtering
            if score < 0.5:
                stricter_aspect_min = 0.8 if not self.simulation_mode else 0.7
                stricter_aspect_max = 1.8 if not self.simulation_mode else 2.0
                if aspect_ratio < stricter_aspect_min or aspect_ratio > stricter_aspect_max:
                    continue
                if box_area < min_area * 1.3:
                    continue
            
            # 6. Box shape validation
            min_dimension = min(box_width, box_height)
            max_dimension = max(box_width, box_height)
            if min_dimension < 20 or max_dimension < 30:
                continue
            
            final_detections.append({
                'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                'confidence': float(score),
                'class_id': int(class_id),
                'area': box_area,
                'aspect_ratio': aspect_ratio
            })
        
        # Sort by confidence weighted with area
        max_area = max(min_area * 10, input_width * input_height * 0.25)
        final_detections.sort(key=lambda x: (x['confidence'] * 0.8 + (x['area'] / max_area) * 0.2), reverse=True)
        
        return final_detections[:3]  # Limit to top 3 detections

    def filter_by_motion(self, frame, detections):
        """Filter detections based on motion (simulation mode aware)"""
        if not detections:
            return detections
        
        # In simulation mode, skip motion filtering but add dummy motion data
        if self.simulation_mode or self.bg_subtractor is None:
            for det in detections:
                # Assign realistic motion ratios based on confidence
                if det['confidence'] > 0.6:
                    det['motion_ratio'] = 0.4  # High confidence = assume good motion
                elif det['confidence'] > 0.45:
                    det['motion_ratio'] = 0.2  # Medium confidence = some motion
                else:
                    det['motion_ratio'] = 0.1  # Low confidence = minimal motion
            return detections
        
        # Real stream: Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        filtered_detections = []
        for det in detections:
            bbox = det['bbox']
            
            # Extract region of interest from motion mask
            x1, y1, x2, y2 = bbox
            roi_mask = fg_mask[y1:y2, x1:x2]
            
            if roi_mask.size == 0:
                continue
            
            # Check motion in bounding box area
            motion_pixels = np.sum(roi_mask > 0)
            total_pixels = roi_mask.shape[0] * roi_mask.shape[1]
            motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
            
            # Allow detections with motion OR high confidence
            if motion_ratio > 0.05 or det['confidence'] > 0.6:
                det['motion_ratio'] = motion_ratio
                filtered_detections.append(det)
        
        return filtered_detections

    def is_consistent_detection(self, detections):
        """Enhanced consistency checking (simulation mode adjusted)"""
        if not detections:
            self.detection_history.append([])
            return False
        
        # Store detection centers for spatial consistency
        current_centers = []
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            current_centers.append((center_x, center_y, det['confidence'], det['area']))
        
        self.detection_history.append(current_centers)
        
        # Adjust requirements based on mode
        if self.simulation_mode:
            min_history = 3  # Less history needed for simulation
            min_frames_with_detections = 2
            min_consistent_frames = 1
            spatial_threshold = 100
        else:
            min_history = 5
            min_frames_with_detections = 3
            min_consistent_frames = 2
            spatial_threshold = 80
        
        # Need minimum frames of history
        if len(self.detection_history) < min_history:
            return False
        
        # Check if we have detections in sufficient recent frames
        recent_frames = list(self.detection_history)[-min_history:]
        frames_with_detections = sum(1 for frame in recent_frames if frame)
        
        if frames_with_detections < min_frames_with_detections:
            return False
        
        # Spatial consistency check
        if len(current_centers) == 0:
            return False
        
        # Check if current detections are near previous detections
        for current_center in current_centers:
            found_consistent = False
            cx, cy, conf, area = current_center
            
            min_conf = 0.35 if self.simulation_mode else 0.4
            if conf < min_conf:
                continue
            
            consistent_count = 0
            check_frames = min_history - 1
            for frame in recent_frames[-check_frames:]:
                if not frame:
                    continue
                    
                for prev_center in frame:
                    px, py, pconf, parea = prev_center
                    distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    area_diff = abs(area - parea) / max(area, parea) if max(area, parea) > 0 else 1
                    
                    # Spatial and size consistency requirements
                    if distance < spatial_threshold and area_diff < 0.6 and pconf > (min_conf - 0.05):
                        consistent_count += 1
                        break
            
            # Check if we found enough consistent detections
            if consistent_count >= min_consistent_frames:
                found_consistent = True
                break
        
        return found_consistent

    def preprocess_frame(self, frame):
        """Enhanced preprocessing with normalization"""
        input_shape = self.input_details[0]['shape']
        model_height, model_width = input_shape[1], input_shape[2]
        
        # Resize frame
        resized_frame = cv2.resize(frame, (model_width, model_height))
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Handle quantization
        if self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (rgb_frame / input_scale + input_zero_point).astype(np.int8)
        else:
            # Float32 model - normalize to 0-1
            input_data = rgb_frame.astype(np.float32) / 255.0
        
        return np.expand_dims(input_data, axis=0)

    def process_frame(self, frame):
        """Main frame processing with comprehensive filtering"""
        start_time = time.time()
        
        # Preprocess
        input_data = self.preprocess_frame(frame)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        inference_start = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - inference_start
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle quantized output
        if self.output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Decode detections
        input_shape = self.input_details[0]['shape']
        detections = self.decode_yolov8_output(output_data, (input_shape[1], input_shape[2]))
        
        # Apply motion filtering
        detections = self.filter_by_motion(frame, detections)
        
        # Check temporal consistency
        is_consistent = self.is_consistent_detection(detections)
        
        # Draw detections on frame
        annotated_frame = self.draw_detections(frame, detections, is_consistent)
        
        # Add performance info and mode indicator
        total_time = time.time() - start_time
        fps = 1 / total_time if total_time > 0 else 0
        
        mode_indicator = "[SIM]" if self.simulation_mode else "[LIVE]"
        cv2.putText(annotated_frame, f"{mode_indicator} FPS: {fps:.1f} | Inference: {inference_time*1000:.0f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detections:
            status_color = (0, 255, 0) if is_consistent else (0, 165, 255)
            status_text = f"RELIABLE TURTLE" if is_consistent else f"POSSIBLE TURTLE ({len(detections)})"
            cv2.putText(annotated_frame, status_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        self.frame_count += 1
        return annotated_frame, detections, is_consistent

    def draw_detections(self, frame, detections, is_consistent):
        """Draw bounding boxes and labels with enhanced info"""
        if not detections:
            return frame.copy()
        
        annotated_frame = frame.copy()
        original_height, original_width = frame.shape[:2]
        input_shape = self.input_details[0]['shape']
        model_height, model_width = input_shape[1], input_shape[2]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            motion_ratio = detection.get('motion_ratio', 0)
            
            # Scale coordinates back to original frame size
            x1 = int(bbox[0] * original_width / model_width)
            y1 = int(bbox[1] * original_height / model_height)
            x2 = int(bbox[2] * original_width / model_width)
            y2 = int(bbox[3] * original_height / model_height)
            
            # Color coding based on reliability
            if is_consistent:
                color = (0, 255, 0)  # Green for consistent
                thickness = 3
            elif confidence > 0.5:
                color = (0, 165, 255)  # Orange for high confidence
                thickness = 2
            else:
                color = (0, 100, 255)  # Red-orange for low confidence
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label
            label = f"turtle: {confidence:.3f}"
            if not self.simulation_mode and motion_ratio > 0:
                label += f" | motion: {motion_ratio:.2f}"
            elif self.simulation_mode:
                label += " | SIM"
            
            # Draw label background and text
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - label_height - baseline - 10),
                         (x1 + label_width, y1), color, cv2.FILLED)
            
            cv2.putText(annotated_frame, label, (x1, y1 - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Add detection rank number
            cv2.putText(annotated_frame, str(i+1), (x1-15, y1+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated_frame


def main():
    parser = argparse.ArgumentParser(description="Turtle Detection System with Simulation Support")
    parser.add_argument("--rtsp-url", type=str, required=True, help="RTSP stream URL")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for detections")
    parser.add_argument("--model-path", type=str, default="pawikan_int8.tflite", help="Model path")
    parser.add_argument("--confidence", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for NMS")
    parser.add_argument("--save-all", action="store_true", help="Save all frames with detections")
    parser.add_argument("--display", action="store_true", help="Display video feed (for debugging)")
    parser.add_argument("--simulation", action="store_true", help="Simulation mode for VLC testing (disables motion detection)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector with simulation mode
    detector = TurtleDetector(args.model_path, args.confidence, args.iou_threshold, args.simulation)
    
    # Open RTSP stream
    cap = cv2.VideoCapture(args.rtsp_url)
    
    if not cap.isOpened():
        logger.error(f"Could not open RTSP stream: {args.rtsp_url}")
        return
    
    mode_str = "simulation" if args.simulation else "live stream"
    logger.info(f"Successfully opened RTSP {mode_str}. Processing frames...")
    
    if args.simulation:
        logger.info("SIMULATION MODE: Motion detection disabled, relaxed filtering enabled")
    else:
        logger.info("LIVE MODE: Full filtering with motion detection enabled")
    
    frame_count = 0
    last_save_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.simulation:
                    logger.info("Video ended, restarting simulation...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                    continue
                else:
                    logger.warning("Could not read frame, retrying connection...")
                    time.sleep(2)
                    cap.open(args.rtsp_url)
                    continue
            
            frame_count += 1
            
            # Process frame
            annotated_frame, detections, is_consistent = detector.process_frame(frame)
            
            # Save frame based on criteria
            current_time = time.time()
            should_save = False
            save_type = ""
            
            # Different saving logic for simulation vs live
            if args.simulation:
                # More frequent saving for simulation testing
                if is_consistent and (current_time - last_save_time) > 2:
                    should_save = True
                    save_type = "reliable_sim"
                elif args.save_all and detections and (current_time - last_save_time) > 1:
                    should_save = True
                    save_type = "detection_sim"
            else:
                # Conservative saving for live deployment
                if is_consistent and (current_time - last_save_time) > 3:
                    should_save = True
                    save_type = "reliable_live"
                elif args.save_all and detections and (current_time - last_save_time) > 5:
                    should_save = True
                    save_type = "detection_live"
            
            if should_save:
                timestamp = int(current_time)
                filename = f"{save_type}_{timestamp}_{frame_count}.jpg"
                output_path = os.path.join(args.output_dir, filename)
                
                cv2.imwrite(output_path, annotated_frame)
                logger.info(f"Saved {save_type} image: {filename} ({len(detections)} detections)")
                last_save_time = current_time
            
            # Display frame for debugging
            if args.display:
                window_title = 'Turtle Detection - Simulation Mode' if args.simulation else 'Turtle Detection - Live Mode'
                cv2.imshow(window_title, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
    except KeyboardInterrupt:
        logger.info("Stopping detection...")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
