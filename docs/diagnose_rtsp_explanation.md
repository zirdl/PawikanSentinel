# `diagnose_rtsp.py` - RTSP Stream Inference and Detection Saving

This script provides a way to run a YOLOv8 TFLite model (specifically an INT8 quantized model) on a live RTSP video stream, detect objects (e.g., sea turtles), and save annotated frames to a specified directory whenever a detection occurs. It does not display a live video feed, focusing solely on background inference and event-based image saving.

## How it Works

The script consists of two main functions: `decode_yolov8_output` for post-processing model predictions and `run_rtsp_inference` for handling the RTSP stream, running inference, and saving detections.

### 1. `decode_yolov8_output(output_data, input_shape, confidence_threshold, iou_threshold, num_classes, max_detections)`

This function is responsible for interpreting the raw output from the YOLOv8 TFLite model and converting it into meaningful object detections.

*   **Raw Output Processing**: The model's output is expected to be in a format like `(1, 5, 8400)`, where `5` represents `[x, y, w, h, confidence]` (or similar, depending on the model's exact output layer for class scores). The function extracts bounding box coordinates (`xywh`) and class scores.
*   **Confidence Filtering**: Detections are filtered based on a `confidence_threshold`. Only detections with a score equal to or higher than this threshold are considered.
*   **Coordinate Conversion**: Bounding box coordinates are converted from `xywh` (center_x, center_y, width, height) format to `x1y1x2y2` (top-left x, top-left y, bottom-right x, bottom-right y).
*   **Non-Maximum Suppression (NMS)**: `cv2.dnn.NMSBoxes` is used to eliminate redundant overlapping bounding boxes, keeping only the most confident detection for a given object. This is controlled by `iou_threshold` (Intersection Over Union).
*   **Scaling**: The detected box coordinates are scaled back to the original input size of the model.
*   **Return Format**: Returns a list of final detections, each containing `(x1, y1, x2, y2, score, class_id)`.

### 2. `run_rtsp_inference(model_path, rtsp_url, gallery_dir, confidence_threshold, iou_threshold, max_detections)`

This is the core function that manages the RTSP stream, runs the inference loop, and saves detected frames.

*   **Model Loading**: Initializes the TFLite interpreter with the provided `model_path` and allocates tensors. It retrieves input and output details, including the expected input shape and quantization parameters.
*   **RTSP Stream Connection**: Uses `cv2.VideoCapture` to connect to the specified `rtsp_url`. It includes basic reconnection logic if the stream fails to grab a frame.
*   **Gallery Directory Setup**: Ensures that the `gallery_dir` exists, creating it if necessary, to store the annotated detection images.
*   **Inference Loop**:
    *   Continuously reads frames from the RTSP stream.
    *   **Preprocessing**: Each frame is resized to the model's input dimensions (`model_input_width`, `model_input_height`) and converted to RGB. If the model is INT8 quantized, the pixel data is scaled and converted to `np.int8`. Otherwise, it's normalized to `0-1` float32.
    *   **Inference**: The preprocessed frame is fed to the TFLite interpreter using `interpreter.set_tensor` and `interpreter.invoke()`.
    *   **Post-processing**: The raw output from the interpreter is dequantized (if INT8) and then passed to `decode_yolov8_output` to get the final detections.
    *   **Detection Visualization**: If detections are found, bounding boxes, labels (e.g., "turtle: 0.95"), and confidence scores are drawn onto a copy of the original frame.
    *   **Saving Detections**: For every frame where detections are made, the annotated image is saved to the `gallery_dir` with a timestamp-based filename (e.g., `detection_20250831-103045.jpg`).
    *   **Performance Logging**: Prints the inference and post-processing times to the console.
*   **Resource Release**: Ensures that the video capture object is released when the loop breaks or an error occurs.

## How to Run

1.  **Ensure Dependencies**: Make sure you have `numpy`, `opencv-python-headless`, and `tflite-runtime` installed in your Python environment.
2.  **Update RTSP URL**: Open `scripts/diagnose_rtsp.py` and replace the placeholder `rtsp_url = "rtsp://user:pass@camera-ip:554/stream"` with your actual RTSP stream URL.
3.  **Execute the Script**: Run the script from your terminal:

    ```bash
    python scripts/diagnose_rtsp.py
    ```

    Or, if using `uv` (as per project context):

    ```bash
    uv run python scripts/diagnose_rtsp.py
    ```

The script will start connecting to the RTSP stream and will print messages to the console regarding connection status, inference times, and when detection images are saved. Annotated images will appear in the `detections` directory (or whatever `gallery_dir` you specify).

## Configuration Parameters

You can adjust the following parameters within the `if __name__ == "__main__":` block of the script:

*   `model_path`: Path to your TFLite model (e.g., `"models/pawikan_int8.tflite"`).
*   `rtsp_url`: The URL of your RTSP video stream. **This is critical and must be updated.**
*   `gallery_dir`: The directory where annotated detection images will be saved (e.g., `"detections"`).
*   `confidence_threshold`: The minimum confidence score (0.0 to 1.0) for a detection to be considered valid and drawn/saved.
*   `iou_threshold`: The Intersection Over Union (IoU) threshold used for Non-Maximum Suppression to filter out overlapping bounding boxes.
*   `max_detections`: The maximum number of detections to keep after NMS.
