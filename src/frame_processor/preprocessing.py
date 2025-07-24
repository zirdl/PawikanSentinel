import cv2
import numpy as np

def preprocess_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Preprocesses a single frame for inference.

    Args:
        frame (np.ndarray): The input frame (from OpenCV).
        target_size (tuple[int, int]): The target size (width, height) for the model.

    Returns:
        np.ndarray: The preprocessed frame.
    """
    # Resize the frame to the target size
    resized_frame = cv2.resize(frame, target_size)

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize the frame to the range [0, 1]
    normalized_frame = rgb_frame / 255.0

    # Add a batch dimension
    # Add a batch dimension and transpose to (1, channels, height, width) for ONNX
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0).transpose(0, 3, 1, 2)

    return preprocessed_frame.astype(np.float32)
