"""
ML Inference Engine Component
"""

import numpy as np
import tflite_runtime.interpreter as tflite
from typing import List, Dict

class MLInferenceEngine:
    """
    Runs the optimized YOLOv5n TFLite model for turtle detection.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_model(self):
        """
        Loads the TFLite model and allocates tensors.
        """
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Model {self.model_path} loaded successfully.")

    def run_inference(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs inference on a single frame.
        """
        if self.interpreter is None:
            raise Exception("Model not loaded. Call load_model() first.")

        # Preprocess frame (example: normalize, resize)
        input_data = np.expand_dims(frame, axis=0)
        input_data = (input_data.astype(np.float32) / 255.0)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Post-process results (example)
        raw_detections = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dummy processing for now
        detections = [] # Process raw_detections to get bounding boxes, scores, etc.
        return detections

