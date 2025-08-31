import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from typing import List, Tuple
from .postprocess import postprocess_output

class Detector:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.model_input_shape = (self.input_details['shape'][1], self.input_details['shape'][2])

    def preprocess_input(self, frame: np.ndarray) -> np.ndarray:
        resized_img = cv2.resize(frame, self.model_input_shape)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        if self.input_details['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details['quantization']
            input_data = (img_rgb / input_scale + input_zero_point).astype(np.int8)
        else:
            input_data = img_rgb.astype(np.float32) / 255.0
        
        return np.expand_dims(input_data, axis=0)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
    ) -> List[dict]:
        original_height, original_width = frame.shape[:2]
        input_data = self.preprocess_input(frame)

        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details["index"])

        if self.output_details['dtype'] == np.int8:
            output_scale, output_zero_point = self.output_details['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        detections = postprocess_output(
            output_data,
            original_shape=(original_height, original_width),
            input_shape=self.model_input_shape,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        return detections