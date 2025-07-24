import tensorflow as tf
import onnxruntime as ort
import os

class ModelLoader:
    """
    Loads a TFLite or ONNX model and prepares it for inference.
    """

    def __init__(self, model_path: str):
        """
        Initializes the ModelLoader.

        Args:
            model_path (str): The path to the model file.
        """
        self.model_path = model_path
        self.interpreter = None
        self.session = None
        self.model_type = self._get_model_type()

    def _get_model_type(self):
        _, ext = os.path.splitext(self.model_path)
        if ext == '.tflite':
            return 'tflite'
        elif ext == '.onnx':
            return 'onnx'
        else:
            return None

    def load(self) -> bool:
        """
        Loads the model and allocates tensors or creates an inference session.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        if self.model_type == 'tflite':
            return self._load_tflite()
        elif self.model_type == 'onnx':
            return self._load_onnx()
        else:
            print(f"Unsupported model type for: {self.model_path}")
            return False

    def _load_tflite(self):
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            if self.interpreter:
                self.interpreter.allocate_tensors()
            print("TFLite model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load TFLite model: {e}")
            return False

    def _load_onnx(self):
        try:
            self.session = ort.InferenceSession(self.model_path)
            print("ONNX model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return False

    def get_input_details(self):
        """
        Gets the input details of the loaded model.
        """
        if self.model_type == 'tflite':
            return self.interpreter.get_input_details()
        elif self.model_type == 'onnx':
            return self.session.get_inputs()
        return None

    def get_output_details(self):
        """
        Gets the output details of the loaded model.
        """
        if self.model_type == 'tflite':
            return self.interpreter.get_output_details()
        elif self.model_type == 'onnx':
            return self.session.get_outputs()
        return None
