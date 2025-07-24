import unittest
from unittest.mock import MagicMock
from src.ml_inference.model_loader import ModelLoader
from src.ml_inference.inference import InferenceEngine
import numpy as np

class TestMLInference(unittest.TestCase):

    def test_model_loader(self):
        # This is a basic test that checks if the loader can be initialized.
        # A more comprehensive test would require a dummy TFLite model.
        loader = ModelLoader("dummy.tflite")
        self.assertIsNotNone(loader)

    def test_inference_engine(self):
        mock_interpreter = MagicMock()
        mock_interpreter.get_input_details.return_value = [{'index': 0, 'shape': [1, 100, 100, 3]}]
        mock_interpreter.get_output_details.return_value = [{'name': 'output', 'index': 0}]

        mock_model_loader = MagicMock(spec=ModelLoader)
        mock_model_loader.model_type = 'tflite'  # Or 'onnx' depending on what you want to test
        mock_model_loader.interpreter = mock_interpreter # For tflite
        # mock_model_loader.session = mock_session # For onnx
        engine = InferenceEngine(mock_model_loader)
        dummy_input = np.zeros((1, 100, 100, 3), dtype=np.float32)
        engine.run(dummy_input)

        mock_interpreter.set_tensor.assert_called_once()
        mock_interpreter.invoke.assert_called_once()

if __name__ == '__main__':
    unittest.main()
