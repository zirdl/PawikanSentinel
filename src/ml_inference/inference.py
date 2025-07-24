import numpy as np
from src.ml_inference.model_loader import ModelLoader

class InferenceEngine:
    """
    Performs inference using a loaded TFLite or ONNX model.
    """

    def __init__(self, model_loader: ModelLoader):
        """
        Initializes the InferenceEngine.

        Args:
            model_loader: The ModelLoader instance with a loaded model.
        """
        self.model_loader = model_loader
        self.model_type = model_loader.model_type

    def run(self, input_data: np.ndarray) -> dict:
        """
        Runs inference on the input data.

        Args:
            input_data (np.ndarray): The preprocessed input data.

        Returns:
            dict: A dictionary containing the raw detection results.
        """
        if self.model_type == 'tflite':
            return self._run_tflite(input_data)
        elif self.model_type == 'onnx':
            return self._run_onnx(input_data)
        else:
            raise ValueError("Unsupported model type")

    def _run_tflite(self, input_data: np.ndarray) -> dict:
        interpreter = self.model_loader.interpreter
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = {output['name']: interpreter.get_tensor(output['index']) for output in output_details}
        return output_data

    def _run_onnx(self, input_data: np.ndarray) -> dict:
        session = self.model_loader.session
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        results = session.run(output_names, {input_name: input_data})

        output_data = {name: res for name, res in zip(output_names, results)}
        return output_data

if __name__ == '__main__':
    # --- Example Usage ---
    # IMPORTANT: Replace this with the actual path to your TFLite model.
    MODEL_PATH = "path/to/your/model.tflite"

    # 1. Load the model
    model_loader = ModelLoader(MODEL_PATH)
    if model_loader.load():
        # 2. Create the inference engine
        inference_engine = InferenceEngine(model_loader.interpreter)

        # 3. Create a dummy input tensor (replace with your actual preprocessed frame)
        #    The shape should match the model's input shape.
        input_details = model_loader.get_input_details()
        input_shape = input_details[0]['shape']
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # 4. Run inference
        results = inference_engine.run(dummy_input)

        # 5. Print the results
        print("Inference Results:")
        for name, tensor in results.items():
            print(f"  - {name}: shape={tensor.shape}")