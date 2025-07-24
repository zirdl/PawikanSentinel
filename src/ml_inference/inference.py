import numpy as np
import tensorflow as tf

class InferenceEngine:
    """
    Performs inference using a loaded TFLite model.
    """

    def __init__(self, interpreter):
        """
        Initializes the InferenceEngine.

        Args:
            interpreter: The loaded TFLite interpreter.
        """
        self.interpreter = interpreter

    def run(self, input_data: np.ndarray) -> dict:
        """
        Runs inference on the input data.

        Args:
            input_data (np.ndarray): The preprocessed input data.

        Returns:
            dict: A dictionary containing the raw detection results.
        """
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # The output is a dictionary with keys corresponding to the output tensor names
        # and values as the output tensors.
        # For YOLOv5, this will typically be a single tensor with shape (1, N, 85)
        # where N is the number of detections and 85 is [x, y, w, h, confidence, class_probs...]
        output_data = {output['name']: self.interpreter.get_tensor(output['index']) for output in output_details}

        return output_data

if __name__ == '__main__':
    from model_loader import ModelLoader

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
