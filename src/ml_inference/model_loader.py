import tensorflow as tf

class ModelLoader:
    """
    Loads a TFLite model and prepares it for inference.
    """

    def __init__(self, model_path: str):
        """
        Initializes the ModelLoader.

        Args:
            model_path (str): The path to the TFLite model file.
        """
        self.model_path = model_path
        self.interpreter = None

    def load(self) -> bool:
        """
        Loads the TFLite model and allocates tensors.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            print("TFLite model loaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to load TFLite model: {e}")
            return False

    def get_input_details(self):
        """
        Gets the input details of the loaded model.

        Returns:
            A list of dictionaries containing input details (name, index, shape, dtype).
        """
        return self.interpreter.get_input_details()

    def get_output_details(self):
        """
        Gets the output details of the loaded model.

        Returns:
            A list of dictionaries containing output details (name, index, shape, dtype).
        """
        return self.interpreter.get_output_details()
