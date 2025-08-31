import tflite_runtime.interpreter as tflite
import numpy as np

def inspect_tflite_model(model_path):
    """Loads a TFLite model and prints its input and output tensor details."""
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        print(f"--- Model: {model_path} ---")

        print("\n--- Inputs ---")
        input_details = interpreter.get_input_details()
        for detail in input_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Data Type: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")
            print("-" * 10)

        print("\n--- Outputs ---")
        output_details = interpreter.get_output_details()
        for detail in output_details:
            print(f"  Name: {detail['name']}")
            print(f"  Shape: {detail['shape']}")
            print(f"  Data Type: {detail['dtype']}")
            print(f"  Quantization: {detail['quantization']}")
            print("-" * 10)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    model_path = "models/best_full_integer_quant.tflite"
    inspect_tflite_model(model_path)
