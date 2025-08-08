import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

try:
    # Load the ONNX model
    onnx_model = onnx.load("model.onnx")
    tf_rep = prepare(onnx_model)

    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_rep.signatures['serving_default']])
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Apply default optimizations, including quantization
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    print("model.onnx successfully converted to model.tflite")
except Exception as e:
    print(f"Error during conversion: {e}")
