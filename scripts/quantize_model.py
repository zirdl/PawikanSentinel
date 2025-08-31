from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Export the model to TFLite dynamic range quantization format with verbose logging
model.export(format='tflite', verbose=True)

print("Model quantized to dynamic range.")