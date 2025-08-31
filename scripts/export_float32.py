
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Export the model to TFLite float32 format
model.export(format='tflite')

print("Model exported to best_float32.tflite")
