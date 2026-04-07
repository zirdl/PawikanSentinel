#!/bin/bash
# test_image.sh — Run the model on a real image.
# Usage: bash test_image.sh path/to/image.jpg

set -euo pipefail

IMAGE="${1:?Usage: bash $0 <image_path>}"

if [ ! -f "$IMAGE" ]; then
    echo "Error: file not found: $IMAGE"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "No .venv found. Run 'bash setup.sh' or 'bash test_model.sh' first."
    exit 1
fi

source .venv/bin/activate

python3 - "$IMAGE" <<'PYEOF'
import sys
from ultralytics import YOLO

image_path = sys.argv[1]

print(f"\n━━━ Loading model ━━━")
model = YOLO("models/turtle_detector.pt")
print(f"  → Classes: {model.names}")

print(f"\n━━━ Inferencing: {image_path} ━━━")
results = model(image_path, imgsz=320, conf=0.25)

for r in results:
    boxes = r.boxes
    if len(boxes) == 0:
        print("  → No detections.")
    else:
        print(f"  → {len(boxes)} detection(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"     {name}  conf={conf:.3f}  box={xyxy.astype(int).tolist()}")

print()
PYEOF
