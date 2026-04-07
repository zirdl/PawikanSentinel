#!/bin/bash
# test_model.sh — Download the YOLO11 turtle model and run a smoke test.
# Non-interactive. Uses a blank image to verify the model loads + infers.

set -euo pipefail

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# 1. Create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

source .venv/bin/activate

# 2. Install deps (with inference extras)
echo "Installing dependencies..."
uv sync --extra inference

# 3. Download model + test inference
python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
import os, time

print("\n━━━ Download ━━━")
start = time.time()
path = hf_hub_download(
    repo_id="BVRA/TurtleDetector",
    filename="turtle_detector.pt",
    local_dir="models",
)
elapsed = time.time() - start
size_mb = os.path.getsize(path) / (1024 * 1024)
print(f"  → {path}  ({size_mb:.1f} MB)  in {elapsed:.0f}s")

print("\n━━━ Load ━━━")
model = YOLO(path)
model.fuse()
print(f"  → Classes: {model.names}")

print("\n━━━ Test inference (320×320 blank image) ━━━")
img = np.zeros((320, 320, 3), dtype=np.uint8)
results = model(img, imgsz=320, conf=0.25, verbose=False)
detections = len(results[0].boxes)
print(f"  → {detections} detection(s) (expected 0 on blank image)")

print("\n✅ Model is working correctly.\n")
PYEOF
