"""
Download BVRA/TurtleDetector from HuggingFace and export to ONNX formats.
Exports both FP32 (full precision) and INT8 (quantized) ONNX models.
Run this once on your dev machine or directly on the Pi.

Usage:
    python export_model.py [--output-dir models] [--quantize]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# HuggingFace model repo
MODEL_REPO = "BVRA/TurtleDetector"

def download_model(output_dir: Path) -> Path:
    """Download the TurtleDetector .pt model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.info("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {MODEL_REPO} from HuggingFace...")
    pt_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="turtle_detector.pt",
        local_dir=str(output_dir)
    )
    
    pt_path = Path(pt_path)
    size_mb = pt_path.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded model to {pt_path} ({size_mb:.1f} MB)")
    return pt_path


def export_to_onnx(pt_path: Path, output_dir: Path, quantize: bool = False) -> Path:
    """Export PyTorch .pt model to ONNX format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.info("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {pt_path}...")
    model = YOLO(str(pt_path))
    
    suffix = "_int8" if quantize else "_fp32"
    onnx_path = output_dir / f"turtle_detector{suffix}.onnx"
    
    logger.info(f"Exporting to ONNX ({'INT8 quantized' if quantize else 'FP32'})...")
    logger.info("This may take a few minutes...")
    
    exported = model.export(
        format="onnx",
        imgsz=640,
        half=False,
        int8=quantize,
        dynamic=False,
        simplify=True,
        opset=12,
    )
    
    # ultralytics exports to the same directory as the .pt file
    # Move to our output dir
    exported_path = Path(exported)
    final_path = onnx_path
    
    if exported_path != final_path:
        import shutil
        shutil.move(str(exported_path), str(final_path))
    
    size_mb = final_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported ONNX model to {final_path} ({size_mb:.1f} MB)")
    return final_path


def export_both(pt_path: Path, output_dir: Path) -> tuple:
    """Export both FP32 and INT8 ONNX models."""
    fp32_path = export_to_onnx(pt_path, output_dir, quantize=False)
    int8_path = export_to_onnx(pt_path, output_dir, quantize=True)
    return fp32_path, int8_path


def main():
    parser = argparse.ArgumentParser(description="Export TurtleDetector to ONNX")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Output directory for exported models (default: models/)"
    )
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="Only export INT8 quantized model (saves time)"
    )
    parser.add_argument(
        "--fp32-only",
        action="store_true",
        help="Only export FP32 model"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Step 1: Download
    pt_path = download_model(output_dir)
    
    # Step 2: Export
    if args.quantize_only:
        int8_path = export_to_onnx(pt_path, output_dir, quantize=True)
        logger.info(f"\nDone! INT8 model: {int8_path}")
    elif args.fp32_only:
        fp32_path = export_to_onnx(pt_path, output_dir, quantize=False)
        logger.info(f"\nDone! FP32 model: {fp32_path}")
    else:
        fp32_path, int8_path = export_both(pt_path, output_dir)
        logger.info(f"\nDone!")
        logger.info(f"  FP32 model:  {fp32_path}")
        logger.info(f"  INT8 model:  {int8_path}")
        logger.info(f"\nFor Raspberry Pi, use the INT8 model for best performance.")


if __name__ == "__main__":
    main()
