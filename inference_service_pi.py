import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from database import get_db_connection
from inference_pi import PiOptimizedWorker

# Load environment variables
load_dotenv()

# Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "pawikansentinel-era7l/2")
MAX_WORKERS = int(os.getenv("MAX_INFERENCE_WORKERS", "2"))

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Pawikan Sentinel Pi Inference Service", version="1.0.0")

# Thread pool for workers
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
active_workers: Dict[int, PiOptimizedWorker] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize workers for active cameras"""
    logger.info("Pi Inference Service: Starting up")
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE active = TRUE")
        active_cameras = c.fetchall()
        conn.close()

        for camera in active_cameras:
            await start_inference_for_camera_internal(camera["id"])
            logger.info(f"Started worker for camera {camera['id']} ({camera['name']})")
            
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown all workers"""
    logger.info("Pi Inference Service: Shutting down")
    
    # Stop all workers
    for camera_id, worker in active_workers.items():
        logger.info(f"Stopping worker for camera {camera_id}")
        worker.stop()
    
    # Wait a moment for workers to stop
    time.sleep(2)
    
    # Shutdown executor
    executor.shutdown(wait=False)
    logger.info("Pi Inference Service: Shutdown complete")

async def start_inference_for_camera_internal(camera_id: int) -> bool:
    """Internal function to start inference for a camera"""
    # Check if worker already exists
    if camera_id in active_workers:
        worker = active_workers[camera_id]
        if worker.running and not worker.paused:
            logger.info(f"Worker for camera {camera_id} is already running")
            return True
    
    # Get camera details
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE id = ?", (camera_id,))
    camera = c.fetchone()
    conn.close()

    if not camera:
        logger.warning(f"Camera {camera_id} not found")
        return False
        
    if not camera["active"]:
        logger.warning(f"Camera {camera_id} is not active")
        return False

    # Stop existing worker if needed
    if camera_id in active_workers:
        active_workers[camera_id].stop()
    
    # Create new worker
    worker = PiOptimizedWorker(
        camera_id=camera["id"],
        rtsp_url=camera["rtsp_url"],
        model_id=ROBOFLOW_MODEL_ID,
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    
    # Start worker
    executor.submit(worker.run)
    active_workers[camera_id] = worker
    
    logger.info(f"Started worker for camera {camera_id} ({camera['name']})")
    return True

@app.post("/inference/start/{camera_id}")
async def start_inference_for_camera(camera_id: int):
    """Start inference for a specific camera"""
    success = await start_inference_for_camera_internal(camera_id)
    if success:
        return {"message": f"Started worker for camera {camera_id}"}
    else:
        raise HTTPException(status_code=400, detail="Failed to start worker")

@app.post("/inference/stop/{camera_id}")
async def stop_inference_for_camera(camera_id: int):
    """Stop inference for a specific camera"""
    if camera_id not in active_workers:
        return {"message": f"No worker running for camera {camera_id}"}

    worker = active_workers[camera_id]
    worker.stop()
    
    del active_workers[camera_id]
    logger.info(f"Stopped worker for camera {camera_id}")
    return {"message": f"Stopped worker for camera {camera_id}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_workers": len(active_workers),
        "workers": list(active_workers.keys())
    }