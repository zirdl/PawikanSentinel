import os
import time
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from pydantic import BaseModel

from ..core.database import get_db_connection
from .inference import RTSPInferenceWorker, DetectionResult

# Load environment variables from .env file
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "pawikansentinel-era7l/2")
MAX_WORKERS = int(os.getenv("MAX_INFERENCE_WORKERS", "10"))

# --- Logging Setup ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply sensitive data filter to prevent exposing secrets in logs
from ..logging import setup_sensitive_data_logging
setup_sensitive_data_logging()
# --- End Logging Setup ---

if not ROBOFLOW_API_KEY:
    logger.warning("ROBOFLOW_API_KEY not set in .env. Inference will not work.")

# Pydantic models for API responses
class WorkerStats(BaseModel):
    camera_id: int
    running: bool
    paused: bool
    buffer_size: int
    stats: dict
    circuit_breaker_state: str

class ServiceHealth(BaseModel):
    status: str
    active_workers: int
    total_cameras: int
    worker_stats: List[WorkerStats]

# FastAPI app for the inference service
app = FastAPI(title="Pawikan Sentinel Inference Service", version="1.0.0")

# Thread pool executor for managing workers
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
active_inference_workers: Dict[int, RTSPInferenceWorker] = {}
worker_futures = {}

# Health and monitoring
service_stats = {
    "start_time": datetime.now().isoformat(),
    "restart_count": 0
}

@app.on_event("startup")
async def startup_event():
    """Initialize inference workers for active cameras"""
    logger.info("Inference Service: Initializing with thread pool executor")
    service_stats["start_time"] = datetime.now().isoformat()
    service_stats["restart_count"] += 1
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE active = TRUE")
        active_cameras = c.fetchall()
        conn.close()

        for camera in active_cameras:
            await start_inference_for_camera_internal(camera["id"])
            logger.info(f"Inference Service: Started worker for camera {camera['id']} ({camera['name']})")
            
    except Exception as e:
        logger.error(f"Inference Service: Error during startup: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown all inference workers"""
    logger.info("Inference Service: Shutting down inference workers...")
    
    # Stop all workers
    stop_futures = []
    for camera_id, worker in active_inference_workers.items():
        logger.info(f"Inference Service: Stopping worker for camera {camera_id}")
        future = executor.submit(worker.stop)
        stop_futures.append((camera_id, future))
    
    # Wait for all workers to stop
    for camera_id, future in stop_futures:
        try:
            future.result(timeout=10)  # 10 second timeout
            logger.info(f"Inference Service: Stopped worker for camera {camera_id}")
        except Exception as e:
            logger.error(f"Inference Service: Error stopping worker {camera_id}: {e}")
    
    # Clear collections
    active_inference_workers.clear()
    worker_futures.clear()
    
    # Shutdown executor
    executor.shutdown(wait=True)
    logger.info("Inference Service: Shutdown complete.")

async def start_inference_for_camera_internal(camera_id: int) -> bool:
    """Internal function to start inference for a camera"""
    # Check if worker already exists
    if camera_id in active_inference_workers:
        worker = active_inference_workers[camera_id]
        if worker.running and not worker.paused:
            logger.info(f"Inference worker for camera {camera_id} is already running.")
            return True
    
    # Get camera details
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE id = ?", (camera_id,))
    camera = c.fetchone()
    conn.close()

    if not camera:
        logger.warning(f"Camera {camera_id} not found for starting inference.")
        return False
        
    if not camera["active"]:
        logger.warning(f"Camera {camera_id} is not active, cannot start inference.")
        return False

    # Stop existing worker if it exists
    if camera_id in active_inference_workers:
        old_worker = active_inference_workers[camera_id]
        old_worker.stop()
        if camera_id in worker_futures:
            worker_futures[camera_id].cancel()
    
    # Create new worker
    worker = RTSPInferenceWorker(
        camera_id=camera["id"],
        rtsp_url=camera["rtsp_url"],
        model_id=ROBOFLOW_MODEL_ID,
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )
    
    # Start worker in thread pool
    future = executor.submit(worker.run)
    active_inference_workers[camera_id] = worker
    worker_futures[camera_id] = future
    
    logger.info(f"Inference Service: Started worker for camera {camera_id} ({camera['name']}).")
    return True

@app.post("/inference/start/{camera_id}")
async def start_inference_for_camera(camera_id: int):
    """Start inference for a specific camera"""
    success = await start_inference_for_camera_internal(camera_id)
    if success:
        return {"message": f"Inference Service: Started worker for camera {camera_id}."}
    else:
        raise HTTPException(status_code=400, detail="Failed to start inference worker.")

@app.post("/inference/stop/{camera_id}")
async def stop_inference_for_camera(camera_id: int):
    """Stop inference for a specific camera"""
    if camera_id not in active_inference_workers:
        logger.info(f"Inference worker for camera {camera_id} is not running.")
        return {"message": f"Inference worker for camera {camera_id} is not running."}

    worker = active_inference_workers[camera_id]
    worker.stop()
    
    # Wait for worker to stop (with timeout)
    if camera_id in worker_futures:
        try:
            worker_futures[camera_id].result(timeout=5)
        except Exception as e:
            logger.warning(f"Timeout waiting for worker {camera_id} to stop: {e}")
    
    # Remove from tracking
    del active_inference_workers[camera_id]
    if camera_id in worker_futures:
        del worker_futures[camera_id]
        
    logger.info(f"Inference Service: Stopped worker for camera {camera_id}.")
    return {"message": f"Inference Service: Stopped worker for camera {camera_id}."}

@app.post("/inference/pause/{camera_id}")
async def pause_inference_for_camera(camera_id: int):
    """Pause inference for a specific camera"""
    if camera_id not in active_inference_workers:
        raise HTTPException(status_code=404, detail="Camera worker not found.")
    
    worker = active_inference_workers[camera_id]
    worker.pause()
    logger.info(f"Inference Service: Paused worker for camera {camera_id}.")
    return {"message": f"Inference Service: Paused worker for camera {camera_id}."}

@app.post("/inference/resume/{camera_id}")
async def resume_inference_for_camera(camera_id: int):
    """Resume inference for a specific camera"""
    if camera_id not in active_inference_workers:
        raise HTTPException(status_code=404, detail="Camera worker not found.")
    
    worker = active_inference_workers[camera_id]
    worker.resume()
    logger.info(f"Inference Service: Resumed worker for camera {camera_id}.")
    return {"message": f"Inference Service: Resumed worker for camera {camera_id}."}

@app.get("/inference/stats/{camera_id}")
async def get_worker_stats(camera_id: int):
    """Get statistics for a specific worker"""
    if camera_id not in active_inference_workers:
        raise HTTPException(status_code=404, detail="Camera worker not found.")
    
    worker = active_inference_workers[camera_id]
    return worker.get_stats()

@app.get("/inference/health")
async def get_service_health():
    """Get overall service health"""
    # Get worker stats
    worker_stats_list = []
    for camera_id, worker in active_inference_workers.items():
        worker_stats_list.append(WorkerStats(**worker.get_stats()))
    
    # Count active cameras
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as count FROM cameras")
    total_cameras = c.fetchone()["count"]
    conn.close()
    
    return ServiceHealth(
        status="healthy" if len(active_inference_workers) > 0 else "degraded",
        active_workers=len(active_inference_workers),
        total_cameras=total_cameras,
        worker_stats=worker_stats_list
    )

@app.get("/inference/workers")
async def list_workers():
    """List all active workers"""
    return {
        "active_workers": list(active_inference_workers.keys()),
        "max_workers": MAX_WORKERS,
        "current_workers": len(active_inference_workers)
    }
