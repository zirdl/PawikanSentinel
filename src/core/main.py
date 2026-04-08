import os
import secrets
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from dotenv import load_dotenv
import httpx
import shutil
import zipfile
from pathlib import Path
import glob
import asyncio
import threading
import sqlite3
import random
import time

from fastapi import FastAPI, HTTPException, Request, Depends, Form, Cookie, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import WebSocket
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- Logging Setup ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

from .database import get_db_connection
from .utils import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_access_token,
    get_current_user_from_cookie,
    get_current_user_for_pages,
    SECRET_KEY,
    SESSION_SECRET_KEY,
    CSRF_SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    pwd_context
)

# Rate limiting instance
limiter = Limiter(key_func=get_remote_address)

# Global variable to store the background task
background_task = None

# Global variable to store the main event loop when available
main_event_loop = None

# ============================================================
# Inference Worker Management (merged from inference_service)
# ============================================================
from ..inference import RTSPInferenceWorker
from ..inference.yolo_detector import load_model, download_model

# YOLO11 model configuration
YOLO_MODEL_DIR = os.getenv("YOLO_MODEL_DIR", "models")
YOLO_INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "320"))
MAX_INFERENCE_WORKERS = int(os.getenv("MAX_INFERENCE_WORKERS", "10"))

# Thread pool for inference workers
inference_executor: Optional[ThreadPoolExecutor] = None
active_inference_workers: Dict[int, RTSPInferenceWorker] = {}
worker_futures = {}

# YOLO model (lazy-loaded)
_yolo_model = None
_model_load_lock = threading.Lock()

logger_inference = logging.getLogger("inference")


# ============================================================
# Inference Worker Management Functions
# ============================================================
async def _start_inference_for_camera_internal(camera_id: int) -> bool:
    """Start inference worker for a camera."""
    global inference_executor, active_inference_workers, worker_futures

    if inference_executor is None:
        logger_inference.warning("Inference executor not initialized.")
        return False

    if camera_id in active_inference_workers:
        worker = active_inference_workers[camera_id]
        if worker.running and not worker.paused:
            logger_inference.info(f"Worker for camera {camera_id} already running.")
            return True

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE id = ?", (camera_id,))
    camera = c.fetchone()
    conn.close()

    if not camera:
        logger_inference.warning(f"Camera {camera_id} not found.")
        return False

    if not camera["active"]:
        logger_inference.warning(f"Camera {camera_id} is not active.")
        return False

    # Stop existing worker
    if camera_id in active_inference_workers:
        old_worker = active_inference_workers[camera_id]
        old_worker.stop()
        if camera_id in worker_futures:
            worker_futures[camera_id].cancel()

    worker = RTSPInferenceWorker(
        camera_id=camera["id"],
        rtsp_url=camera["rtsp_url"],
    )

    future = inference_executor.submit(worker.run)
    active_inference_workers[camera_id] = worker
    worker_futures[camera_id] = future

    logger_inference.info(f"Started inference worker for camera {camera_id} ({camera['name']}).")
    return True


async def _stop_inference_for_camera_internal(camera_id: int) -> bool:
    """Stop inference worker for a camera."""
    global active_inference_workers, worker_futures

    if camera_id not in active_inference_workers:
        return False

    worker = active_inference_workers[camera_id]
    worker.stop()

    if camera_id in worker_futures:
        try:
            worker_futures[camera_id].result(timeout=5)
        except Exception as e:
            logger_inference.warning(f"Timeout stopping worker {camera_id}: {e}")
        del worker_futures[camera_id]

    del active_inference_workers[camera_id]
    logger_inference.info(f"Stopped inference worker for camera {camera_id}.")
    return True


async def _start_all_active_cameras():
    """Start inference workers for all active cameras."""
    global inference_executor, active_inference_workers, worker_futures

    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE active = TRUE")
        active_cameras = c.fetchall()
        conn.close()

        for camera in active_cameras:
            await _start_inference_for_camera_internal(camera["id"])
            logger_inference.info(f"Started worker for camera {camera['id']} ({camera['name']}).")
    except Exception as e:
        logger_inference.error(f"Error starting inference workers: {e}", exc_info=True)


async def _stop_all_inference_workers():
    """Gracefully stop all inference workers."""
    global inference_executor, active_inference_workers, worker_futures

    logger_inference.info("Stopping all inference workers...")

    stop_futures = []
    for camera_id, worker in active_inference_workers.items():
        worker.stop()
        if camera_id in worker_futures:
            stop_futures.append((camera_id, worker_futures[camera_id]))

    for camera_id, future in stop_futures:
        try:
            future.result(timeout=10)
            logger_inference.info(f"Stopped worker for camera {camera_id}.")
        except Exception as e:
            logger_inference.error(f"Error stopping worker {camera_id}: {e}")

    active_inference_workers.clear()
    worker_futures.clear()

    if inference_executor is not None:
        inference_executor.shutdown(wait=True)
        inference_executor = None

    logger_inference.info("All inference workers stopped.")


# ============================================================
# Lifespan (startup + shutdown)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    global main_event_loop, inference_executor, active_inference_workers

    # --- Startup ---
    main_event_loop = asyncio.get_running_loop()
    logger.info(f"Starting Pawikan Sentinel v1.0.0")

    # Ensure directories
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(YOLO_MODEL_DIR, exist_ok=True)
    logger.info(f"Detections directory: {detections_dir}")

    # Create tables
    create_tables()
    logger.info("Database tables created (WAL mode enabled).")

    # Ensure admin user
    conn = get_db_connection()
    user = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if not user:
        hashed_password = get_password_hash("admin123")
        conn.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            ("admin", hashed_password, "admin")
        )
        conn.commit()
        logger.info("Default admin user created: admin/admin123")
    conn.execute("DELETE FROM users WHERE username != ?", ("admin",))
    conn.commit()
    conn.close()

    # Initialize inference executor and start workers
    inference_executor = ThreadPoolExecutor(max_workers=MAX_INFERENCE_WORKERS)
    logger.info(f"Inference executor initialized (max {MAX_INFERENCE_WORKERS} workers).")
    await _start_all_active_cameras()

    # Start WebSocket broadcast background task
    asyncio.create_task(broadcast_detections_task())

    yield

    # --- Shutdown ---
    logger.info("Shutting down Pawikan Sentinel...")
    await _stop_all_inference_workers()
    logger.info("Shutdown complete.")


# Create FastAPI app with lifespan
app = FastAPI(title="Pawikan Sentinel", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=7200)  # 2 hours to match token expiry

# Cache control middleware to prevent sensitive pages from being cached
class CacheControlMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        # Apply cache control headers to authenticated pages only
        if hasattr(request, 'session') and request.session.get('csrf_token'):
            # For authenticated requests, prevent caching of sensitive pages
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        elif '/dashboard' in str(request.url) or '/settings' in str(request.url) or '/cameras' in str(request.url):
            # Also apply to protected routes even if no session yet
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response

# Add the cache control middleware
app.add_middleware(CacheControlMiddleware)

# Create serializer for CSRF tokens
serializer = URLSafeTimedSerializer(CSRF_SECRET_KEY)

# Define the detections directory
detections_dir = os.getenv("DETECTIONS_DIR", "detections")
# Ensure we're using an absolute path
if not os.path.isabs(detections_dir):
    detections_dir = os.path.abspath(detections_dir)

app.mount("/detections", StaticFiles(directory=detections_dir), name="detections")

# Include API routers
from ..api import auth, cameras, contacts, detections, analytics, system
from .database import get_db_connection

app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(contacts.router)
app.include_router(detections.router)
app.include_router(analytics.router)
app.include_router(system.router)

# --- WebSocket Initialization ---
from . import websocket_manager
from .websocket_manager import ConnectionManager, detection_broadcast_queue

# Initialize the global manager
websocket_manager.manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, access_token: Optional[str] = Cookie(None)):
    """WebSocket endpoint for real-time detection updates."""
    # Manual token check for WebSocket
    if not access_token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    payload = decode_access_token(access_token)
    if not payload or not payload.get("sub"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket_manager.manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and wait for client messages if any
            data = await websocket.receive_text()
            # Handle client messages if needed
    except Exception:
        # Client disconnected or error occurred
        pass
    finally:
        websocket_manager.manager.disconnect(websocket)

async def broadcast_detections_task():
    """Background task to broadcast detections from the threaded queue."""
    logger.info("Starting detection broadcast background task...")
    while True:
        try:
            # Use asyncio.to_thread to avoid blocking the event loop with queue.get()
            if not detection_broadcast_queue.empty():
                message = detection_broadcast_queue.get_nowait()
                await websocket_manager.manager.broadcast(message)
                detection_broadcast_queue.task_done()
            else:
                await asyncio.sleep(0.1) # Brief sleep to avoid high CPU usage
        except Exception as e:
            logger.error(f"Error in broadcast task: {e}")
            await asyncio.sleep(1)

# Create tables if they don't exist (uses shared get_db_connection with WAL mode)
def create_tables():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            class TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            FOREIGN KEY (camera_id) REFERENCES cameras (id)
        )
    """)

    # Indexes for query performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections (timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_camera_id ON detections (camera_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_camera_timestamp ON detections (camera_id, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cameras_active ON cameras (active)")

    conn.commit()
    conn.close()

# Detection management functions
def get_detection_stats():
    """Get detection statistics for the dashboard cards"""
    conn = get_db_connection()
    
    # Total detections
    total = conn.execute("SELECT COUNT(*) as count FROM detections").fetchone()["count"]
    
    # Today's detections
    today = conn.execute("""
        SELECT COUNT(*) as count FROM detections 
        WHERE DATE(timestamp) = DATE('now')
    """).fetchone()["count"]
    
    # This month's detections
    this_month = conn.execute("""
        SELECT COUNT(*) as count FROM detections 
        WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
    """).fetchone()["count"]
    
    conn.close()
    return {"total": total, "today": today, "this_month": this_month}

def get_detection_chart_data(period="month", month=None):
    """Get detection confidence data for chart visualization"""
    conn = get_db_connection()
    
    if period == "month":
        if month:
            # Get daily average confidence for that month
            data = conn.execute("""
                SELECT strftime('%Y-%m-%d', timestamp) as date, 
                       AVG(confidence * 100) as confidence
                FROM detections 
                WHERE strftime('%Y-%m', timestamp) = ?
                GROUP BY date ORDER BY date
            """, (month,)).fetchall()
        else:
            # Default: Current month's daily average
            data = conn.execute("""
                SELECT strftime('%m-%d', timestamp) as date, 
                       AVG(confidence * 100) as confidence
                FROM detections 
                WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
                GROUP BY date ORDER BY date
            """,).fetchall()
    
    conn.close()
    return [dict(row) for row in data]

def get_recent_detections(limit=10):
    """Get recent detections with camera locations"""
    conn = get_db_connection()
    detections = conn.execute("""
        SELECT d.id, d.timestamp, c.name as camera_name, c.name as camera_location, d.class, d.confidence, d.image_path
        FROM detections d
        LEFT JOIN cameras c ON d.camera_id = c.id
        ORDER BY d.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(detection) for detection in detections]

# CSRF protection
def generate_csrf_token(request: Request):
    if "csrf_token" not in request.session:
        request.session["csrf_token"] = secrets.token_hex(16)
    return serializer.dumps(request.session["csrf_token"])

async def verify_csrf_token(request: Request):
    try:
        form_csrf_token = request.headers.get("X-CSRFToken") or (await request.form()).get("csrf_token")
        if not form_csrf_token:
            raise HTTPException(status_code=403, detail="CSRF token missing")

        session_token = serializer.loads(form_csrf_token, max_age=3600)
        if session_token != request.session.get("csrf_token"):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
            
        # Regenerate CSRF token after successful validation
        request.session["csrf_token"] = secrets.token_hex(16)
    except SignatureExpired:
        raise HTTPException(status_code=403, detail="CSRF token expired")
    except BadTimeSignature:
        raise HTTPException(status_code=403, detail="Invalid CSRF token signature")
    except Exception as e:
        raise HTTPException(status_code=403, detail="CSRF verification failed")

# API Routes

# Detection API endpoints
@app.get("/api/detections/stats")
async def get_detection_stats_api(username: str = Depends(get_current_user_from_cookie)):
    response = JSONResponse(get_detection_stats())
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/detections/chart")
async def get_detection_chart_api(period: str = "month", month: str = None, username: str = Depends(get_current_user_from_cookie)):
    response = JSONResponse(get_detection_chart_data(period, month))
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/detections/recent")
async def get_recent_detections_api(username: str = Depends(get_current_user_from_cookie)):
    response = JSONResponse(get_recent_detections())
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/detections/gallery")
async def get_detection_gallery_api(username: str = Depends(get_current_user_from_cookie)):
    # Get detection images from the specified directory
    import os
    import glob
    
    detection_dir = os.getenv("DETECTIONS_DIR", "detections")
    # Ensure we're using an absolute path
    if not os.path.isabs(detection_dir):
        detection_dir = os.path.abspath(detection_dir)
    
    images = []
    
    if os.path.exists(detection_dir):
        # Get all image files in the directory and subdirectories
        image_patterns = [
            os.path.join(detection_dir, "**", "*.jpg"),
            os.path.join(detection_dir, "**", "*.jpeg"), 
            os.path.join(detection_dir, "**", "*.png"),
            os.path.join(detection_dir, "**", "*.gif"),
            os.path.join(detection_dir, "**", "*.webp")
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(pattern, recursive=True))
        
        # Sort by modification time (newest first)
        image_files.sort(key=os.path.getmtime, reverse=True)
        
        # Get the 24 most recent images for 6x4 grid
        for image_file in image_files[:24]:
            # Get file info
            stat = os.stat(image_file)
            
            # Extract detection ID or camera ID from filename or path to get camera location
            image_filename = os.path.basename(image_file)
            
            # Try to find corresponding detection record in database to get camera info
            conn = get_db_connection()
            c = conn.cursor()
            # Look for the image in the detections table by filename
            c.execute("""
                SELECT d.camera_id, c.name as camera_name 
                FROM detections d 
                LEFT JOIN cameras c ON d.camera_id = c.id 
                WHERE d.image_path LIKE ?
            """, (f'%{image_filename}',))
            
            detection_record = c.fetchone()
            conn.close()
            
            camera_name = "Unknown Location"
            if detection_record:
                camera_name = detection_record["camera_name"] or f"Camera {detection_record['camera_id']}"
            
            # Create path for the static file mount point
            images.append({
                "filename": image_filename,
                "path": f"/detections/{os.path.relpath(image_file, detection_dir)}",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "camera_name": camera_name  # Add camera location information
            })
    
    response = JSONResponse(images)
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# REMOVED: Registration routes to prevent creation of new accounts
# Only the admin account should exist


@app.post("/logout")
async def logout_user(request: Request):
    # Standardize to JSON response for SPA
    response = JSONResponse({"success": True, "message": "Logged out successfully"})
        
    response.delete_cookie("access_token", httponly=True, samesite="lax", path="/")
    request.session.pop("csrf_token", None)
    # Add cache control headers to ensure browser doesn't cache the logout response
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response




# Configuration API endpoints
@app.get("/api/config")
async def get_config(username: str = Depends(get_current_user_from_cookie)):
    """Get current configuration values"""
    config = {
        "confidence_threshold": os.getenv("CONFIDENCE_THRESHOLD", "80"),
        "frame_skip": os.getenv("FRAME_SKIP", "20"),
        "sms_cooldown": os.getenv("SMS_NOTIFICATION_COOLDOWN", "10")
    }
    response = JSONResponse(config)
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.post("/api/config")
async def update_config(
    request: Request,
    username: str = Depends(get_current_user_from_cookie)
):
    """Update configuration values"""
    try:
        data = await request.json()
        
        # Get current config values with defaults
        confidence_threshold = data.get("confidence_threshold", "80")
        frame_skip = data.get("frame_skip", "20")
        sms_cooldown = data.get("sms_cooldown", "10")
        
        # Update .env file
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, "r") as file:
                lines = file.readlines()
            
            # Update or add the configuration values
            updated_lines = []
            confidence_found = False
            frame_skip_found = False
            sms_cooldown_found = False
            
            for line in lines:
                if line.startswith("CONFIDENCE_THRESHOLD="):
                    updated_lines.append(f"CONFIDENCE_THRESHOLD={confidence_threshold}\n")
                    confidence_found = True
                elif line.startswith("FRAME_SKIP="):
                    updated_lines.append(f"FRAME_SKIP={frame_skip}\n")
                    frame_skip_found = True
                elif line.startswith("SMS_NOTIFICATION_COOLDOWN="):
                    updated_lines.append(f"SMS_NOTIFICATION_COOLDOWN={sms_cooldown}\n")
                    sms_cooldown_found = True
                else:
                    updated_lines.append(line)
            
            # Add missing config values if not found
            if not confidence_found:
                updated_lines.append(f"CONFIDENCE_THRESHOLD={confidence_threshold}\n")
            if not frame_skip_found:
                updated_lines.append(f"FRAME_SKIP={frame_skip}\n")
            if not sms_cooldown_found:
                updated_lines.append(f"SMS_NOTIFICATION_COOLDOWN={sms_cooldown}\n")
            
            # Write updated config back to file
            with open(env_path, "w") as file:
                file.writelines(updated_lines)
            
            # Reload environment variables
            load_dotenv(env_path, override=True)
            
            return JSONResponse({"message": "Configuration updated successfully!"})
        else:
            return JSONResponse({"error": "Configuration file not found"}, status_code=500)
            
    except Exception as e:
        return JSONResponse({"error": f"Failed to update configuration: {str(e)}"}, status_code=500)


# ============================================================
# System Status (single-process aware)
# ============================================================
@app.get("/api/system/status")
async def get_system_status(username: str = Depends(get_current_user_from_cookie)):
    """Get the status of all system components."""
    system_status = {}

    # Local inference service status
    worker_count = len(active_inference_workers)
    running_workers = sum(1 for w in active_inference_workers.values() if w.running and not w.paused)
    
    model_path = Path(YOLO_MODEL_DIR) / "turtle_detector.pt"
    model_status = 'loaded' if model_path.exists() else 'not_downloaded'

    system_status['inference_service'] = {
        'status': 'running' if running_workers > 0 else 'idle',
        'active_workers': running_workers,
        'total_workers': worker_count,
        'model_status': model_status,
        'model_path': str(model_path),
        'message': f"Local PyTorch (YOLO11) - {running_workers} active workers"
    }

    # Camera status & Feed processing
    try:
        conn = get_db_connection()
        cameras = conn.execute("SELECT id, name FROM cameras WHERE active = 1").fetchall()
        conn.close()

        processing_feeds = 0
        now = time.time()
        for cam in cameras:
            worker = active_inference_workers.get(cam["id"])
            if worker and worker.running and not worker.paused:
                last_frame = worker.stats.get("last_frame_time")
                # If frame was received in the last 10 seconds, consider the feed active
                if last_frame and (now - last_frame < 10):
                    processing_feeds += 1

        system_status['cameras'] = {
            'status': 'running' if processing_feeds > 0 else 'idle',
            'active_cameras': len(cameras),
            'processing_feeds': processing_feeds,
            'message': f"{processing_feeds} feeds being processed" if processing_feeds > 0 else "No active feeds"
        }
    except Exception as e:
        system_status['cameras'] = {
            'status': 'error',
            'message': f"Error: {str(e)}",
        }

    response = JSONResponse(system_status)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ============================================================
# Inference Management API Routes
# ============================================================
class WorkerStats:
    """Simple dict-compatible stats object."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@app.post("/api/inference/start/{camera_id}")
async def start_inference_for_camera(
    camera_id: int,
    username: str = Depends(get_current_user_from_cookie),
):
    """Start inference for a specific camera."""
    success = await _start_inference_for_camera_internal(camera_id)
    if success:
        return {"message": f"Started worker for camera {camera_id}."}
    else:
        raise HTTPException(status_code=400, detail="Failed to start inference worker.")


@app.post("/api/inference/stop/{camera_id}")
async def stop_inference_for_camera(
    camera_id: int,
    username: str = Depends(get_current_user_from_cookie),
):
    """Stop inference for a specific camera."""
    if camera_id not in active_inference_workers:
        return {"message": f"No worker running for camera {camera_id}."}

    await _stop_inference_for_camera_internal(camera_id)
    return {"message": f"Stopped worker for camera {camera_id}."}


@app.post("/api/inference/pause/{camera_id}")
async def pause_inference_for_camera(
    camera_id: int,
    username: str = Depends(get_current_user_from_cookie),
):
    """Pause inference for a specific camera."""
    if camera_id not in active_inference_workers:
        raise HTTPException(status_code=404, detail="Camera worker not found.")

    worker = active_inference_workers[camera_id]
    worker.pause()
    return {"message": f"Paused worker for camera {camera_id}."}


@app.post("/api/inference/resume/{camera_id}")
async def resume_inference_for_camera(
    camera_id: int,
    username: str = Depends(get_current_user_from_cookie),
):
    """Resume inference for a specific camera."""
    if camera_id not in active_inference_workers:
        raise HTTPException(status_code=404, detail="Camera worker not found.")

    worker = active_inference_workers[camera_id]
    worker.resume()
    return {"message": f"Resumed worker for camera {camera_id}."}


@app.get("/api/inference/stats/{camera_id}")
async def get_worker_stats(
    camera_id: int,
    username: str = Depends(get_current_user_from_cookie),
):
    """Get statistics for a specific worker."""
    if camera_id not in active_inference_workers:
        raise HTTPException(status_code=404, detail="Camera worker not found.")

    worker = active_inference_workers[camera_id]
    return worker.get_stats()


@app.get("/api/inference/health")
async def get_inference_health(
    username: str = Depends(get_current_user_from_cookie),
):
    """Get overall inference service health."""
    worker_stats_list = []
    for camera_id, worker in active_inference_workers.items():
        worker_stats_list.append(worker.get_stats())

    conn = get_db_connection()
    total_cameras = conn.execute("SELECT COUNT(*) as count FROM cameras").fetchone()["count"]
    conn.close()

    return {
        "status": "healthy" if len(active_inference_workers) > 0 else "idle",
        "active_workers": len(active_inference_workers),
        "total_cameras": total_cameras,
        "worker_stats": worker_stats_list,
    }


@app.get("/api/inference/workers")
async def list_workers(
    username: str = Depends(get_current_user_from_cookie),
):
    """List all active inference workers."""
    return {
        "active_workers": list(active_inference_workers.keys()),
        "max_workers": MAX_INFERENCE_WORKERS,
        "current_workers": len(active_inference_workers),
    }

# Backup API endpoints
@app.post("/api/backup")
async def create_backup(
    request: Request,
    username: str = Depends(get_current_user_from_cookie)
):
    """Create a backup of the database and configuration files"""
    try:
        # Create backups directory if it doesn't exist
        backups_dir = Path("backups")
        backups_dir.mkdir(exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"pawikan_backup_{timestamp}.zip"
        backup_path = backups_dir / backup_filename
        
        # Files to include in backup
        files_to_backup = [
            ".env",
            "pawikan.db",
            "README.md",
            "requirements.txt",
            "pyproject.toml"
        ]
        
        # Create zip file
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_backup:
                if os.path.exists(file_path):
                    zipf.write(file_path)
            
            # Add detections directory if it exists
            if os.path.exists("detections"):
                for root, dirs, files in os.walk("detections"):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, ".")
                        zipf.write(file_path, arc_name)
            
            # Add any other important directories
            important_dirs = ["deployments", "docs"]
            for dir_name in important_dirs:
                if os.path.exists(dir_name):
                    for root, dirs, files in os.walk(dir_name):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, ".")
                            zipf.write(file_path, arc_name)
        
        response = JSONResponse({
            "message": f"Backup created successfully: {backup_filename}",
            "filename": backup_filename
        })
        # Add cache control headers to prevent caching of sensitive API responses
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
        
    except Exception as e:
        response = JSONResponse({"error": f"Failed to create backup: {str(e)}"}, status_code=500)
        # Add cache control headers to prevent caching of sensitive API responses
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


@app.get("/api/backup/history")
async def get_backup_history(
    request: Request,
    username: str = Depends(get_current_user_from_cookie)
):
    """Get list of available backups"""
    try:
        backups_dir = Path("backups")
        if not backups_dir.exists():
            response = JSONResponse({"backups": []})
            # Add cache control headers to prevent caching of sensitive API responses
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        
        backup_files = list(backups_dir.glob("*.zip"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        backups = []
        for backup_file in backup_files[:10]:  # Limit to 10 most recent backups
            stat = backup_file.stat()
            backups.append({
                "filename": backup_file.name,
                "timestamp": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": format_file_size(stat.st_size)
            })
        
        response = JSONResponse({"backups": backups})
        # Add cache control headers to prevent caching of sensitive API responses
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
        
    except Exception as e:
        response = JSONResponse({"error": f"Failed to retrieve backup history: {str(e)}"}, status_code=500)
        # Add cache control headers to prevent caching of sensitive API responses
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"






