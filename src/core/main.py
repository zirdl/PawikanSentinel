import os
import secrets
import logging
from fastapi import FastAPI, HTTPException, Request, Depends, Form, Cookie, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import WebSocket
from fastapi import WebSocket
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import httpx
from starlette.middleware.sessions import SessionMiddleware
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import sqlite3
import random
import shutil
import zipfile
from pathlib import Path
import glob
import asyncio
import threading

# --- Logging Setup ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# Global variable to store the main event loop when available
main_event_loop = None

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load environment variables
load_dotenv()

# Configuration
DATABASE_PATH = "pawikan.db"
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_hex(32))
CSRF_SECRET_KEY = os.getenv("CSRF_SECRET_KEY", secrets.token_hex(16))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global variable to store the background task
background_task = None

# Create FastAPI app without lifespan for now to fix session issues
# We'll start the broadcast thread separately

# Create FastAPI app
app = FastAPI(title="Pawikan Sentinel", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add middleware
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=1800)  # 30 minutes

# Create serializer for CSRF tokens
serializer = URLSafeTimedSerializer(CSRF_SECRET_KEY)

# Global variable to store the broadcast thread
broadcast_thread = None



import asyncio
import queue
import threading

# Thread-safe queue to receive detection broadcasts from inference threads  
detection_broadcast_queue = queue.Queue()

# Asyncio-compatible queue for the background task
async_detection_queue = asyncio.Queue()

# Define the detections directory
detections_dir = os.getenv("DETECTIONS_DIR", "detections")
# Ensure we're using an absolute path
if not os.path.isabs(detections_dir):
    detections_dir = os.path.abspath(detections_dir)

app.mount("/detections", StaticFiles(directory=detections_dir), name="detections")
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

# Import and initialize WebSocket manager after templates are created
from . import websocket_manager
from .websocket_manager import ConnectionManager

# Set the templates in the manager instance
manager = ConnectionManager(templates=templates)
# Update the global manager in the websocket_manager module
websocket_manager.manager = manager



# Include API routers
from src.api import auth, cameras, contacts, detections, analytics

app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(contacts.router)
app.include_router(detections.router)
app.include_router(analytics.router)

manager = ConnectionManager(templates)

# Database helper
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Create tables if they don't exist
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
    """Get detection data for chart visualization"""
    conn = get_db_connection()
    
    if period == "month":
        if month:
            # Get daily counts for a specific month
            data = conn.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM detections 
                WHERE strftime('%Y-%m', timestamp) = ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (month,)).fetchall()
        else:
            # Get daily counts for the current month
            data = conn.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM detections 
                WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """).fetchall()
    else:
        # Get monthly counts for all time
        data = conn.execute("""
            SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) as count 
            FROM detections 
            GROUP BY strftime('%Y-%m', timestamp)
            ORDER BY month
        """).fetchall()
    
    conn.close()
    return [dict(row) for row in data]

def get_recent_detections(limit=10):
    """Get recent detections with camera names"""
    conn = get_db_connection()
    detections = conn.execute("""
        SELECT d.id, d.timestamp, c.name as camera_name, d.class, d.confidence, d.image_path
        FROM detections d
        LEFT JOIN cameras c ON d.camera_id = c.id
        ORDER BY d.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(detection) for detection in detections]

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

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

# Authentication dependencies
async def get_current_user_from_cookie(access_token: str = Cookie(None)):
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    payload = decode_access_token(access_token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return username

async def get_current_user_for_pages(access_token: str = Cookie(None)):
    if not access_token:
        return None
    
    payload = decode_access_token(access_token)
    if payload is None:
        return None
    
    username: str = payload.get("sub")
    return username

# Startup event to create tables
@app.on_event("startup")
async def startup_event():
    # Store the main event loop
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    print(f"Main event loop stored in startup: {main_event_loop}, running: {main_event_loop.is_running()}")
    
    create_tables()
    
    # Ensure only the admin user exists - remove any non-admin users
    conn = get_db_connection()
    # Check if admin user exists
    user = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if not user:
        # Create default admin user if it doesn't exist
        hashed_password = get_password_hash("admin")
        conn.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            ("admin", hashed_password, "admin")
        )
        conn.commit()
    # Remove any non-admin users to enforce single-account policy
    conn.execute("DELETE FROM users WHERE username != ?", ("admin",))
    conn.commit()
    conn.close()
        
    # """ # Add some sample detection data for testing
    # detection_count = conn.execute("SELECT COUNT(*) as count FROM detections").fetchone()["count"]
    # if detection_count == 0:
        # Add sample detections for the past 30 days
        # from datetime import datetime, timedelta
        
        # for i in range(30):
            # date = datetime.now() - timedelta(days=i)
            # Add random number of detections per day (0-10)
            # for j in range(random.randint(0, 10)):
                # camera_id = random.randint(1, 3)  # Assuming we have up to 3 cameras
                # confidence = round(random.uniform(0.7, 0.99), 2)
                # conn.execute(
                    # "INSERT INTO detections (camera_id, timestamp, class, confidence) VALUES (?, ?, ?, ?)",
                    # (camera_id, date, "pawikan", confidence)
                # )
        # conn.commit()
    # conn.close() 
    # """

    # Start the broadcast processing thread
    def run_broadcast_processor():
        try:
            logger.info("Broadcast processor thread started and running...")
            
            # Small initial delay to ensure everything is properly initialized
            import time
            time.sleep(0.5)
            
            logger.debug("Broadcast processor: Starting main processing loop...")
            
            while True:
                try:
                    # Get a detection from the queue (blocking with timeout)
                    logger.debug("Broadcast processor: Waiting for detection from queue...")
                    detection_data = detection_broadcast_queue.get(timeout=2.0)
                    if detection_data is None:  # None is used as a signal to stop
                        logger.info("Broadcast processor thread stopping...")
                        break
                    
                    logger.debug(f"Broadcast processor: Processing detection {detection_data}")
                    
                    # Use the stored main event loop
                    global main_event_loop
                    if main_event_loop is not None:
                        logger.debug(f"Broadcast processor: Using event loop {main_event_loop}")
                        try:
                            # Create the coroutine for broadcasting
                            coro = manager.broadcast({
                                "type": "new_detection",
                                "detection": detection_data
                            })
                            
                            logger.debug(f"Broadcast processor: About to call run_coroutine_threadsafe")
                            
                            # Submit the coroutine to the stored event loop
                            future = asyncio.run_coroutine_threadsafe(coro, main_event_loop)
                            
                            # Wait for the result with a timeout to see if there are any errors
                            try:
                                result = future.result(timeout=5.0)  # 5 second timeout
                                logger.debug(f"Broadcast processor: Broadcast completed successfully, result: {result}")
                            except (asyncio.TimeoutError, Exception) as future_error:
                                logger.error(f"Broadcast processor: Error getting future result: {future_error}")
                                import traceback
                                traceback.print_exc()
                            
                        except Exception as broadcast_error:
                            logger.error(f"Error during broadcast: {broadcast_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.error("Main event loop not available in broadcast processor")
                        
                except queue.Empty:
                    # Timeout occurred, continue the loop
                    logger.debug("Broadcast processor: Queue timeout, continuing...")
                    continue
        except Exception as e:
            logger.error(f"Fatal error in broadcast processor thread: {e}")
            import traceback
            traceback.print_exc()
    
    # Start the broadcast processing thread
    global broadcast_thread
    print("Attempting to start broadcast processing thread...")
    broadcast_thread = threading.Thread(target=run_broadcast_processor, daemon=True)
    broadcast_thread.start()
    print(f"Background thread for processing detection broadcasts started: {broadcast_thread}")

# Routes
@app.on_event("shutdown")
async def shutdown_event():
    global broadcast_thread
    if broadcast_thread:
        # Put None to signal the thread to stop
        detection_broadcast_queue.put(None)
        broadcast_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
    print("Background thread for processing detection broadcasts stopped")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, username: str = Depends(get_current_user_for_pages)):
    if not username:
        return RedirectResponse(url="/login?next=/dashboard")
    
    csrf_token = generate_csrf_token(request)
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "csrf_token": csrf_token, 
        "username": username
    })

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, username: str = Depends(get_current_user_for_pages)):
    if not username:
        return RedirectResponse(url="/login?next=/settings")
    
    csrf_token = generate_csrf_token(request)
    return templates.TemplateResponse("settings.html", {
        "request": request, 
        "csrf_token": csrf_token, 
        "username": username
    })

# Detection API endpoints
@app.get("/api/detections/stats")
async def get_detection_stats_api(username: str = Depends(get_current_user_from_cookie)):
    return get_detection_stats()

@app.get("/api/detections/chart")
async def get_detection_chart_api(period: str = "month", month: str = None, username: str = Depends(get_current_user_from_cookie)):
    return get_detection_chart_data(period, month)

@app.get("/api/detections/recent")
async def get_recent_detections_api(username: str = Depends(get_current_user_from_cookie)):
    return get_recent_detections()

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
            # Create path for the static file mount point
            images.append({
                "filename": os.path.basename(image_file),
                "path": f"/detections/{os.path.relpath(image_file, detection_dir)}",
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
    
    return images

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    csrf_token = generate_csrf_token(request)
    registered = request.query_params.get("registered") == "true"
    next_url = request.query_params.get("next", "/dashboard")
    return templates.TemplateResponse("login.html", {
        "request": request, 
        "csrf_token": csrf_token, 
        "registered": registered, 
        "next_url": next_url
    })

@app.post("/login")
@limiter.limit("5/15minutes")
async def login_user(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    csrf_token = form.get("csrf_token")
    next_url = request.query_params.get("next", "/dashboard")
    
    # Verify CSRF token
    try:
        await verify_csrf_token(request)
    except HTTPException:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid CSRF token", 
            "csrf_token": generate_csrf_token(request),
            "next_url": next_url
        })
    
    # Authenticate user
    conn = get_db_connection()
    user = conn.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if not user or not verify_password(password, user["hashed_password"]):
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Invalid username or password", 
            "csrf_token": generate_csrf_token(request),
            "next_url": next_url
        })
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=access_token_expires
    )
    
    # Redirect with token cookie
    response = RedirectResponse(url=next_url, status_code=302)
    response.set_cookie(
        key="access_token", 
        value=access_token, 
        httponly=True, 
        samesite="lax", 
        secure=False  # Set to True in production with HTTPS
    )
    return response

# REMOVED: Registration routes to prevent creation of new accounts
# Only the admin account should exist

@app.post("/change-password")
async def change_password(request: Request, username: str = Depends(get_current_user_from_cookie)):
    form = await request.form()
    current_password = form.get("current_password")
    new_password = form.get("new_password")
    confirm_new_password = form.get("confirm_new_password")
    
    # Verify current password
    conn = get_db_connection()
    user = conn.execute("SELECT id, hashed_password FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if not user or not verify_password(current_password, user["hashed_password"]):
        return JSONResponse({"error": "Current password is incorrect"}, status_code=400)
    
    # Validate new password
    if not new_password or len(new_password) < 8:
        return JSONResponse({"error": "New password must be at least 8 characters long"}, status_code=400)
    
    if new_password != confirm_new_password:
        return JSONResponse({"error": "New passwords do not match"}, status_code=400)
    
    # Update password
    hashed_password = get_password_hash(new_password)
    conn = get_db_connection()
    conn.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (hashed_password, user["id"]))
    conn.commit()
    conn.close()
    
    return JSONResponse({"message": "Password changed successfully"})

@app.post("/change-username")
async def change_username(request: Request, current_username: str = Depends(get_current_user_from_cookie)):
    form = await request.form()
    current_password = form.get("current_password")
    new_username = form.get("new_username")
    
    # Verify current password
    conn = get_db_connection()
    user = conn.execute("SELECT id, hashed_password FROM users WHERE username = ?", (current_username,)).fetchone()
    
    if not user or not verify_password(current_password, user["hashed_password"]):
        conn.close()
        return JSONResponse({"error": "Current password is incorrect"}, status_code=400)
    
    # Validate new username
    if not new_username or len(new_username) < 3 or len(new_username) > 30:
        conn.close()
        return JSONResponse({"error": "Username must be between 3 and 30 characters"}, status_code=400)
    
    # Check if username is already taken (should not happen with single account but let's be safe)
    existing_user = conn.execute("SELECT id FROM users WHERE username = ? AND id != ?", (new_username, user["id"])).fetchone()
    if existing_user:
        conn.close()
        return JSONResponse({"error": "Username is already taken"}, status_code=400)
    
    # Update username
    conn.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user["id"]))
    conn.commit()
    conn.close()
    
    # Create new access token with new username
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_username}, 
        expires_delta=access_token_expires
    )
    
    response = JSONResponse({"message": "Username changed successfully"})
    response.set_cookie(
        key="access_token", 
        value=access_token, 
        httponly=True, 
        samesite="lax", 
        secure=False
    )
    return response

@app.post("/logout")
async def logout_user(request: Request):
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("access_token", httponly=True, samesite="lax")
    request.session.pop("csrf_token", None)
    return response

# Routes for camera management UI
@app.get("/cameras/list", response_class=HTMLResponse)
async def get_camera_list(request: Request, username: str = Depends(get_current_user_from_cookie)):
    conn = get_db_connection()
    cameras = conn.execute("SELECT id, name, rtsp_url, active FROM cameras").fetchall()
    conn.close()
    return templates.TemplateResponse("_camera_list.html", {
        "request": request, 
        "cameras": [dict(camera) for camera in cameras]
    })

@app.get("/cameras/form", response_class=HTMLResponse)
async def get_camera_form(request: Request, username: str = Depends(get_current_user_from_cookie)):
    csrf_token = generate_csrf_token(request)
    return templates.TemplateResponse("_camera_form.html", {
        "request": request, 
        "csrf_token": csrf_token
    })

@app.get("/cameras/edit-form/{camera_id}", response_class=HTMLResponse)
async def get_edit_camera_form(request: Request, camera_id: int, username: str = Depends(get_current_user_from_cookie)):
    conn = get_db_connection()
    camera = conn.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    conn.close()
    
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return templates.TemplateResponse("_camera_form.html", {
        "request": request, 
        "camera": dict(camera), 
        "csrf_token": generate_csrf_token(request)
    })

@app.post("/cameras/add-or-update", response_class=HTMLResponse)
async def add_or_update_camera(
    request: Request,
    username: str = Depends(get_current_user_from_cookie),
    csrf_token: str = Depends(verify_csrf_token)
):
    form = await request.form()
    camera_id = form.get("camera_id")
    name = form.get("name")
    rtsp_url = form.get("rtsp_url")
    active = form.get("active") == "on"
    
    conn = get_db_connection()
    if camera_id:
        # Update existing camera
        conn.execute(
            "UPDATE cameras SET name = ?, rtsp_url = ?, active = ? WHERE id = ?",
            (name, rtsp_url, active, camera_id)
        )
        message = f"Camera '{name}' updated successfully!"
    else:
        # Add new camera
        conn.execute(
            "INSERT INTO cameras (name, rtsp_url, active) VALUES (?, ?, ?)",
            (name, rtsp_url, active)
        )
        message = f"Camera '{name}' added successfully!"
    
    conn.commit()
    conn.close()
    
    # Return updated camera list
    conn = get_db_connection()
    cameras = conn.execute("SELECT id, name, rtsp_url, active FROM cameras").fetchall()
    conn.close()
    
    response = templates.TemplateResponse("_camera_list.html", {
        "request": request, 
        "cameras": [dict(camera) for camera in cameras],
        "message": message
    })
    response.headers["HX-Trigger"] = "close-modal"
    return response

@app.delete("/api/cameras/{camera_id}", response_class=HTMLResponse)
async def delete_camera_htmx(
    request: Request,
    camera_id: int,
    username: str = Depends(get_current_user_from_cookie),
    csrf_token: str = Depends(verify_csrf_token)
):
    conn = get_db_connection()
    conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
    conn.commit()
    conn.close()
    
    # Return updated camera list
    conn = get_db_connection()
    cameras = conn.execute("SELECT id, name, rtsp_url, active FROM cameras").fetchall()
    conn.close()
    
    return templates.TemplateResponse("_camera_list.html", {
        "request": request, 
        "cameras": [dict(camera) for camera in cameras],
        "message": "Camera deleted successfully!"
    })

# Routes for contact management UI
@app.get("/contacts/list", response_class=HTMLResponse)
async def get_contact_list(request: Request, username: str = Depends(get_current_user_from_cookie)):
    conn = get_db_connection()
    contacts = conn.execute("SELECT id, name, phone FROM contacts").fetchall()
    conn.close()
    return templates.TemplateResponse("_contact_list.html", {
        "request": request, 
        "contacts": [dict(contact) for contact in contacts]
    })

@app.get("/contacts/form", response_class=HTMLResponse)
async def get_contact_form(request: Request, username: str = Depends(get_current_user_from_cookie)):
    csrf_token = generate_csrf_token(request)
    return templates.TemplateResponse("_contact_form.html", {
        "request": request, 
        "csrf_token": csrf_token
    })

@app.get("/contacts/edit-form/{contact_id}", response_class=HTMLResponse)
async def get_edit_contact_form(request: Request, contact_id: int, username: str = Depends(get_current_user_from_cookie)):
    conn = get_db_connection()
    contact = conn.execute("SELECT id, name, phone FROM contacts WHERE id = ?", (contact_id,)).fetchone()
    conn.close()
    
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    return templates.TemplateResponse("_contact_form.html", {
        "request": request, 
        "contact": dict(contact), 
        "csrf_token": generate_csrf_token(request)
    })

@app.post("/contacts/add-or-update", response_class=HTMLResponse)
async def add_or_update_contact(
    request: Request,
    username: str = Depends(get_current_user_from_cookie),
    csrf_token: str = Depends(verify_csrf_token)
):
    form = await request.form()
    contact_id = form.get("contact_id")
    name = form.get("name")
    phone = form.get("phone")
    
    conn = get_db_connection()
    if contact_id:
        # Update existing contact
        conn.execute(
            "UPDATE contacts SET name = ?, phone = ? WHERE id = ?",
            (name, phone, contact_id)
        )
        message = f"Contact '{name}' updated successfully!"
    else:
        # Add new contact
        conn.execute(
            "INSERT INTO contacts (name, phone) VALUES (?, ?)",
            (name, phone)
        )
        message = f"Contact '{name}' added successfully!"
    
    conn.commit()
    conn.close()
    
    # Return updated contact list
    conn = get_db_connection()
    contacts = conn.execute("SELECT id, name, phone FROM contacts").fetchall()
    conn.close()
    
    response = templates.TemplateResponse("_contact_list.html", {
        "request": request, 
        "contacts": [dict(contact) for contact in contacts],
        "message": message
    })
    response.headers["HX-Trigger"] = "close-modal"
    return response

@app.delete("/contacts/{contact_id}", response_class=HTMLResponse)
async def delete_contact_htmx(
    request: Request,
    contact_id: int,
    username: str = Depends(get_current_user_from_cookie),
    csrf_token: str = Depends(verify_csrf_token)
):
    conn = get_db_connection()
    conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
    conn.commit()
    conn.close()
    
    # Return updated contact list
    conn = get_db_connection()
    contacts = conn.execute("SELECT id, name, phone FROM contacts").fetchall()
    conn.close()
    
    return templates.TemplateResponse("_contact_list.html", {
        "request": request, 
        "contacts": [dict(contact) for contact in contacts],
        "message": "Contact deleted successfully!"
    })


# Configuration API endpoints
@app.get("/api/config")
async def get_config(username: str = Depends(get_current_user_from_cookie)):
    """Get current configuration values"""
    config = {
        "confidence_threshold": os.getenv("CONFIDENCE_THRESHOLD", "80"),
        "frame_skip": os.getenv("FRAME_SKIP", "20"),
        "sms_cooldown": os.getenv("SMS_NOTIFICATION_COOLDOWN", "10")
    }
    return config


@app.post("/api/config")
async def update_config(
    request: Request,
    username: str = Depends(get_current_user_from_cookie),
    csrf_token: str = Depends(verify_csrf_token)
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


# Backup API endpoints
@app.post("/api/backup")
async def create_backup(
    request: Request,
    username: str = Depends(get_current_user_from_cookie),
    csrf_token: str = Depends(verify_csrf_token)
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
        
        return JSONResponse({
            "message": f"Backup created successfully: {backup_filename}",
            "filename": backup_filename
        })
        
    except Exception as e:
        return JSONResponse({"error": f"Failed to create backup: {str(e)}"}, status_code=500)


@app.get("/api/backup/history")
async def get_backup_history(
    request: Request,
    username: str = Depends(get_current_user_from_cookie)
):
    """Get list of available backups"""
    try:
        backups_dir = Path("backups")
        if not backups_dir.exists():
            return JSONResponse({"backups": []})
        
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
        
        return JSONResponse({"backups": backups})
        
    except Exception as e:
        return JSONResponse({"error": f"Failed to retrieve backup history: {str(e)}"}, status_code=500)


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        print(f"WebSocket connected. Active connections: {len(manager.active_connections)}")
        # Send a welcome message to confirm connection
        await websocket.send_json({"type": "connected", "message": "WebSocket connection established"})
        
        # Keep the connection alive to receive broadcasts
        # We just need to keep this connection open, not receive any data from the client
        while True:
            # Sleep indefinitely to keep the connection open
            # WebSocket will timeout automatically if the connection is lost
            try:
                await asyncio.sleep(3600)  # Sleep for 1 hour, will be interrupted if connection closes
            except asyncio.CancelledError:
                # This happens when the connection is closed or server shuts down
                print("WebSocket connection task cancelled (normal on disconnect)")
                break
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        # Only try to disconnect if the connection was actually established
        try:
            manager.disconnect(websocket)
        except Exception as disconnect_error:
            print(f"Error during disconnect: {disconnect_error}")
        print(f"WebSocket disconnected. Active connections: {len(manager.active_connections)}")
