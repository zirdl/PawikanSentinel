from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from collections import defaultdict
import json
import secrets
import logging
import os
from pathlib import Path

from ...core import storage, schemas, config
from ...core.database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="src/pawikan/dashboard/templates")
logger = logging.getLogger(__name__)

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")

    # Fetch real data from the database
    events = storage.get_detection_events(db, limit=1000) # Get last 1000 events

    # Process data for the chart (e.g., detections per hour)
    detections_per_hour = defaultdict(int)
    for event in events:
        hour = event.timestamp.strftime("%Y-%m-%d %H:00")
        detections_per_hour[hour] += event.detection_count

    # Sort by hour for a clean chart
    sorted_hours = sorted(detections_per_hour.keys())
    
    chart_data = {
        "labels": sorted_hours,
        "values": [detections_per_hour[hour] for hour in sorted_hours]
    }

    # Fetch gallery images
    gallery_dir = Path(config.settings["gallery"]["dir"])
    gallery_images = []
    if gallery_dir.exists():
        for f in os.listdir(gallery_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                gallery_images.append(f)
        gallery_images.sort(reverse=True) # Newest first
    logger.debug(f"Gallery images found: {gallery_images}")

    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "current_user": {"username": request.session["user"]},
        "detection_data": json.dumps(chart_data), # Pass as a JSON string
        "gallery_images": gallery_images
    })

@router.get("/contacts", response_class=HTMLResponse)
async def contacts(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("contacts.html", {"request": request, "current_user": {"username": request.session["user"]}})

@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request, db: Session = Depends(get_db)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    
    csrf_token = secrets.token_hex(16)
    request.session["csrf_token"] = csrf_token
    logger.debug(f"GET /settings: Generated CSRF token: {csrf_token}")
    logger.debug(f"GET /settings: Session CSRF token set to: {request.session.get('csrf_token')}")
    
    # Fetch current settings to display in the form
    # For now, we just need the username
    current_user = {"username": request.session["user"]}
    
    return templates.TemplateResponse("settings.html", {
        "request": request, 
        "current_user": current_user,
        "csrf_token": csrf_token
    })

@router.post("/settings", response_class=HTMLResponse)
async def update_settings(request: Request, db: Session = Depends(get_db), new_username: str = Form(...), csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")

    if csrf_token != request.session.get("csrf_token"):
        logger.warning(f"CSRF Mismatch: Form token '{csrf_token}' vs Session token '{request.session.get('csrf_token')}'")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    # In a real app, you would have a proper user update function
    # For now, we'll just update the session as a demonstration
    # Note: This does NOT update the database, only the session.
    # A proper implementation would require updating the User model.
    
    old_username = request.session["user"]
    # This is where you would call a function like:
    # storage.update_username(db, old_username, new_username)
    
    # For now, just update the session
    request.session["user"] = new_username
    
    # Clear the CSRF token after use
    if "csrf_token" in request.session:
        del request.session["csrf_token"]

    # Redirect back to settings with a success message (or to the dashboard)
    return RedirectResponse(url="/settings?success=true", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/live-feed", response_class=HTMLResponse)
async def live_feed(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("live_feed.html", {"request": request, "current_user": {"username": request.session["user"]}})
