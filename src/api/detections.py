from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from ..core.database import get_db_connection
from ..core.models import Detection, DetectionBase, DetectionOut

router = APIRouter()

@router.post("/api/detections", response_model=Detection)
async def create_detection(detection: DetectionBase):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO detections (camera_id, timestamp, class, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
              (detection.camera_id, detection.timestamp, detection._class, detection.confidence, detection.image_path))
    conn.commit()
    new_detection_id = c.lastrowid
    c.execute("SELECT id, camera_id, timestamp, class, confidence, image_path FROM detections WHERE id = ?", (new_detection_id,))
    new_detection = c.fetchone()
    conn.close()
    return dict(new_detection) # Convert to dict

@router.get("/api/detections", response_model=List[DetectionOut])
async def get_detections(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    camera_id: Optional[int] = Query(None)
):
    conn = get_db_connection()
    c = conn.cursor()
    query = """SELECT d.id, d.camera_id, c.name as camera_name, d.timestamp, d.class, d.confidence, d.image_path 
               FROM detections d
               JOIN cameras c ON d.camera_id = c.id
               WHERE 1=1"""
    params = []

    if start_date:
        query += " AND d.timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND d.timestamp <= ?"
        params.append(end_date)
    if camera_id:
        query += " AND d.camera_id = ?"
        params.append(camera_id)
    
    query += " ORDER BY d.timestamp DESC"

    c.execute(query, params)
    detections = c.fetchall()
    conn.close()
    return [dict(detection) for detection in detections] # Convert to list of dicts