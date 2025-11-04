from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime

from ..core.database import get_db_connection
from ..core.models import Detection, DetectionBase, DetectionOut

router = APIRouter()

@router.post("/api/detections", response_model=DetectionOut)
async def create_detection(detection: DetectionBase):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO detections (camera_id, timestamp, class, confidence, image_path) VALUES (?, ?, ?, ?, ?)",
              (detection.camera_id, detection.timestamp, detection._class, detection.confidence, detection.image_path))
    conn.commit()
    new_detection_id = c.lastrowid
    
    # Get the newly created detection with camera name
    c.execute("""
        SELECT d.id, d.camera_id, c.name as camera_name, d.timestamp, d.class, d.confidence, d.image_path 
        FROM detections d
        JOIN cameras c ON d.camera_id = c.id
        WHERE d.id = ?
    """, (new_detection_id,))
    new_detection = c.fetchone()
    conn.close()
    
    # For the response, return only the fields that match DetectionOut model
    detection_dict = dict(new_detection)
    response_dict = {
        'id': detection_dict['id'],
        'camera_id': detection_dict['camera_id'],
        'timestamp': detection_dict['timestamp'],
        '_class': detection_dict['class'],  # Pydantic maps _class to class field
        'confidence': detection_dict['confidence'],
        'image_path': detection_dict['image_path'],
        'camera_name': detection_dict['camera_name']
    }
    
    return response_dict

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