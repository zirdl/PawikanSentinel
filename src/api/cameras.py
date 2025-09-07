
from fastapi import APIRouter, HTTPException
from typing import List

from ..core.database import get_db_connection
from ..core.models import Camera, CameraBase

router = APIRouter()

@router.post("/api/cameras", response_model=Camera)
async def create_camera(camera: CameraBase):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO cameras (name, rtsp_url, active) VALUES (?, ?, ?)",
              (camera.name, camera.rtsp_url, camera.active))
    conn.commit()
    new_camera_id = c.lastrowid
    conn.close()
    return {**camera.dict(), "id": new_camera_id}

@router.get("/api/cameras", response_model=List[Camera])
async def get_cameras():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, rtsp_url, active FROM cameras")
    cameras = c.fetchall()
    conn.close()
    # Explicitly convert sqlite3.Row objects to dictionaries
    return [dict(camera) for camera in cameras]

@router.get("/api/cameras/{camera_id}", response_model=Camera)
async def get_camera(camera_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, rtsp_url, active FROM cameras WHERE id = ?", (camera_id,))
    camera = c.fetchone()
    conn.close()
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    # Explicitly convert sqlite3.Row object to a dictionary
    return dict(camera)

@router.put("/api/cameras/{camera_id}", response_model=Camera)
async def update_camera(camera_id: int, camera: CameraBase):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE cameras SET name = ?, rtsp_url = ?, active = ? WHERE id = ?",
              (camera.name, camera.rtsp_url, camera.active, camera_id))
    conn.commit()
    conn.close()
    return {**camera.dict(), "id": camera_id}

@router.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
    conn.commit()
    conn.close()
    return {"message": "Camera deleted successfully"}
