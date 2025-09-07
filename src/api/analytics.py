from fastapi import APIRouter, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.database import get_db_connection

router = APIRouter()

@router.get("/api/analytics/summary")
async def get_detection_summary() -> Dict[str, Any]:
    """Return total detections and today's detections."""
    conn = get_db_connection()
    c = conn.cursor()
    # Total detections
    c.execute("SELECT COUNT(*) as total_count FROM detections")
    total_row = c.fetchone()
    total_count = total_row["total_count"] if total_row and "total_count" in total_row.keys() else (total_row[0] if total_row else 0)

    # Today's detections (from midnight)
    start_of_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    c.execute("SELECT COUNT(*) as today_count FROM detections WHERE timestamp >= ?", (start_of_today,))
    today_row = c.fetchone()
    today_count = today_row["today_count"] if today_row and "today_count" in today_row.keys() else (today_row[0] if today_row else 0)

    conn.close()
    return {"total": total_count, "today": today_count}

@router.get("/api/analytics/counts")
async def get_detection_counts(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    camera_id: Optional[int] = Query(None)
) -> Dict[str, Any]:
    conn = get_db_connection()
    c = conn.cursor()

    query = "SELECT class, COUNT(*) as count FROM detections WHERE 1=1"
    params = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)

    query += " GROUP BY class"
    c.execute(query, params)
    counts = c.fetchall()
    conn.close()

    # Convert sqlite3.Row objects to dictionaries
    return {"counts": [dict(row) for row in counts]}

@router.get("/api/analytics/timeline")
async def get_detection_timeline(
    interval: str = Query("day", regex="^(hour|day|week|month)$", description="Aggregation interval"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    camera_id: Optional[int] = Query(None)
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    c = conn.cursor()

    # Determine the SQLite date/time function based on interval
    if interval == "hour":
        date_format = "%Y-%m-%d %H:00:00"
    elif interval == "day":
        date_format = "%Y-%m-%d 00:00:00"
    elif interval == "week":
        # SQLite doesn't have a direct 'week' function, so we'll group by year and week number
        # This will return results like 'YYYY-WW'
        date_format = "%Y-%W"
    elif interval == "month":
        date_format = "%Y-%m-01 00:00:00"
    else:
        raise HTTPException(status_code=400, detail="Invalid interval. Must be hour, day, week, or month.")

    if interval == "week":
        time_group_col = "STRFTIME('%Y-%W', timestamp)"
    else:
        time_group_col = f"STRFTIME('{date_format}', timestamp)"

    query = f"SELECT {time_group_col} as time_group, class, COUNT(*) as count FROM detections WHERE 1=1"
    params = []

    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)

    query += f" GROUP BY time_group, class ORDER BY time_group"
    c.execute(query, params)
    timeline_data = c.fetchall()
    conn.close()

    # Format the output for consistency and convert sqlite3.Row to dict
    formatted_timeline = []
    for row in timeline_data:
        formatted_timeline.append(dict(row)) # Convert to dict

    return formatted_timeline