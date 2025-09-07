
import sqlite3
import os

DB_FILE = os.getenv("DATABASE_PATH", "pawikan.db")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Contacts table
    c.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Cameras table
    c.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            active BOOLEAN DEFAULT TRUE
        )
    """)

    # Detections table
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            class TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            FOREIGN KEY (camera_id) REFERENCES cameras (id)
        )
    """)

    # Indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections (timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()
