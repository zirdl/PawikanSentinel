import asyncio
from fastapi import WebSocket
from fastapi.templating import Jinja2Templates
import queue
import threading
import logging
import os

# --- Logging Setup ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# Thread-safe queue to receive detection broadcasts from inference threads  
detection_broadcast_queue = queue.Queue()

# Asyncio-compatible queue for the background task
async_detection_queue = asyncio.Queue()

# WebSocket Manager for real-time updates
class ConnectionManager:
    def __init__(self, templates: Jinja2Templates = None):
        self.active_connections: list[WebSocket] = []
        self.templates = templates

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            # Connection already removed
            pass

    async def broadcast(self, message: dict):
        # Reduced logging to prevent verbose debug messages
        if not self.active_connections:
            return  # No connections to broadcast to, just return silently

        # Prepare the full message with all necessary data
        full_message = {"type": message.get("type"), "data": message.get("detection"), "toast_html": ""}

        # Render the toast HTML if it's a new detection message
        if message.get("type") == "new_detection" and "detection" in message and self.templates:
            detection_data = message["detection"]
            # Create a dummy request object for Jinja2 rendering
            # This is a workaround as Jinja2Templates.TemplateResponse expects a request
            # For simple rendering without request-specific context, a minimal object suffices
            class DummyRequest:
                def __init__(self):
                    self.scope = {"type": "http"}
                
                async def form(self):
                    return {}

            try:
                template_context = {"request": DummyRequest(), "detection": detection_data}
                toast_html = self.templates.get_template("_new_detection_toast.html").render(template_context)
                full_message["toast_html"] = toast_html
            except Exception as e:
                logger.error(f"Error rendering toast template: {e}")  # Only log errors, not normal operations
                import traceback
                traceback.print_exc()

        # Create a copy of the connections list to avoid modification during iteration
        active_connections_copy = self.active_connections.copy()
        
        for i, connection in enumerate(active_connections_copy):
            try:
                await connection.send_json(full_message)
            except Exception as e:
                # Only log connection errors, not successful operations
                logger.error(f"WebSocket error sending to connection: {e}")
                import traceback
                traceback.print_exc()
                # Safely remove the disconnected connection
                try:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
                except ValueError:
                    # Connection already removed, ignore
                    pass

# Global manager instance - will be initialized in main.py after templates are created
manager = None

# Initialize the queue and async queue at module level
detection_broadcast_queue = queue.Queue()
async_detection_queue = asyncio.Queue()