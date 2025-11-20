import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the main application setup from the Vercel-compatible file
from src.core.main_for_vercel import app

# Vercel expects the application object to be named 'app'

# Mount static files for Vercel deployment
# Mount static assets
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Define the detections directory - for serverless, we can't persist files
detections_dir = os.getenv("DETECTIONS_DIR", "detections")
# For Vercel serverless functions, we should skip mounting detections dir
# since files can't be persisted between requests. The app already handles this with in-memory DB.
# Only mount if the directory exists and is accessible (for local dev)
if os.path.exists(detections_dir):
    try:
        app.mount("/detections", StaticFiles(directory=detections_dir), name="detections")
    except:
        # If mounting fails (could be due to permissions or other issues in serverless env)
        # we simply don't mount, which is fine since detection images in serverless env
        # would need to be served differently (e.g., from a CDN or database)
        pass