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