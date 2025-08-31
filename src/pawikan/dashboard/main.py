import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from pawikan.core.config import settings as config_settings
from .routers import auth, dashboard, settings, contacts

app = FastAPI()

# Add session middleware
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "super-secret-dev-key-please-change")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

# Mount static files
app.mount("/static", StaticFiles(directory="src/pawikan/dashboard/static"), name="static")

# Mount gallery directory from config
gallery_dir = os.path.expanduser(config_settings['gallery']['dir'])
if not os.path.exists(gallery_dir):
    os.makedirs(gallery_dir)
app.mount("/gallery-images", StaticFiles(directory=gallery_dir), name="gallery-images")

# Include routers
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(settings.router)
app.include_router(contacts.router)

@app.get("/")
def read_root(request: Request):
    if "user" in request.session:
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")
