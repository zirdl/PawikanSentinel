from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from .routers import auth, dashboard, settings

app = FastAPI()

# Add session middleware
import os

# Add session middleware
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "super-secret-dev-key-please-change")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

# Mount static files
app.mount("/static", StaticFiles(directory="src/pawikan/dashboard/static"), name="static")

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
