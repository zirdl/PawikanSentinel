from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import secrets
import logging

from ...core import storage
from ...core.database import get_db
from .. import auth

router = APIRouter()
templates = Jinja2Templates(directory="src/pawikan/dashboard/templates")
logger = logging.getLogger(__name__)

@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    csrf_token = secrets.token_hex(16)
    request.session["csrf_token"] = csrf_token
    logger.debug(f"GET /login: Generated CSRF token: {csrf_token}")
    logger.debug(f"GET /login: Session CSRF token set to: {request.session.get('csrf_token')}")
    return templates.TemplateResponse("login.html", {"request": request, "csrf_token": csrf_token})

@router.post("/login", response_class=HTMLResponse)
async def login(request: Request, db: Session = Depends(get_db), username: str = Form(...), password: str = Form(...), csrf_token: str = Form(...)):
    logger.debug(f"POST /login: Form CSRF token received: {csrf_token}")
    logger.debug(f"POST /login: Session CSRF token: {request.session.get('csrf_token')}")
    if csrf_token != request.session.get("csrf_token"):
        logger.warning(f"CSRF Mismatch: Form token '{csrf_token}' vs Session token '{request.session.get('csrf_token')}'")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")
    
    user = storage.get_user_by_username(db, username=username)
    if not user or not auth.verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    
    request.session["user"] = user.username
    del request.session["csrf_token"] # Clear CSRF token after successful login
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)

@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")
