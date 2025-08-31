from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import secrets

from ...core import storage
from ...core.database import get_db
from .. import auth

router = APIRouter()
templates = Jinja2Templates(directory="src/pawikan/dashboard/templates")

@router.get("/settings", response_class=HTMLResponse)
async def get_settings(request: Request, db: Session = Depends(get_db)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")

    current_username = request.session["user"]
    
    # Fetch current notification settings
    notify_enabled = storage.get_setting(db, "notify_enabled")
    notify_provider = storage.get_setting(db, "notify_provider")
    notification_cooldown_seconds = storage.get_setting(db, "notification_cooldown_seconds")
    twilio_account_sid = storage.get_setting(db, "twilio_account_sid")
    twilio_auth_token = storage.get_setting(db, "twilio_auth_token")
    twilio_from_number = storage.get_setting(db, "twilio_from_number")

    csrf_token = secrets.token_hex(16)
    request.session["csrf_token"] = csrf_token

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "current_user": {"username": current_username},
        "csrf_token": csrf_token,
        "notify_enabled": notify_enabled.value if notify_enabled else "false",
        "notify_provider": notify_provider.value if notify_provider else "none",
        "notification_cooldown_seconds": notification_cooldown_seconds.value if notification_cooldown_seconds else "1200",
        "twilio_account_sid": twilio_account_sid.value if twilio_account_sid else "",
        "twilio_auth_token": twilio_auth_token.value if twilio_auth_token else "",
        "twilio_from_number": twilio_from_number.value if twilio_from_number else "",
        "message": request.session.pop("message", None), # Pop messages for one-time display
        "error": request.session.pop("error", None),
    })

@router.post("/settings/update-user", response_class=HTMLResponse)
async def update_user(request: Request, db: Session = Depends(get_db), 
                      new_username: str = Form(None), 
                      current_password_user: str = Form(...), 
                      csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    current_username = request.session["user"]
    user_obj = storage.get_user_by_username(db, username=current_username)

    if not user_obj or not auth.verify_password(current_password_user, user_obj.hashed_password):
        request.session["error"] = "Incorrect current password."
        return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)

    if new_username and new_username != current_username:
        existing_user = storage.get_user_by_username(db, username=new_username)
        if existing_user:
            request.session["error"] = "Username already taken."
            return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)
        
        user_obj.username = new_username
        db.add(user_obj)
        db.commit()
        db.refresh(user_obj)
        request.session["user"] = new_username # Update session with new username
        request.session["message"] = "Username updated successfully."
    else:
        request.session["error"] = "No new username provided or username is the same."

    return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/settings/update-password", response_class=HTMLResponse)
async def update_password(request: Request, db: Session = Depends(get_db), 
                          current_password_pass: str = Form(...), 
                          new_password: str = Form(...), 
                          confirm_new_password: str = Form(...), 
                          csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    current_username = request.session["user"]
    user_obj = storage.get_user_by_username(db, username=current_username)

    if not user_obj or not auth.verify_password(current_password_pass, user_obj.hashed_password):
        request.session["error"] = "Incorrect current password."
        return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)

    if new_password != confirm_new_password:
        request.session["error"] = "New password and confirmation do not match."
        return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)
    
    if len(new_password) < 8: # Example: Minimum password length
        request.session["error"] = "New password must be at least 8 characters long."
        return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)

    user_obj.hashed_password = auth.get_password_hash(new_password)
    db.add(user_obj)
    db.commit()
    db.refresh(user_obj)
    request.session["message"] = "Password updated successfully."
    return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/settings/update-notifications", response_class=HTMLResponse)
async def update_notifications(request: Request, db: Session = Depends(get_db),
                               notify_enabled: str = Form(...),
                               notify_provider: str = Form(...),
                               notification_cooldown_seconds: int = Form(...),
                               twilio_account_sid: str = Form(None),
                               twilio_auth_token: str = Form(None),
                               twilio_from_number: str = Form(None),
                               csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    try:
        storage.update_setting(db, "notify_enabled", notify_enabled)
        storage.update_setting(db, "notify_provider", notify_provider)
        storage.update_setting(db, "notification_cooldown_seconds", str(notification_cooldown_seconds))

        if notify_provider == "twilio":
            storage.update_setting(db, "twilio_account_sid", twilio_account_sid or "")
            storage.update_setting(db, "twilio_auth_token", twilio_auth_token or "")
            storage.update_setting(db, "twilio_from_number", twilio_from_number or "")
        else:
            # Clear Twilio settings if provider is not Twilio
            storage.update_setting(db, "twilio_account_sid", "")
            storage.update_setting(db, "twilio_auth_token", "")
            storage.update_setting(db, "twilio_from_number", "")
        
        request.session["message"] = "Notification settings updated successfully."
    except Exception as e:
        request.session["error"] = f"Error updating notification settings: {e}"

    return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)