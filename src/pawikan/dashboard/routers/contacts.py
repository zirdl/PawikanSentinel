from fastapi import APIRouter, Depends, Request, Form, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import secrets

from ....core import storage
from ....core.database import get_db
from ....core import schemas

router = APIRouter()
templates = Jinja2Templates(directory="src/pawikan/dashboard/templates")

@router.get("/contacts", response_class=HTMLResponse)
async def get_contacts(request: Request, db: Session = Depends(get_db)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")

    contacts = storage.get_all_contacts(db)
    csrf_token = secrets.token_hex(16)
    request.session["csrf_token"] = csrf_token

    return templates.TemplateResponse("contacts.html", {
        "request": request,
        "contacts": contacts,
        "csrf_token": csrf_token,
        "message": request.session.pop("message", None),
        "error": request.session.pop("error", None),
    })

@router.post("/contacts/add", response_class=HTMLResponse)
async def add_contact(request: Request, db: Session = Depends(get_db),
                      name: str = Form(...),
                      phone_number: str = Form(...),
                      csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    try:
        contact = schemas.ContactCreate(name=name, phone_number=phone_number)
        storage.create_contact(db, contact)
        request.session["message"] = "Contact added successfully."
    except Exception as e:
        request.session["error"] = f"Error adding contact: {e}"

    return RedirectResponse(url="/contacts", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/contacts/edit", response_class=HTMLResponse)
async def edit_contact(request: Request, db: Session = Depends(get_db),
                       contact_id: int = Form(...),
                       name: str = Form(...),
                       phone_number: str = Form(...),
                       csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    try:
        contact = schemas.ContactUpdate(name=name, phone_number=phone_number)
        storage.update_contact(db, contact_id, contact)
        request.session["message"] = "Contact updated successfully."
    except Exception as e:
        request.session["error"] = f"Error updating contact: {e}"

    return RedirectResponse(url="/contacts", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/contacts/delete", response_class=HTMLResponse)
async def delete_contact(request: Request, db: Session = Depends(get_db),
                         contact_id: int = Form(...),
                         csrf_token: str = Form(...)):
    if "user" not in request.session:
        return RedirectResponse(url="/login")
    if csrf_token != request.session.get("csrf_token"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF token mismatch")

    try:
        storage.delete_contact(db, contact_id)
        request.session["message"] = "Contact deleted successfully."
    except Exception as e:
        request.session["error"] = f"Error deleting contact: {e}"

    return RedirectResponse(url="/contacts", status_code=status.HTTP_303_SEE_OTHER)
