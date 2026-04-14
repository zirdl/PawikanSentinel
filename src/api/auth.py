import re
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..core.database import get_db_connection
from ..core.models import Token, User, UserCreate
from ..core.utils import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)

router = APIRouter()

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, username, hashed_password, role FROM users WHERE username = ?",
        (username,),
    )
    user = c.fetchone()
    conn.close()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return User(**user)


@router.post("/auth/login", response_model=Token)
@limiter.limit("5/15minutes")
async def login_for_access_token(
    request: Request, form_data: OAuth2PasswordRequestForm = Depends()
):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, username, hashed_password, role FROM users WHERE username = ? COLLATE NOCASE",
        (form_data.username,),
    )
    user = c.fetchone()
    conn.close()

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        # Use a generic error message to prevent username enumeration
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    response = JSONResponse({"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,  # Set to True in production
        path="/",
    )
    # Add cache control headers to prevent caching of authentication responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


@router.post("/auth/change-password")
async def change_password(
    data: PasswordChange, current_user: User = Depends(get_current_user)
):
    # Validate new password strength
    if len(data.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters long",
        )

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT hashed_password FROM users WHERE id = ?", (current_user.id,))
    user_data = c.fetchone()
    conn.close()

    if not verify_password(data.old_password, user_data["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect old password"
        )

    hashed_new_password = get_password_hash(data.new_password)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET hashed_password = ? WHERE id = ?",
        (hashed_new_password, current_user.id),
    )
    conn.commit()
    conn.close()

    response = JSONResponse({"message": "Password changed successfully"})
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@router.post("/auth/change-username")
async def change_username(
    new_username: str,
    current_password: str,
    current_user: User = Depends(get_current_user),
):
    # Validate new username
    if not new_username or not new_username.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username cannot be empty"
        )
    if len(new_username.strip()) < 3 or len(new_username.strip()) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be between 3 and 30 characters",
        )

    # Verify password
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT hashed_password FROM users WHERE id = ?", (current_user.id,))
    user_data = c.fetchone()

    if not verify_password(current_password, user_data["hashed_password"]):
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect password"
        )

    # Check if new username is already taken (should not happen with single account but let's be safe)
    c.execute(
        "SELECT id FROM users WHERE username = ? COLLATE NOCASE AND id != ?",
        (new_username, current_user.id),
    )
    existing_user = c.fetchone()
    if existing_user:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
        )

    # Update username
    c.execute(
        "UPDATE users SET username = ? WHERE id = ?", (new_username, current_user.id)
    )
    conn.commit()
    conn.close()

    response = JSONResponse(
        {
            "message": "Username changed successfully. Please log in again with your new username."
        }
    )
    # Add cache control headers to prevent caching of sensitive API responses
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@router.get("/api/auth/me")
async def get_me(request: Request):
    # Try to get from cookie (used by frontend)
    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_access_token(access_token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {"username": username, "role": "admin"}
