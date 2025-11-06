
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# JWT handling
import os
import secrets
import logging

# Setup logging
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    logger.critical("SECRET_KEY not set in .env. Generating a temporary key, but this is NOT secure for production!")
    SECRET_KEY = secrets.token_urlsafe(32)  # Generate a secure random key
elif SECRET_KEY == "0_y-2-1-3-5-7-9-b-d-f-h-j-l-n-p-r-t-v-x-z-A-C-E-G-I-K-M-O-Q-S-U-W-Y-a-c-e-g-i-k-m-o-q-s-u-w-y-0-2-4-6-8-B-D-F-H-J-L-N-P-R-T-V-X-Z":
    logger.critical("Using the default SECRET_KEY from .env file - THIS IS NOT SECURE! Please generate a new one.")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120  # Extended from 30 minutes to 2 hours

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
