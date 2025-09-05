from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional

class CameraBase(BaseModel):
    name: str
    rtsp_url: str
    active: bool = True
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name must be less than 100 characters')
        return v.strip()
    
    @validator('rtsp_url')
    def rtsp_url_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('RTSP URL cannot be empty')
        if not v.startswith('rtsp://'):
            raise ValueError('RTSP URL must start with "rtsp://"')
        if len(v) > 500:
            raise ValueError('RTSP URL must be less than 500 characters')
        return v.strip()

class Camera(CameraBase):
    id: int

    class Config:
        orm_mode = True

class ContactBase(BaseModel):
    name: str
    phone: str
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        if len(v) > 100:
            raise ValueError('Name must be less than 100 characters')
        return v.strip()
    
    @validator('phone')
    def phone_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('Phone cannot be empty')
        if len(v) > 20:
            raise ValueError('Phone number is too long')
        return v.strip()

class Contact(ContactBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class DetectionBase(BaseModel):
    camera_id: int
    timestamp: datetime
    _class: str
    confidence: float
    image_path: Optional[str] = None

class Detection(DetectionBase):
    id: int

    class Config:
        orm_mode = True
        fields = {'_class': 'class'} # Map _class to class for database column

class DetectionOut(Detection):
    camera_name: str

class UserBase(BaseModel):
    username: str
    role: str = "user"
    
    @validator('username')
    def username_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError('Username cannot be empty')
        if len(v) < 3 or len(v) > 30:
            raise ValueError('Username must be between 3 and 30 characters')
        return v.strip()

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def password_must_be_strong(cls, v):
        if not v or len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class User(UserBase):
    id: int

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str