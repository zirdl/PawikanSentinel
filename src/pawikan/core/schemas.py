from pydantic import BaseModel, ConfigDict
import datetime

class ContactBase(BaseModel):
    name: str
    phone_number: str

class ContactCreate(ContactBase):
    pass

class ContactUpdate(ContactBase):
    pass

class Contact(ContactBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class SettingBase(BaseModel):
    key: str
    value: str

class SettingCreate(SettingBase):
    pass

class Setting(SettingBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class DetectionEventBase(BaseModel):
    detection_count: int
    average_confidence: float

class DetectionEventCreate(DetectionEventBase):
    pass

class DetectionEvent(DetectionEventBase):
    id: int
    timestamp: datetime.datetime
    model_config = ConfigDict(from_attributes=True)
