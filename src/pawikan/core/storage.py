"""
Provides high-level functions to interact with the database, using the SQLAlchemy setup.
This acts as a data access layer.
"""
from sqlalchemy.orm import Session
from . import models, schemas
from ..dashboard import auth

# --- User Functions ---

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# --- Contact Functions ---

def get_all_contacts(db: Session):
    return db.query(models.Contact).order_by(models.Contact.name).all()

def create_contact(db: Session, contact: schemas.ContactCreate):
    db_contact = models.Contact(**contact.dict())
    db.add(db_contact)
    db.commit()
    db.refresh(db_contact)
    return db_contact

# ... other CRUD functions for contacts ...

# --- Settings Functions ---

def get_setting(db: Session, key: str):
    return db.query(models.Setting).filter(models.Setting.key == key).first()

def update_setting(db: Session, key: str, value: str):
    db_setting = db.query(models.Setting).filter(models.Setting.key == key).first()
    if db_setting:
        db_setting.value = value
    else:
        db_setting = models.Setting(key=key, value=value)
        db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting

# --- Analytics Functions ---

def create_detection_event(db: Session, event: schemas.DetectionEventCreate):
    db_event = models.DetectionEvent(**event.dict())
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event

def get_detection_events(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.DetectionEvent).offset(skip).limit(limit).all()
