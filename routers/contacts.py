from fastapi import APIRouter, HTTPException
from typing import List

from database import get_db_connection
from models import Contact, ContactBase

router = APIRouter()

@router.post("/api/contacts", response_model=Contact)
async def create_contact(contact: ContactBase):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO contacts (name, phone) VALUES (?, ?)",
              (contact.name, contact.phone))
    conn.commit()
    new_contact_id = c.lastrowid
    c.execute("SELECT id, name, phone, created_at FROM contacts WHERE id = ?", (new_contact_id,))
    new_contact = c.fetchone()
    conn.close()
    return dict(new_contact) # Convert to dict

@router.get("/api/contacts", response_model=List[Contact])
async def get_contacts():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, phone, created_at FROM contacts")
    contacts = c.fetchall()
    conn.close()
    return [dict(contact) for contact in contacts] # Convert to list of dicts

@router.get("/api/contacts/{contact_id}", response_model=Contact)
async def get_contact(contact_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, phone, created_at FROM contacts WHERE id = ?", (contact_id,))
    contact = c.fetchone()
    conn.close()
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return dict(contact) # Convert to dict

@router.put("/api/contacts/{contact_id}", response_model=Contact)
async def update_contact(contact_id: int, contact: ContactBase):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE contacts SET name = ?, phone = ? WHERE id = ?",
              (contact.name, contact.phone, contact_id))
    conn.commit()
    c.execute("SELECT id, name, phone, created_at FROM contacts WHERE id = ?", (contact_id,))
    updated_contact = c.fetchone()
    conn.close()
    return dict(updated_contact) # Convert to dict

@router.delete("/api/contacts/{contact_id}")
async def delete_contact(contact_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
    conn.commit()
    conn.close()
    return {"message": "Contact deleted successfully"}