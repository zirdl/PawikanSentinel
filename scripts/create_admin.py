import argparse
import sys
import os

# This is a bit of a hack to make sure we can import the core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pawikan.core.database import SessionLocal, init_db
from pawikan.core import models
from pawikan.core.storage import get_user_by_username
from pawikan.dashboard.auth import get_password_hash

def main():
    """Creates the initial admin user."""
    parser = argparse.ArgumentParser(description="Create Pawikan Sentinel Admin User.")
    parser.add_argument("--password", required=True, help="Password for the admin user. Warning: This may be stored in your shell history.")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    print("--- Create Pawikan Sentinel Admin User ---")

    # Check if an admin user already exists
    if get_user_by_username(db, "admin"):
        print("Admin user already exists. Exiting.")
        db.close()
        return

    try:
        username = "admin"
        password = args.password

        if not password:
            print("Password cannot be empty. Exiting.")
            return
        
        hashed_password = get_password_hash(password)
        db_user = models.User(username=username, hashed_password=hashed_password)
        
        db.add(db_user)
        db.commit()
        
        print(f"Admin user '{username}' created successfully.")

    finally:
        db.close()

if __name__ == "__main__":
    main()
