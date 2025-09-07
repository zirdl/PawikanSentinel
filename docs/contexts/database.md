### database.md

**Your Role**: Database Architect

* **Tables**:
  * `users`: id, username, hashed\_password, role.
  * `contacts`: id, name, phone, created\_at.
  * `detections`: id, camera\_id, timestamp, class, confidence, image\_path.
  * `cameras`: id, name, rtsp\_url, active.
* **Indexes**: timestamp (detections), username (users).
* SQLite, migrate with Alembic (optional).
