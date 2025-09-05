### `security.md`

**Role**: Security Engineer

* Run FastAPI with Uvicorn behind Nginx (reverse proxy).
* HTTPS via self-signed cert (LAN use).
* Secrets in `.env`, never hardcoded.
* Input validation everywhere.
* Least-privilege DB access.
* Regular logs rotated.
