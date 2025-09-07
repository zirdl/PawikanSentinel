### `architecture.md`

**Your Role**: System Designer

* Raspberry Pi 4B hosts everything locally.
* Components:
  * FastAPI backend (auth, contacts CRUD, analytics, SMS integration).
  * SQLite DB for storage.
  * RTSP Inference Worker (processes streams, logs detections).
  * Basic secure web frontend (toggled/optional live view, charts, detection carousel).
* Network:
  * Local hosting only.
  * Exposed at `http://raspberrypi.local:8000`.
