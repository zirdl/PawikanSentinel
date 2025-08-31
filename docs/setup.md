## Setup Guide

### Hardware & OS Target
- Raspberry Pi 4B with 8GB RAM
- Raspberry Pi OS 64-bit (Bookworm recommended)
- Stable LAN/Wi-Fi connection for RTSP cameras

### Python Environment with uv
The project leverages **uv** for efficient Python package and environment management.

#### Key uv commands:
- `uv init` → initialize project (`pyproject.toml`)
- `uv add <package>` → add dependencies
- `uv run pawikan-dashboard` → run the web application
- `uv run pawikan-inference` → run the background inference service
- `uv run create-admin` → create the initial admin user
- `uv lock` → write lockfile
- `uv sync` → sync environment from lockfile

#### Core Dependencies:
- `fastapi`
- `uvicorn`
- `passlib[bcrypt]`
- `python-multipart`
- `jinja2`
- `opencv-python-headless`
- `tflite-runtime`
- `python-json-logger`
- `tenacity`
- `twilio` (for SMS via Twilio API)
- `pyserial` (for SMS via GSM modem)

### Data Storage & Paths
- **Config**: `~/.config/pawikan/config.toml` (for static config like model path, RTSP URL).
- **Gallery**: `~/.local/share/pawikan/gallery/`
- **Logs**: `/var/log/pawikan/` (`pawikan.json.log`)
- **SQLite DB**: `~/.local/share/pawikan/pawikan.db` (stores contacts, **users**, **settings**, and **analytics**).
- **Model**: `~/.local/share/pawikan/models/pawikan_int8.tflite`

*The application will create missing directories with safe permissions.*

### Configuration File (TOML)
A sample configuration file. Note that sensitive keys and dynamic settings will be managed via the web UI and stored in the database.

```toml
[model]
path = "/home/pi/.local/share/pawikan/models/pawikan_int8.tflite"
confidence_threshold = 0.5
nms_iou_threshold = 0.45
max_detections = 50

[video]
rtsp_url = "rtsp://user:pass@camera-ip:554/stream"
reconnect_initial_seconds = 2
reconnect_max_seconds = 30

[gallery]
dir = "/home/pi/.local/share/pawikan/gallery"
save_interval_minutes = 10

[logging]
dir = "/var/log/pawikan"
level = "INFO"
```
