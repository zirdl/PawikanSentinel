# Pawikan Sentinel

A real-time wildlife monitoring system for detecting and tracking marine turtles (*pawikan*) using **YOLO11** computer vision and **RTSP camera streams**.

## Features

- **Multi-camera RTSP monitoring** — add, edit, and manage unlimited camera feeds via web UI
- **Real-time YOLO11 inference** — local model (no cloud/API needed), auto-downloaded from HuggingFace on first run
- **Live annotated detections** — bounding boxes with labels saved per detection event
- **SMS alerts** — configurable iprog SMS notifications when turtles are detected (with cooldown)
- **System health monitoring** — CPU & thermal throttling with automatic frame skipping
- **SQLite-backed** — all cameras, detections, and users stored locally
- **Web dashboard** — camera management, detection history, user authentication, rate limiting

## Architecture

```
┌──────────────────────────────────┐
│         Web App (FastAPI)        │
│  :8000  Dashboard + Camera CRUD  │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│      Inference App (FastAPI)     │
│  :8001  RTSP worker management   │
│  ┌──────────┐  ┌──────────┐     │
│  │Camera 1  │  │Camera 2  │ …   │
│  │ Worker   │  │ Worker   │     │
│  └────┬─────┘  └────┬─────┘     │
│       ▼              ▼           │
│   cv2.VideoCapture  cv2.VideoCapture │
│       ▼              ▼           │
│   YOLO11 model     YOLO11 model  │
│   (local .pt)      (local .pt)   │
└──────────────────────────────────┘
```

Each camera runs in its own thread (`RTSPInferenceWorker`) inside a `ThreadPoolExecutor`, with independent circuit breakers, reconnect logic, and detection buffers.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) — Python package manager and project runner
- Node.js and npm (for building Tailwind CSS)

## Quick Start

### 1. Install dependencies

```bash
uv sync --extra inference
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings (SMS tokens, thresholds, etc.)
```

Key configuration in `.env`:

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL_DIR` | `models` | Directory for the local `.pt` model file |
| `YOLO_INPUT_SIZE` | `320` | Resize frames before inference |
| `CONFIDENCE_THRESHOLD` | `0.8` | Minimum detection confidence to log |
| `FRAME_SKIP` | `10` | Process every Nth frame |
| `CPU_THRESHOLD` | `80` | Pause inference if CPU % exceeds this |
| `TEMP_THRESHOLD` | `80` | Pause inference if temp (°C) exceeds this |
| `MAX_INFERENCE_WORKERS` | `2` | Max concurrent camera workers |
| `IPROG_API_TOKEN` | — | API token for SMS alerts |
| `SMS_NOTIFICATION_COOLDOWN` | `10` | Minutes between SMS alerts |

### 3. Run the apps

**Web dashboard** (camera management, detections, auth):
```bash
./run-web.sh
# → http://localhost:8000
```

**Inference service** (RTSP workers, YOLO detection):
```bash
./run-inference.sh
# → http://localhost:8001
```

Both apps share the same SQLite database (`pawikan.db`).

## Adding a Camera

1. Open the web dashboard → **Cameras** page
2. Click **Add Camera**, enter a name and RTSP URL (e.g. `rtsp://192.168.1.100/stream`)
3. Set **Active** and save — an inference worker starts automatically

Or via REST API:

```bash
curl -X POST http://localhost:8000/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"name": "Beach Cam 1", "rtsp_url": "rtsp://192.168.1.100/stream", "active": true}'
```

## Local Development (No Real Camera)

Use **VLC** to broadcast a video file as a fake RTSP stream:

1. Open VLC → **Media → Stream** → add a video file
2. Check **Stream**, click **Next**
3. Destination: **RTSP**, port `8554`, path `/test`
4. Start streaming

Then add camera with URL `rtsp://localhost:8554/test` in the dashboard.

You can also run the standalone inference script for quick testing:

```bash
uv run python src/inference/rtsp_infer.py --rtsp-url rtsp://localhost:8554/test
```

## Project Structure

```
├── src/
│   ├── api/            # REST API routers (cameras, detections)
│   ├── core/           # Main app, database, models, auth
│   │   ├── main.py             # Web app entry point + worker management
│   │   ├── database.py         # SQLite schema (cameras, detections, users)
│   │   └── models.py           # Pydantic models
│   ├── inference/      # YOLO11 inference pipeline
│   │   ├── inference.py        # RTSPInferenceWorker (core worker)
│   │   ├── yolo_detector.py    # YOLO model wrapper
│   │   ├── rtsp_infer.py       # Standalone testing script
│   │   ├── inference_service.py# Worker lifecycle manager
│   │   └── sms_sender.py       # iprog SMS notification
│   └── web/            # Templates & static assets (Jinja2 + Tailwind)
├── models/             # Local YOLO11 .pt model files
├── detections/         # Saved annotated detection images
├── backups/            # Database backups
├── docs/
│   ├── DEVELOPMENT.md  # Detailed dev guide
│   ├── scaling.md      # Scaling guide
│   └── contexts/       # Architecture & subsystem docs
├── .env.example        # Environment template
├── pyproject.toml      # Python dependencies
└── tailwind.config.js  # Tailwind CSS config
```

## Model Details

The YOLO11 turtle detection model is **auto-downloaded from HuggingFace** ([`BVRA/TurtleDetector`](https://huggingface.co/BVRA/TurtleDetector)) on the first inference run. No Docker, no external API — it runs as a local `.pt` file via `ultralytics`.

Subsequent runs use the cached model from the `models/` directory.

## SMS Notifications

When a turtle is detected, an SMS alert is sent via the **iprog** API. A configurable cooldown period (`SMS_NOTIFICATION_COOLDOWN`) prevents spam. Set `IPROG_API_TOKEN` and `IPROG_SENDER_NAME` in `.env` to enable.

## Backups

Database backups are stored in `backups/` with configurable retention (`PAWIKAN_BACKUP_RETENTION` in days, default: 7).

## Detailed Documentation

- [DEVELOPMENT.md](docs/DEVELOPMENT.md) — Local development, VLC simulation, testing workflows
- [scaling.md](docs/scaling.md) — Scaling to multiple cameras and edge devices
- [contexts/](docs/contexts/) — Deep dives into each subsystem
