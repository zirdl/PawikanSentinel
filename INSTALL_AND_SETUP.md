# Pawikan Sentinel: Complete Setup, Deployment, and Testing Guide

This guide provides a comprehensive walkthrough for setting up the hardware and software, deploying the application, and testing the Pawikan Sentinel system.

---

## 1. Raspberry Pi: Initial Hardware Setup

This section covers the initial configuration of your Raspberry Pi 4B.

### 1.1. Update System Packages

Ensure your system's package list and installed packages are up-to-date.

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

### 1.2. Install Essential Tools & Libraries

These packages are required for building Python modules and for handling various image and video processing tasks.

```bash
sudo apt-get install -y build-essential git python3-pip python3-venv libatlas-base-dev libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev
```

### 1.3. Install OpenCV

OpenCV is a critical library for capturing and processing video frames from the RTSP camera.

```bash
sudo apt-get install -y python3-opencv
```

---

## 2. Application Deployment

This section outlines the steps to deploy the Pawikan Sentinel application onto your configured Raspberry Pi.

### 2.1. Clone the Repository

First, clone the project repository from your version control system (e.g., GitHub).

```bash
# SSH into your Raspberry Pi first
ssh pi@<your-rpi-ip-address>

# Clone the repository
git clone <your-repository-url>
cd PawikanSentinel
```

### 2.2. Set Up the Environment with `uv`

This project uses `uv` for fast and modern Python package management.

```bash
# Create a Python virtual environment using uv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install the project and its dependencies from pyproject.toml
uv pip install -e .
```

### 2.3. Application Configuration

Create a `config.ini` file to store sensitive information, such as API keys and camera URLs.

**Important**: Ensure `config.ini` is listed in your `.gitignore` file to prevent it from being committed to version control.

Example `config.ini`:
```ini
[APP]
RTSP_URL = rtsp://user:pass@your_camera_ip:554/stream1
MODEL_PATH = /path/to/your/model.tflite
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEDUPLICATION_WINDOW_MINUTES = 10
LOG_FILE = pawikan_sentinel.log

[TWILIO]
ACCOUNT_SID = ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AUTH_TOKEN = your_auth_token
PHONE_NUMBER = +15017122661
RECIPIENT_PHONE_NUMBER = +15558675309
```

### 2.4. Running the Application

Once the environment is set up and configured, you can run the main application script.

```bash
# Ensure the virtual environment is active
source .venv/bin/activate

# Run the main script
python src/main.py
```

### 2.5. Set Up as a Systemd Service (Recommended)

For a robust deployment, run the application as a `systemd` service. This ensures it starts automatically on boot.

1.  Create a service file:
    ```bash
    sudo nano /etc/systemd/system/pawikan-sentinel.service
    ```

2.  Add the following content, replacing `<your-user>` and `<your-project-path>`:

    ```ini
    [Unit]
    Description=Pawikan Sentinel Application
    After=network.target

    [Service]
    User=<your-user>
    WorkingDirectory=<your-project-path>/PawikanSentinel
    ExecStart=<your-project-path>/PawikanSentinel/.venv/bin/python src/main.py
    Restart=always
    RestartSec=10

    [Install]
    WantedBy=multi-user.target
    ```

3.  Enable and start the service:
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable pawikan-sentinel.service
    sudo systemctl start pawikan-sentinel.service
    sudo systemctl status pawikan-sentinel.service
    ```

---

## 3. Testing the Application

Here are two methods for testing the turtle detection application.

### 3.1. Method 1: Simulated Live Test (End-to-End)

This method uses VLC to stream a video file over RTSP, simulating a live camera feed to test the entire pipeline.

**What You'll Need:**
1.  A video file of a sea turtle (e.g., `turtle_test.mp4`).
2.  **VLC Media Player** installed on your computer.

**Steps:**

1.  **Set up the RTSP Stream with VLC:**
    a. Open VLC and go to `Media` -> `Stream`.
    b. Add your `turtle_test.mp4` file.
    c. Click `Stream`, select `RTSP` as the destination, and click `Add`.
    d. Set the "Path" to `/test` (the default port is `8554`).
    e. Click `Next`, then `Stream`. VLC is now streaming at `rtsp://127.0.0.1:8554/test`.

2.  **Configure Pawikan Sentinel:**
    Update your `config.ini` to point to the local VLC stream:
    ```ini
    [APP]
    RTSP_URL = rtsp://127.0.0.1:8554/test
    ...
    ```

3.  **Run the Application:**
    With the VLC stream running, start the Pawikan Sentinel application:
    ```bash
    source .venv/bin/activate
    python3 src/main.py
    ```

### 3.2. Method 2: Direct File Processing

This method modifies the `main.py` script to read directly from a local video file, which is useful for quickly testing the ML model.

**Steps:**

1.  **Modify `src/main.py`:**
    Comment out the RTSP client initialization and replace it with a `cv2.VideoCapture` object that reads from your local file.

    ```python
    # In main():
    # Comment out the RTSP client initialization
    # rtsp_client = RTSPClient(RTSP_URL)
    
    # Add this line to read from a local video file
    video_path = 'path/to/your/turtle_test.mp4' # <--- CHANGE THIS
    video_capture = cv2.VideoCapture(video_path)
    
    # In the main loop:
    # Replace the frame reading logic with:
    success, frame = video_capture.read()
    if not success:
        print("End of video file.")
        break
    
    # In the finally block:
    # release the video capture object
    video_capture.release()
    ```

2.  **Run the Modified Script:**
    ```bash
    source .venv/bin/activate
    python3 src/main.py
    ```
**Important:** Remember to revert the changes to `src/main.py` before final deployment.

---

## 4. Future Implementation: Web Dashboard

This section outlines the plan for a future web dashboard to manage and monitor the system.

### 4.1. Goal

Create a web interface for viewing the live camera feed, detection history, and managing team contacts.

### 4.2. Proposed Architecture

A separate web application (backend, database, frontend) that communicates with the Raspberry Pi detection script.

```mermaid
graph TD
    subgraph Raspberry Pi
        A[RTSP Camera] --> B(Detection Script);
    end

    subgraph Web Server (Cloud or On-Premise)
        D[Web Backend API];
        E[Database];
        F[Web Frontend];
    end

    subgraph End User
        G[Web Browser];
        H[Conservation Team Phone];
    end

    B -->|1. Send Detection Data (HTTP POST)| D;
    B -->|2. Send SMS (via SIM)| H;
    D <-->|3. Read/Write Data| E;
    F <-->|4. Fetch Data via API| D;
    G -->|5. Views Dashboard| F;
    F -->|6. Displays Live Feed| A;
```

### 4.3. Key Features
-   User Authentication
-   Team Contact Management
-   Detection Event Dashboard
-   Live Video Feed

### 4.4. Implementation Roadmap
1.  **Phase 1:** Backend (FastAPI/Flask) & Database (PostgreSQL/SQLite) Setup.
2.  **Phase 2:** Create an API endpoint for the Raspberry Pi to log detection events.
3.  **Phase 3:** Frontend (React/Vue.js) development for the UI.
4.  **Phase 4:** Integrate a live RTSP stream into the web interface.
