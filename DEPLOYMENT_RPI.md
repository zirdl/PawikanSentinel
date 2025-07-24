# Raspberry Pi Deployment Guide

This guide outlines the steps to deploy the Pawikan Sentinel application onto a Raspberry Pi 4B.

## 1. Clone the Repository

First, clone the project repository from your version control system (e.g., GitHub) onto your Raspberry Pi.

```bash
# SSH into your Raspberry Pi first
ssh pi@<your-rpi-ip-address>

# Clone the repository
git clone <your-repository-url>
cd PawikanSentinel

# Switch to the development branch (or the desired deployment branch)
git checkout development
```

## 2. Set Up the Environment

The application requires a specific set of Python packages. It is highly recommended to use a virtual environment to manage these dependencies.

```bash
# Create a Python virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```
**Note:** You may also need to install the TensorFlow Lite runtime separately if it's not included in the `requirements.txt` file. Refer to the official TensorFlow documentation for instructions specific to your Raspberry Pi's architecture.

## 3. Application Configuration

The application requires sensitive information, such as API keys and camera URLs, which should not be stored directly in the repository. Create a configuration file to hold these values.

1.  Create a configuration file (e.g., `config.ini` or `.env`).
2.  Add the necessary configuration variables (e.g., Twilio SID, Auth Token, RTSP URL).
3.  **Crucially, ensure this file is listed in your `.gitignore` file to prevent it from being committed to version control.**

Example `.env` file:
```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+15017122661
RECIPIENT_PHONE_NUMBER=+15558675309
RTSP_URL=rtsp://user:pass@your_camera_ip:554/stream1
```

## 4. Running the Application

Once the environment is set up and configured, you can run the main application script.

```bash
# Ensure the virtual environment is active
source .venv/bin/activate

# Run the main script
python src/main.py
```

## 5. Set Up as a Systemd Service (Recommended)

For a robust deployment, you should run the application as a `systemd` service. This will ensure it starts automatically on boot and can be managed easily.

1.  Create a service file:
    ```bash
    sudo nano /etc/systemd/system/pawikan-sentinel.service
    ```

2.  Add the following content to the file, making sure to replace `<your-user>` and `<your-project-path>` with your actual user and project path:

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
    # Reload the systemd daemon to recognize the new service
    sudo systemctl daemon-reload

    # Enable the service to start on boot
    sudo systemctl enable pawikan-sentinel.service

    # Start the service immediately
    sudo systemctl start pawikan-sentinel.service

    # Check the status of the service
    sudo systemctl status pawikan-sentinel.service
    ```
