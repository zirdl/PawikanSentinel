# Pawikan Sentinel: Raspberry Pi Setup Guide

This guide provides the step-by-step commands to configure your Raspberry Pi 4B for the Pawikan Sentinel project.

## 1. Update System Packages

First, ensure your system's package list and installed packages are up-to-date.

```bash
sudo apt-get update && sudo apt-get upgrade -y
```

## 2. Install Essential Tools & Libraries

These packages are required for building Python modules and for handling various image and video processing tasks.

```bash
sudo apt-get install -y build-essential git python3-pip python3-venv libatlas-base-dev libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev
```

## 3. Install OpenCV

OpenCV is a critical library for capturing and processing video frames from the RTSP camera.

```bash
sudo apt-get install -y python3-opencv
```

## 4. Create a Python Virtual Environment

Using a virtual environment is highly recommended to isolate the project's dependencies and avoid conflicts with system-wide packages.

```bash
# Create the virtual environment in your home directory
python3 -m venv ~/pawikan-env

# Add a line to your .bashrc to automatically activate the environment in new terminals
echo "source ~/pawikan-env/bin/activate" >> ~/.bashrc

# Activate the environment for the current session
source ~/pawikan-env/bin/activate
```
**Note:** After running these commands, you may need to restart your terminal or run `source ~/.bashrc` for the environment to activate automatically in every new session.

## 5. Install Core Python Dependencies

With the virtual environment active, install the necessary Python packages for running the inference engine.

```bash
pip install numpy
pip install tflite-runtime
```

After completing these steps, your Raspberry Pi will be ready for the Pawikan Sentinel application.
