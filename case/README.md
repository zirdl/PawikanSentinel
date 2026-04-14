# Pawikan Sentinel Case Daemon

The Raspberry Pi Case Daemon is a standalone, lightweight background service located in `case/daemon` designed to natively manage the hardware associated with the Pawikan Sentinel hardware casing, which typically includes an Ice Tower or customized case with:

1. **A mini I2C OLED Screen (commonly 128x32)**
2. **Addressable RGB LEDs (WS2812B / NeoPixel) integrated into the cooling fan**
3. **PWM-controlled cooling fan speeds**

This daemon ensures your Raspberry Pi looks beautiful and runs quietly while actively processing inference logic.

## Capabilities

*   **OLED Display Manager (`oled_manager.py`)**: 
    *   Dynamically cycles between a real-time **System Metrics** display (IP address, CPU load, RAM usage, CPU temp).
    *   A **Brand Applet** that seamlessly parses and plays any custom image or `.gif` file beside the `PAWIKAN SENTINEL` title label.
*   **RGB Lighting Controller (`rgb_manager.py`)**:
    *   Leverages direct low-level DMA arrays via the `rpi_ws281x` C-library bindings to send precise digital lighting signals to the hardware without taxing the CPU.
    *   Includes dynamic profiles such as `breathing` (slow ocean pulses) and `thermal` (shifts color based on CPU temperature).
*   **Intelligent PWM Fan Curve (`fan_manager.py`)**:
    *   Silences the fan entirely when the system is idling below 45°C.
    *   Linearly increases fan RPM automatically up to 100% as the CPU reaches intensive thermal loads (>70°C).

## Prerequisites (Raspberry Pi Target)

The Python daemon expects several hardware-binding libraries to be installed locally on the deployment Raspberry Pi:

```bash
# Update and install build requirements
sudo apt update
sudo apt install python3-pip python3-pil libjpeg-dev zlib1g-dev libfreetype6-dev liblcms2-dev libopenjp2-7 libtiff5

# Install python libraries
sudo pip3 install luma.oled rpi_ws281x gpiozero psutil 
```
*(Note: If testing on a non-Raspberry Pi Linux machine, the python script uses defensive coding and will safely execute via Mock debugging.)*

## Usage & Configuration

The service is highly modular and allows parameters directly executed during start.
To test manually:

```bash
cd daemon
sudo python3 pawikan_daemon.py --rgb-mode breathing --media /path/to/your/custom_logo.gif
```

### CLI Arguments
*   `--rgb-mode`: Sets the startup state of the fan lighting. Options: `breathing` (default), `thermal`, `inference`.
*   `--media`: An absolute path to any static PNG/JPEG or animated `.gif` file. The file is mapped, resized, and rendered efficiently inside the daemon.

## Systemd Installation

To ensure the hardware initializes on boot seamlessly, a system daemon file is included.
We recommend deploying the app to `/opt/PawikanSentinel/` on the target Pi to maintain standardized paths. 

```bash
# 1. Edit the service file if your absolute paths differ
nano pawikan-case.service

# 2. Copy the service definition to the system directory
sudo cp pawikan-case.service /etc/systemd/system/

# 3. Reload, Enable, and Start
sudo systemctl daemon-reload
sudo systemctl enable pawikan-case
sudo systemctl start pawikan-case

# 4. Check status
sudo systemctl status pawikan-case
```
