#!/bin/bash
# Pawikan Sentinel - Interactive Setup Script for Raspberry Pi
# Run this on your Raspberry Pi to set up the entire application.
#
# Prerequisites: uv (Python package manager)
#   Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage: bash setup.sh

set -euo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
BOLD='\033[1m'

# Defaults
PROJECT_DIR="${PWD}"
INSTALL_DIR=""
USE_SYSTEMD=true
START_SERVICE=true

########################################
# Utility Functions
########################################
print_header() {
    echo -e "\n${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_step() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
print_err()  { echo -e "${RED}[✗]${NC} $1"; }

ask_yes_no() {
    local prompt="$1"
    local default="${2:-y}"
    local yn
    while true; do
        if [ "$default" = "y" ]; then
            echo -n "$prompt [Y/n]: "
        else
            echo -n "$prompt [y/N]: "
        fi
        read -r yn
        yn="${yn:-$default}"
        case "$yn" in
            [yY]|[yY][eE][sS]) return 0 ;;
            [nN]|[nN][oO])     return 1 ;;
            *) echo "  Please answer y or n." ;;
        esac
    done
}

read_input() {
    local prompt="$1"
    local default="$2"
    local val
    echo -n "  $prompt"
    if [ -n "$default" ]; then echo -n " [$default]"; fi
    echo -n ": "
    read -r val
    echo "${val:-$default}"
}

check_command() {
    command -v "$1" &>/dev/null
}

check_system() {
    local errors=0

    if [[ $(uname -m) != "aarch64" ]]; then
        print_warn "Not running on ARM64 (aarch64). This is expected on a dev machine."
        print_warn "  Deployment to Pi requires an ARM64 environment."
    fi

    if ! check_command python3; then
        print_err "python3 is required but not installed."
        errors=$((errors + 1))
    fi

    if ! check_command git; then
        print_err "git is required but not installed."
        errors=$((errors + 1))
    fi

    if ! check_command uv; then
        print_err "uv is required but not installed."
        print_err "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        errors=$((errors + 1))
    fi

    return $errors
}

########################################
# Interactive Setup
########################################
main() {
    print_header "Pawikan Sentinel — Setup"
    echo "This script will set up the sea turtle detection system."
    echo ""

    # --- Step 1: System Check ---
    print_header "Step 1: System Check"

    if check_system; then
        print_step "All required dependencies found."
    else
        print_err "Missing dependencies. Install them with:"
        echo "  sudo apt update && sudo apt install -y python3 git curl"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo ""
        if ask_yes_no "  Attempt to install now?" "y"; then
            sudo apt update && sudo apt install -y python3 git curl
            curl -LsSf https://astral.sh/uv/install.sh | sh
        else
            echo "Aborting. Install the missing packages and try again."
            exit 1
        fi
    fi

    # --- Step 2: Installation Directory ---
    print_header "Step 2: Installation Directory"

    echo "Current directory: $PROJECT_DIR"
    INSTALL_DIR="$PROJECT_DIR"

    if ask_yes_no "  Install here?" "y"; then
        : # Use PROJECT_DIR
    else
        INSTALL_DIR=$(read_input "  Full path to install directory" "/home/pi/pawikan-sentinel")
    fi

    mkdir -p "$INSTALL_DIR"/{detections,logs,models,backups}
    print_step "Directories created: detections, logs, models, backups"

    # --- Step 3: Python Virtual Environment ---
    print_header "Step 3: Python Virtual Environment"

    if [ -d "$INSTALL_DIR/.venv" ]; then
        print_warn "Existing venv found at $INSTALL_DIR/.venv"
        if ask_yes_no "  Remove and recreate?" "n"; then
            rm -rf "$INSTALL_DIR/.venv"
        fi
    fi

    if [ ! -d "$INSTALL_DIR/.venv" ]; then
        echo "  Creating venv with uv..."
        cd "$INSTALL_DIR"
        uv venv
    fi

    source "$INSTALL_DIR/.venv/bin/activate"
    print_step "Virtual environment ready: $(python3 --version)"

    # --- Step 4: Python Dependencies ---
    print_header "Step 4: Install Python Dependencies"

    echo "Installing (via uv):"
    echo "  - fastapi, uvicorn (web server)"
    echo "  - ultralytics (YOLO11 inference)"
    echo "  - opencv-python-headless (camera capture)"
    echo "  - huggingface_hub (model download)"
    echo "  - auth, SMS, utilities"
    echo ""

    if ask_yes_no "  Install now?" "y"; then
        cd "$INSTALL_DIR"
        uv sync --extra inference
        print_step "All Python packages installed."
    else
        print_warn "Skipping. Install manually later with:"
        echo "  uv sync --extra inference"
    fi

    # --- Step 5: Environment Configuration ---
    print_header "Step 5: Environment Configuration"

    ENV_FILE="$INSTALL_DIR/.env"

    if [ -f "$ENV_FILE" ]; then
        print_warn "Existing .env found."
        if ask_yes_no "  Overwrite?" "n"; then
            : # Keep existing
        fi
    fi

    if [ ! -f "$ENV_FILE" ]; then
        echo "  Generating configuration..."
        echo ""

        # Secret keys
        SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        SESSION_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        CSRF_KEY=$(python3 -c "import secrets; print(secrets.token_hex(16))")

        # SMS token
        IPROG_TOKEN=$(read_input "  iprog API token (for SMS alerts, or leave blank)" "")

        # Pi-specific settings
        if [[ $(uname -m) == "aarch64" ]]; then
            MAX_WORKERS=$(read_input "  Max inference workers (Pi 4B: 1-2)" "2")
            INPUT_SIZE=$(read_input "  Model input resolution (320 for Pi, 640 for desktop)" "320")
        else
            MAX_WORKERS=$(read_input "  Max inference workers" "10")
            INPUT_SIZE=$(read_input "  Model input resolution" "640")
        fi

        FRAME_SKIP=$(read_input "  Frame skip (process every Nth frame)" "10")
        CONF_THRESHOLD=$(read_input "  Confidence threshold for alerts (0.0-1.0)" "0.8")
        CPU_LIMIT=$(read_input "  CPU usage threshold (%) for throttling" "80")
        TEMP_LIMIT=$(read_input "  Temperature threshold (C) for throttling" "80")

        # Backup settings
        echo ""
        echo "  --- Backup Configuration ---"
        BACKUP_BASE=$(read_input "  Backup directory (absolute path recommended)" "$INSTALL_DIR/backups")
        BACKUP_DAYS=$(read_input "  Retention period (days)" "7")
        mkdir -p "$BACKUP_BASE"

        cat > "$ENV_FILE" << EOF
# Pawikan Sentinel — Auto-generated $(date '+%Y-%m-%d %H:%M')
SECRET_KEY=${SECRET_KEY}
SESSION_SECRET_KEY=${SESSION_KEY}
CSRF_SECRET_KEY=${CSRF_KEY}

# SMS (iprog)
IPROG_API_TOKEN=${IPROG_TOKEN:-}
IPROG_SENDER_NAME=PawikanSentinel

# YOLO11 Model — auto-downloaded from HuggingFace (BVRA/TurtleDetector)
YOLO_MODEL_DIR=models
YOLO_INPUT_SIZE=${INPUT_SIZE}
YOLO_CONF_THRESHOLD=0.25

# Detection
CONFIDENCE_THRESHOLD=${CONF_THRESHOLD}
FRAME_SKIP=${FRAME_SKIP}
RESIZE_WIDTH=${INPUT_SIZE}
RESIZE_HEIGHT=${INPUT_SIZE}
CPU_THRESHOLD=${CPU_LIMIT}
TEMP_THRESHOLD=${TEMP_LIMIT}
MAX_INFERENCE_WORKERS=${MAX_WORKERS}

# Backups
PAWIKAN_BACKUP_DIR=${BACKUP_BASE}
PAWIKAN_BACKUP_RETENTION=${BACKUP_DAYS}

# Directories
DETECTIONS_DIR=detections
DATABASE_PATH=pawikan.db
EOF
        print_step "Configuration saved to .env"
    fi

    # --- Step 6: systemd Service ---
    print_header "Step 6: systemd Service (Linux)"

    if [ "$(id -u)" -eq 0 ]; then
        HAS_SUDO=true
    elif check_command sudo; then
        HAS_SUDO=true
    else
        HAS_SUDO=false
    fi

    if [ "$HAS_SUDO" = true ]; then
        SERVICE_FILE="/etc/systemd/system/pawikan-sentinel.service"

        if [ -f "$SERVICE_FILE" ]; then
            print_warn "Existing service found at $SERVICE_FILE"
            if ask_yes_no "  Replace?" "y"; then
                : # Will overwrite below
            else
                USE_SYSTEMD=false
            fi
        fi

        if [ "$USE_SYSTEMD" = true ]; then
            echo "  Installing systemd service..."
            cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Pawikan Sentinel - Sea Turtle Detection System
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${INSTALL_DIR}/.venv/bin"
ExecStart=${INSTALL_DIR}/.venv/bin/uvicorn src.core.main:app --host 0.0.0.0 --port 8000 --no-reload
Restart=always
RestartSec=10
MemoryMax=1G
CPUQuota=150%

[Install]
WantedBy=multi-user.target
EOF
            systemctl daemon-reload
            systemctl enable pawikan-sentinel.service
            print_step "systemd service installed and enabled."
        fi
    else
        print_warn "No sudo access — cannot install systemd service."
        print_warn "You can still run the app manually with: bash run-web.sh"
        USE_SYSTEMD=false
    fi

    # --- Step 7: Start ---
    print_header "Step 7: Start the Application"

    echo "Summary:"
    echo "  Install dir:  $INSTALL_DIR"
    echo "  Web port:     8000"
    echo "  Model:        YOLO11 (BVRA/TurtleDetector, ~20MB)"
    echo "  Model source: HuggingFace (auto-download on first use)"
    echo "  Login:        admin / admin"
    echo ""

    if [ "$USE_SYSTEMD" = true ]; then
        if ask_yes_no "  Start the service now?" "y"; then
            systemctl start pawikan-sentinel.service
            sleep 2
            if systemctl is-active pawikan-sentinel.service &>/dev/null; then
                print_step "Service running!"
            else
                print_err "Service failed to start. Check logs:"
                echo "  journalctl -u pawikan-sentinel.service -n 50 --no-pager"
            fi
        fi
    else
        if ask_yes_no "  Start manually now?" "y"; then
            echo ""
            echo "  Running: uvicorn src.core.main:app --host 0.0.0.0 --port 8000"
            echo "  Press Ctrl+C to stop."
            echo ""
            cd "$INSTALL_DIR"
            source .venv/bin/activate
            uvicorn src.core.main:app --host 0.0.0.0 --port 8000
        fi
    fi

    fi

    # --- Step 8: Automation (Backups & Logs) ---
    print_header "Step 8: Automation (Backups & Logs)"

    if ask_yes_no "  Setup automated daily backups at 2 AM?" "y"; then
        (crontab -l 2>/dev/null | grep -v "scripts/backup.sh"; echo "0 2 * * * /bin/bash $INSTALL_DIR/scripts/backup.sh >> $INSTALL_DIR/logs/backup.log 2>&1") | crontab -
        print_step "Cronjob added for daily backups."
    fi

    if [ "$HAS_SUDO" = true ]; then
        if ask_yes_no "  Setup logrotate for Sentinel logs?" "y"; then
            $SUDO cp "$INSTALL_DIR/deployments/logrotate.conf" /etc/logrotate.d/pawikan-sentinel
            $SUDO chown root:root /etc/logrotate.d/pawikan-sentinel
            $SUDO chmod 644 /etc/logrotate.d/pawikan-sentinel
            print_step "logrotate configured for Sentinel."
        fi
    fi

    # --- Done ---
    print_header "Setup Complete"

    if [ "$USE_SYSTEMD" = true ]; then
        echo "Service management:"
        echo "  status:   sudo systemctl status pawikan-sentinel.service"
        echo "  stop:     sudo systemctl stop pawikan-sentinel.service"
        echo "  restart:  sudo systemctl restart pawikan-sentinel.service"
        echo "  logs:     journalctl -u pawikan-sentinel.service -f"
    else
        echo "Run the app:"
        echo "  cd $INSTALL_DIR && bash run-web.sh"
    fi

    echo ""
    echo "Open in browser: http://<your-ip>:8000"
    echo ""
}

main "$@"
