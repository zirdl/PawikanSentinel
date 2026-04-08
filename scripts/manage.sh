#!/bin/bash
# Pawikan Sentinel - Management CLI
# Usage: ./manage.sh [command]

set -euo pipefail

# Find project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="pawikan-sentinel.service"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_usage() {
    echo -e "${CYAN}Pawikan Sentinel Management CLI${NC}"
    echo "Usage: ./scripts/manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start       Start the systemd service"
    echo "  stop        Stop the systemd service"
    echo "  restart     Restart the systemd service"
    echo "  status      Check the status of the service"
    echo "  dev         Run the web app locally with auto-reload (no sudo)"
    echo "  logs        View recent service logs via journalctl"
    echo "  follow      Follow service logs live"
    echo "  update      Pull latest code and restart service"
    echo "  backup      Trigger a manual backup of DB and detections"
    echo ""
}

check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        else
            echo -e "${RED}Error: This command Requires root privileges to manage systemd.${NC}"
            exit 1
        fi
    else
        SUDO=""
    fi
}

case "${1:-}" in
    start)
        check_sudo
        echo "Starting $SERVICE_NAME..."
        $SUDO systemctl start "$SERVICE_NAME"
        ;;
    stop)
        check_sudo
        echo "Stopping $SERVICE_NAME..."
        $SUDO systemctl stop "$SERVICE_NAME"
        ;;
    restart)
        check_sudo
        echo "Restarting $SERVICE_NAME..."
        $SUDO systemctl restart "$SERVICE_NAME"
        ;;
    status)
        check_sudo
        $SUDO systemctl status "$SERVICE_NAME" || true
        ;;
    logs)
        check_sudo
        $SUDO journalctl -u "$SERVICE_NAME" -n 50 --no-pager
        ;;
    follow)
        check_sudo
        $SUDO journalctl -u "$SERVICE_NAME" -f
        ;;
    update)
        check_sudo
        echo -e "${YELLOW}Updating Pawikan Sentinel...${NC}"
        cd "$PROJECT_ROOT"
        git pull origin main
        uv sync
        echo "Restarting service..."
        $SUDO systemctl restart "$SERVICE_NAME"
        echo -e "${GREEN}Update complete!${NC}"
        ;;
    backup)
        echo "Triggering manual backup..."
        cd "$PROJECT_ROOT"
        if [ -f "scripts/backup.sh" ]; then
            bash scripts/backup.sh
        else
            echo -e "${RED}Error: scripts/backup.sh not found.${NC}"
            exit 1
        fi
        ;;
    dev)
        echo -e "${CYAN}Starting Pawikan Sentinel in DEV mode (auto-reload)...${NC}"
        cd "$PROJECT_ROOT"
        if [ ! -d ".venv" ]; then
            echo -e "${YELLOW}Warning: .venv not found. Creating it...${NC}"
            uv venv
            uv sync --extra inference
        fi
        
        echo -e "${CYAN}Starting backend FastAPI server (port 8000)...${NC}"
        uv run uvicorn src.core.main:app --host 127.0.0.1 --port 8000 --reload &
        BACKEND_PID=$!
        
        echo -e "${CYAN}Starting frontend Vite server (port 5173)...${NC}"
        cd "$PROJECT_ROOT/frontend"
        if [ ! -d "node_modules" ]; then
            echo -e "${YELLOW}Warning: node_modules not found. Running npm install...${NC}"
            npm install
        fi
        npm run dev &
        FRONTEND_PID=$!
        
        # Cleanup trap to kill background processes on exit
        trap "echo -e '\n${YELLOW}Stopping development servers...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM EXIT
        
        echo -e "${GREEN}Development servers are running!${NC}"
        echo -e "Backend:  http://localhost:8000"
        echo -e "Frontend: http://localhost:5173"
        echo -e "Press Ctrl+C to stop both servers."
        
        # Wait indefinitely for processes to finish
        wait
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
