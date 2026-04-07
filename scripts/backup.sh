#!/bin/bash
# Pawikan Sentinel - Backup Script
# Usage: ./scripts/backup.sh

set -euo pipefail

# Configuration
# Default backup folder relative to project root, or custom from env
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${PAWIKAN_BACKUP_DIR:-$PROJECT_ROOT/backups}"
DB_PATH="$PROJECT_ROOT/pawikan.db"
DETECTIONS_DIR="$PROJECT_ROOT/detections"

# Number of backups to retain
RETENTION_DAYS="${PAWIKAN_BACKUP_RETENTION:-7}"

DATE_STR=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR="$BACKUP_DIR/snapshot_$DATE_STR"

echo "Starting backup process..."
mkdir -p "$SNAPSHOT_DIR"

# 1. Database Backup
if [ -f "$DB_PATH" ]; then
    echo "Backing up database..."
    sqlite3 "$DB_PATH" ".backup '$SNAPSHOT_DIR/pawikan.db'"
else
    echo "No database found at $DB_PATH to backup."
fi

# 2. Detections Backup (Incremental or Full compressed)
if [ -d "$DETECTIONS_DIR" ]; then
    echo "Backing up detections..."
    tar -czf "$SNAPSHOT_DIR/detections.tar.gz" -C "$PROJECT_ROOT" detections/
fi

# 3. Compress snapshot directory
cd "$BACKUP_DIR"
tar -czf "snapshot_$DATE_STR.tar.gz" "snapshot_$DATE_STR"/
rm -rf "snapshot_$DATE_STR"

echo "Backup created at $BACKUP_DIR/snapshot_$DATE_STR.tar.gz"

# 4. Clean up old backups
echo "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "snapshot_*.tar.gz" -mtime +"$RETENTION_DAYS" -exec rm {} \;

echo "Backup complete!"
