#!/bin/bash
# auto-update.sh - Runs on the host, watches for update trigger from the bot
# The bot creates /trigger/update file when /update command is received
# This script detects it, does git pull + rebuild

REPO_DIR="$HOME/Apps Zaino/telegram_zaino"
TRIGGER_DIR="$REPO_DIR/.trigger"
TRIGGER_FILE="$TRIGGER_DIR/update"
LOG_FILE="$REPO_DIR/update.log"

mkdir -p "$TRIGGER_DIR"

echo "👀 Watching for update triggers in $TRIGGER_FILE..."

while true; do
    if [ -f "$TRIGGER_FILE" ]; then
        echo "$(date): Update triggered!" >> "$LOG_FILE"
        rm -f "$TRIGGER_FILE"

        cd "$REPO_DIR" || continue

        # Reset to latest origin/main (avoids divergent branch issues)
        git fetch origin main >> "$LOG_FILE" 2>&1
        git reset --hard origin/main >> "$LOG_FILE" 2>&1

        # Rebuild and restart
        GIT_INFO=$(git log --oneline -5)
        echo "$(date): Rebuilding..." >> "$LOG_FILE"
        docker compose down >> "$LOG_FILE" 2>&1
        docker compose build --no-cache --build-arg "GIT_INFO=$GIT_INFO" >> "$LOG_FILE" 2>&1
        docker compose up -d >> "$LOG_FILE" 2>&1
        echo "$(date): Update complete" >> "$LOG_FILE"
        echo "---" >> "$LOG_FILE"
    fi
    sleep 2
done
