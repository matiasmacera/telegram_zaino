#!/bin/bash
# Auto-update script for Zaino Telegram Bot
# Runs via cron every 5 minutes
# Only rebuilds if there are actual changes from git

REPO_DIR="$HOME/Apps Zaino/telegram_zaino"
LOG_FILE="$REPO_DIR/update.log"

cd "$REPO_DIR" || exit 1

# Fetch latest changes
git fetch origin main --quiet 2>/dev/null

# Check if there are new commits
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    # No changes, exit silently
    exit 0
fi

# There are changes — pull and rebuild
echo "$(date): Update detected ($LOCAL -> $REMOTE)" >> "$LOG_FILE"

git pull origin main --quiet >> "$LOG_FILE" 2>&1

echo "$(date): Rebuilding container..." >> "$LOG_FILE"
docker compose down >> "$LOG_FILE" 2>&1
docker compose build --no-cache >> "$LOG_FILE" 2>&1
docker compose up -d >> "$LOG_FILE" 2>&1

echo "$(date): Update complete" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
