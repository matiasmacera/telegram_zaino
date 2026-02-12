#!/bin/bash
# ============================================================
# Zaino Unified Monitor
# Single service that manages everything on the Mac Mini:
# - Watches for /update trigger from Telegram bot
# - Checks HAOS VM is running and responsive (every 3 min)
# - Checks Telegram bot Docker container is healthy (every 3 min)
# - Sends notifications via Telegram API
# ============================================================

# === SINGLE INSTANCE LOCK ===
LOCK_FILE="$HOME/Apps Zaino/zaino-monitor/.lock"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another instance is already running. Exiting."
    exit 0
fi

# === CONFIG ===
VM_NAME="Home Assistant"
MAX_RETRIES=3
RETRY_DELAY=30

REPO_DIR="$HOME/Apps Zaino/telegram_zaino"
CONTAINER_NAME="zaino-telegram-bot"
TRIGGER_FILE="$REPO_DIR/.trigger/update"

LOG_FILE="$HOME/Apps Zaino/zaino-monitor/monitor.log"
LAST_CHECK_FILE="$HOME/Apps Zaino/zaino-monitor/.last_check"
UTMCTL="/Applications/UTM.app/Contents/MacOS/utmctl"

# Load config from .env.config (non-secret) and .env (secrets)
if [[ -f "$REPO_DIR/.env.config" ]]; then
    export $(grep -E '^(TELEGRAM_USER_ID|HA_URL)=' "$REPO_DIR/.env.config" | xargs)
fi
if [[ -f "$REPO_DIR/.env" ]]; then
    export $(grep -E '^(TELEGRAM_BOT_TOKEN|HA_TOKEN)=' "$REPO_DIR/.env" | xargs)
else
    echo "ERROR: .env not found at $REPO_DIR/.env" >&2
    exit 1
fi
# Admin = first ID (before comma)
TELEGRAM_ADMIN_ID="${TELEGRAM_USER_ID%%,*}"
HA_URL="${HA_URL:-http://192.168.99.232:8123}"

# === SETUP ===
mkdir -p "$HOME/Apps Zaino/zaino-monitor"
mkdir -p "$REPO_DIR/.trigger"

# === FUNCTIONS ===
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

send_telegram() {
    local message="$1"
    curl -s -X POST \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_ADMIN_ID}" \
        -d text="$message" \
        -d parse_mode="HTML" > /dev/null 2>&1
}

check_ha() {
    curl -s -o /dev/null -w "%{http_code}" \
        --connect-timeout 10 --max-time 15 "$HA_URL" 2>/dev/null
}

restart_vm() {
    log "HAOS: Reiniciando VM '$VM_NAME'..."
    send_telegram "⚠️ <b>Home Assistant no responde</b> - Reiniciando VM..."

    $UTMCTL stop "$VM_NAME" 2>/dev/null
    sleep 10
    $UTMCTL start "$VM_NAME"

    log "HAOS: VM reiniciada. Esperando 120s..."
    sleep 120

    response=$(check_ha)
    if [[ "$response" =~ ^(200|301|302|401|403)$ ]]; then
        log "HAOS: Recuperado después de reinicio"
        send_telegram "✅ <b>Home Assistant recuperado</b> - VM reiniciada exitosamente."
    else
        log "HAOS: Sigue sin responder después de reinicio"
        send_telegram "🔴 <b>ALERTA:</b> HA sigue sin responder después de reiniciar la VM."
    fi
}

check_docker_container() {
    local status
    status=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)

    if [[ "$status" != "running" ]]; then
        log "DOCKER: Container '$CONTAINER_NAME' not running (status: $status). Starting..."
        send_telegram "🐳 <b>Bot caído</b> - Reiniciando container..."
        cd "$REPO_DIR" || return
        docker compose up -d >> "$LOG_FILE" 2>&1
        sleep 10

        status=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null)
        if [[ "$status" == "running" ]]; then
            log "DOCKER: Container recovered"
            send_telegram "✅ <b>Bot recuperado</b> - Container reiniciado."
        else
            log "DOCKER: Container still not running after restart"
            send_telegram "🔴 <b>ALERTA:</b> Bot no pudo reiniciar. Revisar manualmente."
        fi
    fi
}

check_update_trigger() {
    if [[ -f "$TRIGGER_FILE" ]]; then
        log "UPDATE: Trigger detected, fetching and rebuilding..."
        rm -f "$TRIGGER_FILE"

        cd "$REPO_DIR" || return

        # Fetch and reset to origin/main (avoids divergent branch issues)
        git fetch origin main >> "$LOG_FILE" 2>&1
        git reset --hard origin/main >> "$LOG_FILE" 2>&1

        GIT_INFO=$(git log --oneline -5)
        log "UPDATE: Rebuilding container..."
        docker compose down >> "$LOG_FILE" 2>&1
        docker compose build --no-cache --build-arg "GIT_INFO=$GIT_INFO" >> "$LOG_FILE" 2>&1
        docker compose up -d >> "$LOG_FILE" 2>&1

        log "UPDATE: Complete"
    fi
}

# === MAIN LOOP ===
log "Monitor started"

while true; do
    # 1. Check for /update trigger (every 2s via the loop)
    check_update_trigger

    # 2. Every 3 minutes: check HAOS + Docker
    CURRENT_TIME=$(date +%s)

    if [[ -f "$LAST_CHECK_FILE" ]]; then
        LAST_CHECK=$(cat "$LAST_CHECK_FILE")
    else
        LAST_CHECK=0
    fi

    ELAPSED=$((CURRENT_TIME - LAST_CHECK))

    if [[ $ELAPSED -ge 180 ]]; then
        # Check HAOS
        failures=0
        for ((i=1; i<=MAX_RETRIES; i++)); do
            response=$(check_ha)
            if [[ "$response" =~ ^(200|301|302|401|403)$ ]]; then
                log "HAOS: OK (response: $response)"
                failures=0
                break
            else
                failures=$((failures + 1))
                log "HAOS: Intento $i/$MAX_RETRIES fallido (response: $response)"
                if [[ $i -lt $MAX_RETRIES ]]; then
                    sleep $RETRY_DELAY
                fi
            fi
        done

        if [[ $failures -ge $MAX_RETRIES ]]; then
            restart_vm
        fi

        # Check Docker container
        check_docker_container

        # Update last check time
        echo "$CURRENT_TIME" > "$LAST_CHECK_FILE"
    fi

    sleep 2
done
