#!/bin/bash
# One-time setup for Zaino Mac Mini services
# Installs the unified monitor as a LaunchAgent + cron backup

set -e

REPO_DIR="$HOME/Apps Zaino/telegram_zaino"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

echo "🏠 Zaino Mac Mini - Setup"
echo ""

cd "$REPO_DIR"

# Make scripts executable
chmod +x zaino-monitor.sh update.sh

# === LaunchAgent: zaino-monitor ===
echo "🔧 Instalando LaunchAgent (zaino-monitor)..."

# Unload old agents if present
launchctl unload "$LAUNCH_AGENTS/com.zaino.telegram-bot-updater.plist" 2>/dev/null || true
launchctl unload "$LAUNCH_AGENTS/com.zaino.monitor.plist" 2>/dev/null || true
rm -f "$LAUNCH_AGENTS/com.zaino.telegram-bot-updater.plist"

# Install new agent
cp com.zaino.monitor.plist "$LAUNCH_AGENTS/"
launchctl load "$LAUNCH_AGENTS/com.zaino.monitor.plist"

echo "   ✅ zaino-monitor instalado (arranca al boot, se reinicia si se cae)"

# === Cron: update.sh backup ===
echo "⏰ Configurando cron backup (cada 5 minutos)..."
CRON_JOB="*/5 * * * * $REPO_DIR/update.sh"
(crontab -l 2>/dev/null | grep -v "telegram_zaino/update.sh"; echo "$CRON_JOB") | crontab -

echo "   ✅ Cron backup instalado"

# === Create required directories ===
mkdir -p "$HOME/Apps Zaino/zaino-monitor"
mkdir -p "$REPO_DIR/.trigger"

echo ""
echo "✅ Setup completo!"
echo ""
echo "Servicios activos:"
echo "  1. zaino-monitor.sh (LaunchAgent) → monitorea HAOS, Docker, /update"
echo "  2. update.sh (cron cada 5 min) → backup de auto-update"
echo ""
echo "Logs:"
echo "  Monitor: ~/Apps Zaino/zaino-monitor/monitor.log"
echo "  Update:  $REPO_DIR/update.log"
echo ""
echo "Verificar: ps aux | grep zaino-monitor"
