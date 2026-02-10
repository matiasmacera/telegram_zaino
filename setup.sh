#!/bin/bash
# One-time setup for Zaino Telegram Bot auto-updates
# Run this once after creating the GitHub repo

set -e

REPO_DIR="$HOME/zaino-telegram-bot"

echo "🏠 Zaino Telegram Bot - Setup"
echo ""

# Check if repo URL was provided
if [ -z "$1" ]; then
    echo "Uso: ./setup.sh git@github.com:TU_USUARIO/zaino-telegram-bot.git"
    echo ""
    echo "Pasos previos:"
    echo "1. Creá un repo privado en GitHub llamado 'zaino-telegram-bot'"
    echo "2. Ejecutá este script con la URL SSH del repo"
    exit 1
fi

REPO_URL="$1"

# Initialize git repo if needed
if [ ! -d "$REPO_DIR/.git" ]; then
    echo "📦 Inicializando repo..."
    cd "$REPO_DIR"
    git init
    git remote add origin "$REPO_URL"
    git branch -M main
else
    echo "📦 Repo ya inicializado"
    cd "$REPO_DIR"
fi

# Create .gitignore
cat > .gitignore << 'EOF'
.env
update.log
__pycache__/
*.pyc
EOF

# First commit and push
echo "📤 Pusheando código..."
git add -A
git commit -m "Initial commit: Zaino Telegram Bot" 2>/dev/null || echo "  (nada nuevo para commitear)"
git push -u origin main

# Make update script executable
chmod +x update.sh

# Setup cron job
echo "⏰ Configurando cron (cada 5 minutos)..."
CRON_JOB="*/5 * * * * $REPO_DIR/update.sh"

# Add to crontab if not already there
(crontab -l 2>/dev/null | grep -v "zaino-telegram-bot/update.sh"; echo "$CRON_JOB") | crontab -

echo ""
echo "✅ Setup completo!"
echo ""
echo "El bot se actualiza automáticamente cada 5 minutos."
echo "Para actualizar: pusheá cambios a 'main' y esperá."
echo "Logs de updates: $REPO_DIR/update.log"
echo ""
echo "Próximo paso: verificá que el .env existe y levantá el bot:"
echo "  cd $REPO_DIR"
echo "  docker compose up -d --build"
