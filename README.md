# Zaino Home Assistant Bot

Bot de Telegram que controla Home Assistant usando Claude AI como motor de inteligencia. Soporta texto y voz.

## Funcionalidades

- **Control por lenguaje natural** — Escribí o mandá un audio y el bot interpreta lo que querés hacer
- **Home Assistant completo** — Luces, aires, cortinas, cerraduras, alarma, media players, aspiradoras
- **Voz** — Transcripción con Google Gemini, procesamiento con Claude
- **Pileta** — Monitoreo de química del agua (WaterGuru), filtrado, llenado, temperatura
- **Música** — Control de 18+ parlantes Sonos, multiroom, agrupación por zonas
- **WaterGuru** — Notificaciones automáticas cuando hay nueva medición
- **Administración remota** — Logs, versión, update y cambio de token desde Telegram

## Comandos de Telegram

| Comando | Descripción |
|---------|-------------|
| `/start` | Muestra ayuda y comandos disponibles |
| `/status` | Resumen rápido del estado de la casa |
| `/pileta` | Estado completo de la pileta (química, temperatura, filtrado) |
| `/musica` | Qué está sonando en cada parlante |
| `/reset` | Limpia el historial de conversación |
| `/update` | Actualiza el bot desde GitHub (git pull + rebuild) |
| `/settoken <token>` | Actualiza el token de Home Assistant remotamente |
| `/logs [N]` | Muestra las últimas N líneas de logs (default: 30) |
| `/version` | Versión actual y uptime del bot |

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                   Telegram                          │
│              (texto / voz / comandos)               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              Docker Container                       │
│           (zaino-telegram-bot)                      │
│                                                     │
│  bot.py                                             │
│  ├─ Telegram handlers (texto, voz, comandos)        │
│  ├─ Claude AI (tools para Home Assistant)           │
│  ├─ Gemini (transcripción de audio)                 │
│  ├─ WaterGuru polling (cada 30 min)                 │
│  └─ Heartbeat (cada 30s → /tmp/bot_healthy)         │
│                                                     │
│  Health check: Docker verifica cada 60s             │
│  Restart policy: always                             │
│  Logs: JSON, 10MB x 3 archivos (30MB max)          │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            Home Assistant API                       │
│         (http://homeassistant.local:8123)           │
└─────────────────────────────────────────────────────┘
```

## Monitoreo (macOS host)

El script `zaino-monitor.sh` corre como servicio launchd y:

- **Cada 3 min** — Chequea que Home Assistant responda y que el container Docker esté corriendo
- **Cada 2s** — Detecta triggers de `/update` para actualizar el bot
- **Auto-recovery** — Reinicia el container si se cae, reinicia la VM de HA si no responde
- **Notificaciones** — Manda alertas por Telegram ante caídas y recuperaciones

### Servicios launchd

| Plist | Función |
|-------|---------|
| `com.zaino.monitor.plist` | Monitor unificado (HA + Docker + updates) |
| `com.zaino.telegram-bot-updater.plist` | Auto-updater (legacy, reemplazado por monitor) |

## Estructura del proyecto

```
telegram_zaino/
├── bot.py                              # Bot principal
├── Dockerfile                          # Python 3.12-slim + git
├── docker-compose.yml                  # Container config + healthcheck
├── requirements.txt                    # Dependencies
├── .env                                # Credenciales (no se commitea)
├── .env.example                        # Template de variables necesarias
├── .gitignore
├── zaino-monitor.sh                    # Monitor del host (HA + Docker)
├── auto-update.sh                      # Watcher de triggers de update
├── update.sh                           # Update por cron (legacy)
├── setup.sh                            # Setup inicial
├── com.zaino.monitor.plist             # launchd: monitor
├── com.zaino.telegram-bot-updater.plist # launchd: updater
└── .github/
    └── workflows/
        └── auto-merge-claude.yml       # Auto-merge PRs de Claude Code
```

## Variables de entorno

Ver `.env.example` para el template completo.

| Variable | Requerida | Descripción |
|----------|-----------|-------------|
| `ANTHROPIC_API_KEY` | Si | API key de Anthropic (Claude) |
| `TELEGRAM_BOT_TOKEN` | Si | Token del bot de Telegram |
| `TELEGRAM_USER_ID` | Si | ID numérico del usuario autorizado |
| `HA_URL` | Si | URL de Home Assistant |
| `HA_TOKEN` | Si | Long-lived access token de HA |
| `CLAUDE_MODEL` | No | Modelo de Claude (default: claude-sonnet-4-5-20250929) |
| `GEMINI_API_KEY` | No | API key de Google Gemini (para voz) |

## Instalación

```bash
# 1. Clonar
git clone git@github.com:matiasmacera/telegram_zaino.git
cd telegram_zaino

# 2. Configurar
cp .env.example .env
# Editar .env con las credenciales

# 3. Levantar
docker compose up -d --build

# 4. (Opcional) Instalar monitor en macOS
cp zaino-monitor.sh ~/Apps\ Zaino/zaino-monitor/
cp com.zaino.monitor.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.zaino.monitor.plist
```

## Flujo de actualización

1. Se pushean cambios a `main` en GitHub
2. GitHub Action auto-mergea PRs de ramas `claude/*`
3. Desde Telegram: `/update`
4. El monitor detecta el trigger, hace `git pull` + `docker compose rebuild`
5. El bot se reinicia con la nueva versión
