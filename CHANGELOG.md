# Changelog

## v0.9.1 — 2025-02-11
- Test: bump de versión para verificar flujo de actualización
- El bot ahora notifica `v0.9.0 → v0.9.1` al reiniciar tras `/update`

## v0.9.0 — 2025-02-11
- CI: GitHub Action para auto-merge de PRs desde ramas `claude/*`
- Refactor: `zaino-monitor.sh` lee credenciales del `.env` (sin más tokens hardcodeados)
- Agregado `.env.example` como template de configuración
- Documentación: README.md y CHANGELOG.md

## v0.8.0
- Feat: comando `/settoken` para actualizar el token de HA remotamente desde Telegram
- El token se persiste en volumen Docker y sobrevive reinicios

## v0.7.0
- Fix: manejo de errores en tools de control y servicios de HA
- Fix: handler de voz más robusto ante fallos de transcripción

## v0.6.0
- Fix: indicador de typing se mantiene activo durante operaciones largas (refresh cada 4s)

## v0.5.0
- Fix: parsing de Markdown en reportes de WaterGuru (escape de texto de advice, fallback a plain text)

## v0.4.0
- Feat: comandos `/logs` y `/version` para administración remota
- `/logs [N]` muestra las últimas N líneas de logs del container
- `/version` muestra commits recientes y uptime

## v0.3.0
- Feat: notificación automática de WaterGuru por polling (cada 30 min)
- Detecta nuevas mediciones y envía reporte formateado con química completa
- Eliminado webhook server (no requiere config en HA)

## v0.2.0
- Feat: notificación al iniciar/reiniciar el bot
- Feat: skill de música/Sonos — control multiroom, agrupación, volumen, playback
- Feat: skill `/pileta` — monitoreo completo con química WaterGuru
- Feat: `zaino-monitor.sh` + plist de launchd para monitoreo unificado
- Feat: health check con heartbeat (Docker auto-restart si el bot se cuelga)
- Feat: comando `/update` + servicio de auto-update

## v0.1.0
- Initial release: bot de Telegram con Claude AI + transcripción de voz con Gemini
- Control de Home Assistant por lenguaje natural
- Tools: list_entities, get_entity, control_entity, call_service, search_entities, get_history, get_ha_config
- Auth por TELEGRAM_USER_ID
- Docker con restart: always
