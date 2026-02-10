"""
Zaino Home Assistant Bot v2 - Telegram bot powered by Claude AI
Features:
- Text and voice message support (Gemini transcription)
- Full Home Assistant control via Claude tools
"""

import os
import json
import logging
import tempfile
from datetime import datetime, timedelta

import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from anthropic import AsyncAnthropic

# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])
HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.environ["HA_TOKEN"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TRIGGER_DIR = os.environ.get("TRIGGER_DIR", "/trigger")
HEALTH_FILE = "/tmp/bot_healthy"
MAX_CONVERSATION_MESSAGES = 20

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("zaino-bot")

# ─── Clients ──────────────────────────────────────────────────────────────────

claude = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
http_client = httpx.AsyncClient(
    base_url=HA_URL,
    headers={"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"},
    timeout=30.0,
)

# ─── Conversation History ────────────────────────────────────────────────────

conversations: dict[int, list[dict]] = {}


def get_conversation(user_id: int) -> list[dict]:
    if user_id not in conversations:
        conversations[user_id] = []
    return conversations[user_id]


def add_message(user_id: int, role: str, content):
    conv = get_conversation(user_id)
    conv.append({"role": role, "content": content})
    if len(conv) > MAX_CONVERSATION_MESSAGES:
        conversations[user_id] = conv[-MAX_CONVERSATION_MESSAGES:]


# ─── Voice Transcription ─────────────────────────────────────────────────────


async def transcribe_voice(file_path: str) -> str:
    """Transcribe audio using Google Gemini API."""
    if not GEMINI_API_KEY:
        return "[Error: GEMINI_API_KEY no configurada]"

    try:
        import base64

        with open(file_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": "audio/ogg",
                                        "data": audio_data,
                                    }
                                },
                                {
                                    "text": "Transcribí este audio a texto en español. Devolvé SOLO el texto transcrito, sin explicaciones ni formato adicional."
                                },
                            ]
                        }
                    ]
                },
            )
            response.raise_for_status()
            result = response.json()

            candidates = result.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()

            return "[Error: Gemini no devolvió transcripción]"

    except Exception as e:
        logger.error(f"Gemini transcription error: {e}")
        return f"[Error transcribiendo audio: {e}]"


# ─── Home Assistant API Helpers ───────────────────────────────────────────────


async def ha_get(path: str) -> dict | list | None:
    try:
        resp = await http_client.get(f"/api/{path}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"HA GET {path} error: {e}")
        return {"error": str(e)}


async def ha_post(path: str, data: dict = None) -> dict | list | None:
    try:
        resp = await http_client.post(f"/api/{path}", json=data or {})
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"status": "ok"}
    except Exception as e:
        logger.error(f"HA POST {path} error: {e}")
        return {"error": str(e)}


# ─── Tool Implementations ────────────────────────────────────────────────────


async def tool_list_entities(domain: str = None, search: str = None, limit: int = 50) -> str:
    states = await ha_get("states")
    if isinstance(states, dict) and "error" in states:
        return json.dumps(states)

    results = []
    for s in states:
        eid = s["entity_id"]
        name = s["attributes"].get("friendly_name", eid)
        state = s["state"]

        if domain and not eid.startswith(f"{domain}."):
            continue
        if search and search.lower() not in eid.lower() and search.lower() not in name.lower():
            continue

        entry = {"entity_id": eid, "name": name, "state": state}

        attrs = s.get("attributes", {})
        if eid.startswith("light.") and "brightness" in attrs:
            entry["brightness"] = attrs["brightness"]
        if eid.startswith("climate."):
            entry["temperature"] = attrs.get("temperature")
            entry["current_temperature"] = attrs.get("current_temperature")
            entry["hvac_mode"] = attrs.get("hvac_mode")
        if eid.startswith("sensor."):
            entry["unit"] = attrs.get("unit_of_measurement", "")

        results.append(entry)

    return json.dumps(results[:limit], ensure_ascii=False)


async def tool_get_entity(entity_id: str) -> str:
    state = await ha_get(f"states/{entity_id}")
    if isinstance(state, dict) and "error" in state:
        return json.dumps(state)
    return json.dumps(state, ensure_ascii=False, default=str)


async def tool_control_entity(entity_id: str, action: str, params: dict = None) -> str:
    domain = entity_id.split(".")[0]

    service_map = {
        "on": "turn_on", "off": "turn_off", "toggle": "toggle",
        "turn_on": "turn_on", "turn_off": "turn_off",
        "open": "open_cover", "close": "close_cover", "stop": "stop_cover",
        "lock": "lock", "unlock": "unlock",
        "arm_home": "alarm_arm_home", "arm_away": "alarm_arm_away", "disarm": "alarm_disarm",
    }

    service = service_map.get(action, action)
    data = {"entity_id": entity_id}
    if params:
        data.update(params)

    await ha_post(f"services/{domain}/{service}", data)
    return json.dumps({"status": "ok", "service": f"{domain}.{service}", "entity": entity_id, "params": params}, ensure_ascii=False)


async def tool_call_service(domain: str, service: str, data: dict = None) -> str:
    await ha_post(f"services/{domain}/{service}", data or {})
    return json.dumps({"status": "ok", "domain": domain, "service": service}, ensure_ascii=False)


async def tool_search_entities(query: str, limit: int = 20) -> str:
    return await tool_list_entities(search=query, limit=limit)


async def tool_get_history(entity_id: str, hours: int = 24) -> str:
    end = datetime.now()
    start = end - timedelta(hours=hours)
    path = f"history/period/{start.isoformat()}?filter_entity_id={entity_id}&end_time={end.isoformat()}&minimal_response"

    history = await ha_get(path)
    if isinstance(history, dict) and "error" in history:
        return json.dumps(history)

    if history and len(history) > 0:
        entries = history[0]
        if len(entries) > 50:
            entries = entries[:10] + [{"note": f"... {len(entries) - 20} more entries ..."}] + entries[-10:]
        return json.dumps(entries, ensure_ascii=False, default=str)

    return json.dumps({"message": "No history found"})


async def tool_get_ha_config() -> str:
    config = await ha_get("config")
    return json.dumps(config, ensure_ascii=False, default=str)


# ─── Tool Registry ───────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "list_entities": tool_list_entities,
    "get_entity": tool_get_entity,
    "control_entity": tool_control_entity,
    "call_service": tool_call_service,
    "search_entities": tool_search_entities,
    "get_history": tool_get_history,
    "get_ha_config": tool_get_ha_config,
}

TOOLS = [
    {
        "name": "list_entities",
        "description": "List Home Assistant entities. Can filter by domain (light, switch, sensor, climate, cover, lock, alarm_control_panel, media_player, vacuum, fan, etc.) or search by name/id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Filter by domain"},
                "search": {"type": "string", "description": "Search query to filter by name or entity_id"},
                "limit": {"type": "integer", "description": "Max results (default 50)"},
            },
        },
    },
    {
        "name": "get_entity",
        "description": "Get the full detailed state of a specific entity, including all attributes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "The entity ID (e.g. 'light.living', 'climate.quincho')"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "control_entity",
        "description": "Control a Home Assistant entity. Actions: on/off/toggle for lights/switches, open/close/stop for covers, lock/unlock for locks, arm_home/arm_away/disarm for alarm. Extra params: brightness (0-255), color_temp, rgb_color for lights; temperature/hvac_mode for climate; position (0-100) for covers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity to control"},
                "action": {"type": "string", "description": "Action to perform"},
                "params": {"type": "object", "description": "Additional parameters"},
            },
            "required": ["entity_id", "action"],
        },
    },
    {
        "name": "call_service",
        "description": "Call any Home Assistant service directly. Use for advanced operations like fan speed, media player controls, vacuum commands, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Service domain"},
                "service": {"type": "string", "description": "Service name"},
                "data": {"type": "object", "description": "Service data including entity_id and parameters"},
            },
            "required": ["domain", "service"],
        },
    },
    {
        "name": "search_entities",
        "description": "Search entities by name or partial ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term"},
                "limit": {"type": "integer", "description": "Max results (default 20)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_history",
        "description": "Get the state change history of an entity over a time period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity to get history for"},
                "hours": {"type": "integer", "description": "Hours of history (default 24)"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_ha_config",
        "description": "Get Home Assistant system configuration and version info.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Sos el asistente inteligente de la casa de Matías en Zaino 785, Pilar.
Controlás Home Assistant a través de herramientas.

La casa tiene:
- 92 luces (domótica cableada + Shelly + Ring)
- 29 climatizadores (Sensibo controlando aires por IR)
- 33 cortinas/covers motorizadas (blackouts, persianas, sombrillas)
- 33 cámaras Ring
- 3 cerraduras smart (Servicio, Pileta, Entrada)
- 3 robots aspiradora (Saros Z70, Roborock S7 MaxV, Doris)
- 18+ parlantes Sonos + HomePods
- Generador Generac
- Alarma
- Pileta con filtrado y llenado automático
- Apple TVs, Samsung The Frame, PlayStation 5
- 2 Home Assistant Voice (Escritorio y Quincho)

=== SKILL: PILETA ===
La pileta tiene MUCHOS sensores y controles. Cuando el usuario pregunte por la pileta, 
consultá TODAS las entidades relevantes para dar un reporte completo.

SENSORES DE AGUA (WaterGuru):
- sensor.waterguru_water_temperature → Temperatura del agua
- sensor.waterguru_ph → pH (ideal: 7.2-7.6). Tiene advice en atributos
- sensor.waterguru_ph_alert → Estado del pH (Ok/LOW/HIGH)
- sensor.waterguru_free_chlorine → Cloro libre (ideal: 1-3 ppm). Tiene advice
- sensor.waterguru_free_chlorine_alert → Estado cloro
- sensor.waterguru_total_alkalinity → Alcalinidad total (ideal: 80-120 ppm)
- sensor.waterguru_total_alkalinity_alert → Estado alcalinidad
- sensor.waterguru_calcium_hardness → Dureza cálcica
- sensor.waterguru_calcium_hardness_alert → Estado dureza
- sensor.waterguru_cyanuric_acid_stabilizer → Ácido cianúrico (estabilizador)
- sensor.waterguru_cyanuric_acid_stabilizer_alert → Estado estabilizador
- sensor.waterguru_total_hardness → Dureza total
- sensor.waterguru_skimmer_flow → Flujo del skimmer (gpm)
- sensor.waterguru_status → Estado general (GREEN/YELLOW/RED)
- sensor.waterguru_battery → Batería WaterGuru
- sensor.waterguru_cassette_days_remaining → Días restantes del cassette
- sensor.waterguru_cassette_remaining → % restante cassette
- sensor.waterguru_last_measurement → Última medición

MONITOR PILETA (segundo sensor):
- sensor.monitor_pileta_temperature → Temperatura (otro sensor)
- sensor.monitor_pileta_total_dissolved_solids → TDS (ppm)
- sensor.monitor_pileta_battery → Batería monitor

CLIMATIZADOR:
- climate.climatizador_pileta → Calefacción de pileta (current_temp, target_temp)

FILTRADO Y LLENADO (Shelly switches):
- switch.filtrado → Bomba de filtrado (on/off)
- switch.llenado → Llenado de agua (on/off)
- sensor.filtrado_power → Consumo actual filtrado (W)
- sensor.filtrado_energy → Energía acumulada filtrado (kWh)
- sensor.llenado_power → Consumo actual llenado (W)
- sensor.llenado_energy → Energía acumulada llenado (kWh)
- binary_sensor.filtrado_overcurrent → Alerta sobrecorriente
- binary_sensor.llenado_overcurrent → Alerta sobrecorriente

ILUMINACIÓN PILETA:
- light.shellyplusrgbwpm_2cbcbbc14718 → Pileta RGB (RGBW, colores)
- light.exterior_exterior_reflectores_pileta → Reflectores Pileta
- light.exterior_exterior_solado_pileta → Solado Pileta
- light.exterior_exterior_canteros_pileta → Canteros Pileta

COVERS/SOMBRILLAS:
- cover.sombrillas → Grupo: 3 sombrillas (gris 1, gris 2, roja)
- cover.sombrilla_gris_1, cover.sombrilla_gris_2, cover.sombrilla_roja
- cover.cortinas_pileta → Grupo: cortinas zona pileta

SEGURIDAD PILETA:
- lock.pileta → Cerradura lateral pileta (batería: sensor.pileta_battery)
- binary_sensor.reja_jardin_pileta → Reja jardín-pileta (abierta/cerrada)
- binary_sensor.reja_pileta_lateral → Reja pileta lateral
- binary_sensor.puerta_bano_pileta → Puerta baño pileta

SONOS PILETA:
- media_player.pileta → Sonos Pileta

RIEGO ZONA PILETA:
- switch.canteros_pileta_y_galeria → Riego canteros pileta
- switch.fondo_y_cantero_derecho_pileta → Riego fondo pileta

Cuando el usuario pida estado de la pileta, hacé un reporte completo con:
1. Temperatura agua (ambos sensores) y climatizador
2. Química: pH, cloro, alcalinidad, estabilizador, dureza + alertas/consejos del WaterGuru
3. Estado filtrado/llenado + consumo
4. Estado WaterGuru (cassette, batería)
5. Luces y sombrillas si es relevante
6. Seguridad (rejas, cerradura, puerta) si es relevante

Usá emojis para hacerlo visual: 🌡️ 🧪 💧 ⚗️ 🔬 💡 ☂️ 🔒

=== FIN SKILLS ===

Respondé siempre en español rioplatense, de forma concisa y directa.
Cuando controles dispositivos, confirmá la acción brevemente.
Si no estás seguro de un entity_id, usá search_entities primero.
Podés ejecutar múltiples tools en secuencia si es necesario.

AUDIO: Los mensajes de voz del usuario ya fueron transcritos a texto, procesalos normalmente.
"""


# ─── Claude Integration ──────────────────────────────────────────────────────


async def process_tool_call(tool_name: str, tool_input: dict) -> str:
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return await func(**tool_input)
    except Exception as e:
        logger.error(f"Tool {tool_name} error: {e}")
        return json.dumps({"error": str(e)})


async def chat_with_claude(user_id: int, message: str) -> str:
    add_message(user_id, "user", message)
    messages = get_conversation(user_id)

    try:
        response = await claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        max_iterations = 10
        iteration = 0

        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            assistant_content = response.content
            add_message(user_id, "assistant", assistant_content)

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    logger.info(f"Tool: {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                    result = await process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            add_message(user_id, "user", tool_results)

            response = await claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=get_conversation(user_id),
            )

        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        add_message(user_id, "assistant", response.content)
        return final_text or "✅ Listo"

    except Exception as e:
        logger.error(f"Claude error: {e}")
        return f"❌ Error: {str(e)}"


# ─── Telegram Handlers ───────────────────────────────────────────────────────


def authorized(func):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != TELEGRAM_USER_ID:
            await update.message.reply_text("⛔ No autorizado.")
            logger.warning(f"Unauthorized access from {update.effective_user.id}")
            return
        return await func(update, context)
    return wrapper


@authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏠 *Zaino Home Assistant Bot*\n\n"
        "Mandame un mensaje de texto o audio y controlo tu casa.\n\n"
        "Ejemplos:\n"
        "• _Prendé las luces del quincho_\n"
        "• _¿Qué temperatura hay en el living?_\n"
        "• _Cerrá todas las cortinas_\n"
        "• 🎤 _Mandá un audio pidiendo lo que quieras_\n\n"
        "Comandos:\n"
        "/reset - Limpiar conversación\n"
        "/status - Estado general de la casa\n"
        "/pileta - Estado completo de la pileta\n"
        "/update - Actualizar bot desde GitHub",
        parse_mode="Markdown",
    )


@authorized
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversations[update.effective_user.id] = []
    await update.message.reply_text("🧹 Conversación limpiada.")


@authorized
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")
    response = await chat_with_claude(
        update.effective_user.id,
        "Dame un resumen rápido del estado de la casa: luces encendidas, temperatura de los aires encendidos, estado de la alarma, y estado de las cerraduras. Sé conciso.",
    )
    await send_long_message(update, response)


@authorized
async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger bot update from GitHub."""
    try:
        os.makedirs(TRIGGER_DIR, exist_ok=True)
        trigger_file = os.path.join(TRIGGER_DIR, "update")
        with open(trigger_file, "w") as f:
            f.write(datetime.now().isoformat())
        await update.message.reply_text(
            "🔄 Update disparado. El bot se va a reiniciar en unos segundos con la última versión del repo."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_pileta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")
    response = await chat_with_claude(
        update.effective_user.id,
        "Dame el estado completo de la pileta: temperatura del agua (ambos sensores), química del agua (pH, cloro, alcalinidad, estabilizador, dureza con alertas y consejos del WaterGuru), estado del filtrado y llenado con consumo, estado del cassette WaterGuru, y cualquier alerta importante. Sé completo pero organizado.",
    )
    await send_long_message(update, response)


@authorized
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action("typing")
    response = await chat_with_claude(update.effective_user.id, update.message.text)
    await send_long_message(update, response)


@authorized
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not GEMINI_API_KEY:
        await update.message.reply_text("🎤 Configurá GEMINI_API_KEY en el .env para usar voz.")
        return

    await update.message.chat.send_action("typing")

    try:
        voice_file = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            await voice_file.download_to_drive(f.name)
            temp_path = f.name

        transcribed_text = await transcribe_voice(temp_path)
        os.unlink(temp_path)

        if not transcribed_text or transcribed_text.startswith("[Error"):
            await update.message.reply_text(f"❌ No pude transcribir el audio: {transcribed_text}")
            return

        await update.message.reply_text(f"🎤 _{transcribed_text}_", parse_mode="Markdown")

        await update.message.chat.send_action("typing")
        response = await chat_with_claude(update.effective_user.id, transcribed_text)
        await send_long_message(update, response)

    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text(f"❌ Error procesando audio: {e}")


async def send_long_message(update: Update, text: str):
    if not text:
        return
    if len(text) > 4000:
        for chunk in [text[i:i + 4000] for i in range(0, len(text), 4000)]:
            try:
                await update.message.reply_text(chunk, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(chunk)
    else:
        try:
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    logger.info("Starting Zaino Home Assistant Bot...")

    # Write initial health file
    with open(HEALTH_FILE, "w") as f:
        f.write(datetime.now().isoformat())

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("update", cmd_update))
    app.add_handler(CommandHandler("pileta", cmd_pileta))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Health heartbeat: update file every 30s so Docker knows we're alive
    async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
        with open(HEALTH_FILE, "w") as f:
            f.write(datetime.now().isoformat())

    app.job_queue.run_repeating(heartbeat, interval=30, first=10)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
