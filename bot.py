"""
Zaino Home Assistant Bot v2 - Telegram bot powered by Claude AI
Features:
- Text and voice message support (Gemini transcription)
- Full Home Assistant control via Claude tools
"""

import os
import sys
import json
import signal
import logging
import tempfile
import asyncio
import functools
from datetime import datetime, timedelta
from time import time as monotime

import httpx
from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from anthropic import AsyncAnthropic

# ─── Config Validation ────────────────────────────────────────────────────────

REQUIRED_ENV = ["ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_USER_ID", "HA_TOKEN"]
_missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    print(f"ERROR: Faltan variables de entorno requeridas: {', '.join(_missing)}", file=sys.stderr)
    print("Revisá tu archivo .env o docker-compose.yml", file=sys.stderr)
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_USER_ID = int(os.environ["TELEGRAM_USER_ID"])
HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.environ["HA_TOKEN"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TRIGGER_DIR = os.environ.get("TRIGGER_DIR", "/trigger")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
WATERGURU_POLL_INTERVAL = int(os.environ.get("WATERGURU_POLL_INTERVAL", "1800"))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "3"))
HEALTH_FILE = os.path.join(DATA_DIR, "bot_healthy")
WATERGURU_LAST_FILE = os.path.join(DATA_DIR, "waterguru_last")
HA_TOKEN_FILE = os.path.join(TRIGGER_DIR, "ha_token")
PREV_VERSION_FILE = os.path.join(TRIGGER_DIR, "prev_version")

os.makedirs(DATA_DIR, exist_ok=True)

# Load version
try:
    with open("VERSION", "r") as f:
        BOT_VERSION = f.read().strip()
except FileNotFoundError:
    BOT_VERSION = "unknown"

# Use persisted token from volume if available (set via /settoken)
try:
    with open(HA_TOKEN_FILE, "r") as f:
        _saved_token = f.read().strip()
    if _saved_token:
        HA_TOKEN = _saved_token
        logging.info("Using HA token from persisted file")
except FileNotFoundError:
    pass
MAX_CONVERSATION_MESSAGES = 20

# Rate limiting state
_last_message_time: dict[int, float] = {}

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

    result = await ha_post(f"services/{domain}/{service}", data)
    if isinstance(result, dict) and "error" in result:
        return json.dumps(result)
    return json.dumps({"status": "ok", "service": f"{domain}.{service}", "entity": entity_id, "params": params}, ensure_ascii=False)


async def tool_call_service(domain: str, service: str, data: dict = None) -> str:
    result = await ha_post(f"services/{domain}/{service}", data or {})
    if isinstance(result, dict) and "error" in result:
        return json.dumps(result)
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

SYSTEM_PROMPT_FILE = os.environ.get("SYSTEM_PROMPT_FILE", "system_prompt.txt")
try:
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        SYSTEM_PROMPT = f.read().strip()
    logger.info(f"System prompt loaded from {SYSTEM_PROMPT_FILE}")
except FileNotFoundError:
    logger.error(f"System prompt file not found: {SYSTEM_PROMPT_FILE}")
    sys.exit(1)


# ─── Claude Integration ──────────────────────────────────────────────────────


async def process_tool_call(tool_name: str, tool_input: dict) -> str:
    func = TOOL_FUNCTIONS.get(tool_name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = await func(**tool_input)
        logger.info(f"Tool {tool_name} result: {result[:200]}{'...' if len(result) > 200 else ''}")
        return result
    except Exception as e:
        logger.error(f"Tool {tool_name} error: {e}")
        return json.dumps({"error": str(e)})


async def chat_with_claude(user_id: int, message: str) -> str:
    add_message(user_id, "user", message)
    messages = get_conversation(user_id)

    try:
        response = await asyncio.wait_for(
            claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            ),
            timeout=120,
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

            response = await asyncio.wait_for(
                claude.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=get_conversation(user_id),
                ),
                timeout=120,
            )

        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        add_message(user_id, "assistant", response.content)
        return final_text or "✅ Listo"

    except asyncio.TimeoutError:
        logger.error("Claude API timeout (120s)")
        return "⏱ Claude tardó demasiado en responder. Intentá de nuevo."
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return "⚠️ Hubo un problema procesando tu mensaje. Intentá de nuevo en unos segundos."


# ─── Telegram Handlers ───────────────────────────────────────────────────────


async def keep_typing(chat, stop_event):
    """Keep sending 'typing' action every 4s until stop_event is set."""
    while not stop_event.is_set():
        try:
            await chat.send_action("typing")
        except Exception:
            pass
        await asyncio.sleep(4)


async def run_with_typing(update, coro):
    """Run a coroutine while showing typing indicator."""
    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(update.message.chat, stop_event))
    try:
        result = await coro
        return result
    finally:
        stop_event.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass


def authorized(func):
    @functools.wraps(func)
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
        "/history - Ver historial de conversación\n"
        "/status - Estado general de la casa\n"
        "/pileta - Estado completo de la pileta\n"
        "/musica - ¿Qué suena en la casa?\n"
        "/logs - Ver logs del bot\n"
        "/version - Versión y uptime\n"
        "/update - Actualizar bot desde GitHub",
        parse_mode="Markdown",
    )


@authorized
async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversations[update.effective_user.id] = []
    await update.message.reply_text("🧹 Conversación limpiada.")


@authorized
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await run_with_typing(update, chat_with_claude(
        update.effective_user.id,
        "Dame un resumen rápido del estado de la casa: luces encendidas, temperatura de los aires encendidos, estado de la alarma, y estado de las cerraduras. Sé conciso.",
    ))
    await send_long_message(update, response)


@authorized
async def cmd_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger bot update from GitHub."""
    try:
        os.makedirs(TRIGGER_DIR, exist_ok=True)
        trigger_file = os.path.join(TRIGGER_DIR, "update")
        # Save who requested the update for post-update notification
        update_info = {
            "time": datetime.now().isoformat(),
            "version": BOT_VERSION,
            "user_id": update.effective_user.id,
        }
        with open(trigger_file, "w") as f:
            json.dump(update_info, f)
        await update.message.reply_text(
            "🔄 Update disparado. El bot se va a reiniciar en unos segundos con la última versión del repo."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_settoken(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Update HA token remotely. Usage: /settoken <new_token>"""
    # Delete the message with the token for security
    try:
        await update.message.delete()
    except Exception:
        pass

    args = context.args
    if not args:
        await update.message.chat.send_message("Uso: `/settoken <nuevo_token_HA>`", parse_mode="Markdown")
        return

    new_token = args[0]

    # Validate the new token against HA
    try:
        async with httpx.AsyncClient(timeout=10.0) as test_client:
            resp = await test_client.get(
                f"{HA_URL}/api/config",
                headers={"Authorization": f"Bearer {new_token}"},
            )
            resp.raise_for_status()
    except Exception as e:
        await update.message.chat.send_message(f"❌ Token inválido. HA respondió: {e}")
        return

    # Save to persistent volume
    try:
        os.makedirs(TRIGGER_DIR, exist_ok=True)
        with open(HA_TOKEN_FILE, "w") as f:
            f.write(new_token)
    except Exception as e:
        await update.message.chat.send_message(f"❌ No pude guardar el token: {e}")
        return

    # Update in-memory client
    http_client.headers["Authorization"] = f"Bearer {new_token}"

    await update.message.chat.send_message("✅ Token de Home Assistant actualizado y guardado. Funciona incluso después de reiniciar.")
    logger.info("HA token updated via /settoken")


@authorized
async def cmd_pileta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await run_with_typing(update, chat_with_claude(
        update.effective_user.id,
        "Dame el estado completo de la pileta: temperatura del agua (ambos sensores), química del agua (pH, cloro, alcalinidad, estabilizador, dureza con alertas y consejos del WaterGuru), estado del filtrado y llenado con consumo, estado del cassette WaterGuru, y cualquier alerta importante. Sé completo pero organizado.",
    ))
    await send_long_message(update, response)


@authorized
async def cmd_musica(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = await run_with_typing(update, chat_with_claude(
        update.effective_user.id,
        "Dame el estado de la música en la casa: qué parlantes están reproduciendo algo, qué suena en cada uno, volumen, y si hay grupos armados. Solo mostrá los que estén activos (playing/paused), no los idle.",
    ))
    await send_long_message(update, response)


@authorized
async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show last N lines of bot logs."""
    try:
        n = 30
        args = context.args
        if args and args[0].isdigit():
            n = min(int(args[0]), 100)

        import subprocess
        result = subprocess.run(
            ["tail", "-n", str(n), "/proc/1/fd/1"],
            capture_output=True, text=True, timeout=5
        )
        # Fallback: read from docker logs via internal log
        if not result.stdout.strip():
            lines = []
            for h in logging.getLogger().handlers:
                if hasattr(h, 'stream'):
                    lines.append("(logs only available via docker logs)")
                    break
            output = "\n".join(lines) if lines else "No logs available. Usá `docker logs zaino-telegram-bot --tail 30` en el host."
        else:
            output = result.stdout.strip()

        if len(output) > 4000:
            output = output[-4000:]

        await update.message.reply_text(f"```\n{output}\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_version(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current bot version and uptime."""
    try:
        # Git info from file generated at build time
        try:
            with open("git_info.txt", "r") as f:
                git_info = f.read().strip()
        except FileNotFoundError:
            git_info = "Git info not available"

        # Uptime
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.read().split()[0])
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)

        # Container start time
        start_time = datetime.now() - timedelta(seconds=uptime_seconds)

        text = (
            f"🤖 *Zaino Bot v{BOT_VERSION}*\n\n"
            f"⏱ Uptime: {hours}h {minutes}m\n"
            f"🕐 Inicio: {start_time.strftime('%d/%m %H:%M')}\n\n"
            f"📦 Últimos commits:\n```\n{git_info}\n```"
        )
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


@authorized
async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current conversation history summary."""
    user_id = update.effective_user.id
    conv = get_conversation(user_id)
    if not conv:
        await update.message.reply_text("No hay conversación activa. Mandame un mensaje para empezar.")
        return

    lines = [f"*Conversación activa:* {len(conv)} mensajes\n"]
    for i, msg in enumerate(conv):
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            if isinstance(content, str):
                preview = content[:80] + ("..." if len(content) > 80 else "")
                lines.append(f"`{i+1}.` Tu: _{preview}_")
            elif isinstance(content, list):
                lines.append(f"`{i+1}.` Tu: (resultado de tool)")
        elif role == "assistant":
            if isinstance(content, list):
                texts = [b.text for b in content if hasattr(b, "text")]
                tools = [b.name for b in content if hasattr(b, "name")]
                if texts:
                    preview = texts[0][:80] + ("..." if len(texts[0]) > 80 else "")
                    lines.append(f"`{i+1}.` Bot: _{preview}_")
                if tools:
                    lines.append(f"      Tools: {', '.join(tools)}")
            elif isinstance(content, str):
                preview = content[:80] + ("..." if len(content) > 80 else "")
                lines.append(f"`{i+1}.` Bot: _{preview}_")

    text = "\n".join(lines)
    try:
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(text.replace("*", "").replace("_", "").replace("`", ""))


@authorized
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    now = monotime()
    last = _last_message_time.get(user_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        return
    _last_message_time[user_id] = now

    response = await run_with_typing(update, chat_with_claude(user_id, update.message.text))
    await send_long_message(update, response)


@authorized
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not GEMINI_API_KEY:
        await update.message.reply_text("🎤 Configurá GEMINI_API_KEY en el .env para usar voz.")
        return

    await update.message.chat.send_action("typing")

    temp_path = None
    try:
        voice_file = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            await voice_file.download_to_drive(f.name)
            temp_path = f.name

        transcribed_text = await transcribe_voice(temp_path)

        if not transcribed_text or transcribed_text.startswith("[Error"):
            await update.message.reply_text(f"❌ No pude transcribir el audio: {transcribed_text}")
            return

        try:
            await update.message.reply_text(f"🎤 _{transcribed_text}_", parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(f"🎤 {transcribed_text}")

        response = await run_with_typing(update, chat_with_claude(update.effective_user.id, transcribed_text))
        await send_long_message(update, response)

    except httpx.TimeoutException:
        logger.error("Voice transcription timeout")
        await update.message.reply_text("⏱ El audio tardó demasiado en transcribir. Probá con uno más corto.")
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text("⚠️ No pude procesar el audio. Intentá de nuevo.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


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


# ─── WaterGuru Polling ────────────────────────────────────────────────────────

telegram_bot: Bot = None


async def send_waterguru_report():
    """Fetch all WaterGuru sensors and send a formatted report."""
    try:
        sensor_ids = [
            "sensor.waterguru_water_temperature",
            "sensor.waterguru_ph",
            "sensor.waterguru_ph_alert",
            "sensor.waterguru_free_chlorine",
            "sensor.waterguru_free_chlorine_alert",
            "sensor.waterguru_total_alkalinity",
            "sensor.waterguru_total_alkalinity_alert",
            "sensor.waterguru_cyanuric_acid_stabilizer",
            "sensor.waterguru_cyanuric_acid_stabilizer_alert",
            "sensor.waterguru_calcium_hardness",
            "sensor.waterguru_calcium_hardness_alert",
            "sensor.waterguru_total_hardness",
            "sensor.waterguru_total_hardness_alert",
            "sensor.waterguru_skimmer_flow",
            "sensor.waterguru_skimmer_flow_alert",
            "sensor.waterguru_status",
            "sensor.waterguru_cassette_days_remaining",
            "sensor.waterguru_cassette_remaining",
            "sensor.monitor_pileta_temperature",
            "sensor.monitor_pileta_total_dissolved_solids",
            "switch.filtrado",
            "switch.llenado",
        ]

        data = {}
        for eid in sensor_ids:
            try:
                resp = await http_client.get(f"/api/states/{eid}")
                if resp.status_code == 200:
                    s = resp.json()
                    data[eid] = {
                        "state": s["state"],
                        "attrs": s.get("attributes", {}),
                    }
            except Exception:
                pass

        status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(
            data.get("sensor.waterguru_status", {}).get("state", ""), "⚪"
        )

        def get_val(eid):
            return data.get(eid, {}).get("state", "?")

        def get_alert(eid):
            s = data.get(eid, {})
            state = s.get("state", "")
            color = s.get("attrs", {}).get("status_color", "")
            emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(color, "")
            return f"{emoji} {state}" if state and state != "Ok" else "🟢"

        def get_advice(eid):
            return data.get(eid, {}).get("attrs", {}).get("advice", "")

        temp = get_val("sensor.waterguru_water_temperature")
        try:
            temp_str = f"{float(temp):.1f}°C"
        except (ValueError, TypeError):
            temp_str = f"{temp}"

        lines = [
            f"🏊 *Medición WaterGuru* {status_emoji}",
            "",
            f"🌡️ *Agua:* {temp_str}",
        ]

        monitor_temp = get_val("sensor.monitor_pileta_temperature")
        if monitor_temp != "?":
            lines.append(f"🌡️ *Monitor:* {monitor_temp}°C")

        tds = get_val("sensor.monitor_pileta_total_dissolved_solids")
        if tds != "?":
            lines.append(f"💧 *TDS:* {tds} ppm")

        lines.append("")
        lines.append("*Química del agua:*")

        ph = get_val("sensor.waterguru_ph")
        ph_alert = get_alert("sensor.waterguru_ph_alert")
        lines.append(f"⚗️ pH: *{ph}* {ph_alert}")
        advice = get_advice("sensor.waterguru_ph_alert")
        if advice:
            lines.append(f"   → {advice}")

        cl = get_val("sensor.waterguru_free_chlorine")
        cl_alert = get_alert("sensor.waterguru_free_chlorine_alert")
        lines.append(f"🧪 Cloro: *{cl} ppm* {cl_alert}")
        advice = get_advice("sensor.waterguru_free_chlorine_alert")
        if advice:
            lines.append(f"   → {advice}")

        alk = get_val("sensor.waterguru_total_alkalinity")
        alk_alert = get_alert("sensor.waterguru_total_alkalinity_alert")
        lines.append(f"💧 Alcalinidad: *{alk} ppm* {alk_alert}")

        cya = get_val("sensor.waterguru_cyanuric_acid_stabilizer")
        cya_alert = get_alert("sensor.waterguru_cyanuric_acid_stabilizer_alert")
        lines.append(f"🛡️ Estabilizador: *{cya} ppm* {cya_alert}")
        advice = get_advice("sensor.waterguru_cyanuric_acid_stabilizer_alert")
        if advice:
            lines.append(f"   → {advice}")

        hard = get_val("sensor.waterguru_calcium_hardness")
        hard_alert = get_alert("sensor.waterguru_calcium_hardness_alert")
        lines.append(f"🧱 Dureza: *{hard} ppm* {hard_alert}")

        flow = get_val("sensor.waterguru_skimmer_flow")
        flow_alert = get_alert("sensor.waterguru_skimmer_flow_alert")
        lines.append(f"🌊 Flujo: *{flow} gpm* {flow_alert}")

        cassette_days = get_val("sensor.waterguru_cassette_days_remaining")
        cassette_pct = get_val("sensor.waterguru_cassette_remaining")
        lines.append("")
        lines.append(f"📦 Cassette: {cassette_pct}% ({cassette_days} días)")

        filtrado = get_val("switch.filtrado")
        llenado = get_val("switch.llenado")
        filt_str = "🟢 ON" if filtrado == "on" else "⚪ OFF"
        llen_str = "🟢 ON" if llenado == "on" else "⚪ OFF"
        lines.append(f"⚙️ Filtrado: {filt_str} | Llenado: {llen_str}")

        report = "\n".join(lines)

        # Try Markdown first, fallback to plain text
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=report,
                parse_mode="Markdown",
            )
        except Exception:
            # Strip markdown formatting and send plain
            plain = report.replace("*", "").replace("_", "")
            await telegram_bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=plain,
            )
        logger.info("WaterGuru report sent")

    except Exception as e:
        logger.error(f"WaterGuru report error: {e}")
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=f"🏊 Nueva medición WaterGuru pero hubo error: {e}",
            )
        except Exception:
            pass


async def waterguru_poll(context):
    """Poll WaterGuru last_measurement every 30 min. Send report if new."""
    try:
        resp = await http_client.get("/api/states/sensor.waterguru_last_measurement")
        if resp.status_code != 200:
            return
        current = resp.json()["state"]

        last = ""
        try:
            with open(WATERGURU_LAST_FILE, "r") as f:
                last = f.read().strip()
        except FileNotFoundError:
            with open(WATERGURU_LAST_FILE, "w") as f:
                f.write(current)
            logger.info(f"WaterGuru poll: initial value saved ({current})")
            return

        if current != last and current not in ("unknown", "unavailable", ""):
            logger.info(f"WaterGuru: new measurement detected ({last} -> {current})")
            with open(WATERGURU_LAST_FILE, "w") as f:
                f.write(current)
            await send_waterguru_report()
        else:
            logger.debug(f"WaterGuru poll: no change ({current})")

    except Exception as e:
        logger.error(f"WaterGuru poll error: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    global telegram_bot
    logger.info(f"Starting Zaino Home Assistant Bot v{BOT_VERSION}...")

    # Write initial health file
    with open(HEALTH_FILE, "w") as f:
        f.write(datetime.now().isoformat())

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_bot = app.bot

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("update", cmd_update))
    app.add_handler(CommandHandler("settoken", cmd_settoken))
    app.add_handler(CommandHandler("pileta", cmd_pileta))
    app.add_handler(CommandHandler("musica", cmd_musica))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("version", cmd_version))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))

    # Health heartbeat: update file every 30s so Docker knows we're alive
    async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
        try:
            with open(HEALTH_FILE, "w") as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            logger.error(f"Heartbeat write error: {e}")

    # Send startup notification with version info
    async def startup_notify(context: ContextTypes.DEFAULT_TYPE):
        try:
            # Check if there was a previous version (update scenario)
            prev_version = None
            try:
                with open(PREV_VERSION_FILE, "r") as f:
                    prev_version = f.read().strip()
            except FileNotFoundError:
                pass

            # Save current version for next restart
            os.makedirs(TRIGGER_DIR, exist_ok=True)
            with open(PREV_VERSION_FILE, "w") as f:
                f.write(BOT_VERSION)

            if prev_version and prev_version != BOT_VERSION:
                text = (
                    f"🔄 *Zaino Bot actualizado*\n"
                    f"v{prev_version} → v{BOT_VERSION}\n\n"
                    f"Conectado y listo."
                )
            else:
                text = f"✅ *Zaino Bot v{BOT_VERSION} iniciado*\nConectado y listo."

            await context.bot.send_message(
                chat_id=TELEGRAM_USER_ID,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Startup notification error: {e}")

    app.job_queue.run_repeating(heartbeat, interval=30, first=10)
    app.job_queue.run_repeating(waterguru_poll, interval=WATERGURU_POLL_INTERVAL, first=60)
    app.job_queue.run_once(startup_notify, when=2)

    # Graceful shutdown handler
    def graceful_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
