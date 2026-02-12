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
from datetime import datetime, timedelta, timezone
from time import time as monotime

import httpx
from collections import Counter, defaultdict
from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from anthropic import AsyncAnthropic
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
import websockets

# ─── Config Validation ────────────────────────────────────────────────────────

REQUIRED_ENV = ["ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_USER_ID", "HA_TOKEN"]
_missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if _missing:
    print(f"ERROR: Faltan variables de entorno requeridas: {', '.join(_missing)}", file=sys.stderr)
    print("Revisá tus archivos .env / .env.config o docker-compose.yml", file=sys.stderr)
    sys.exit(1)

# ─── Config ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
_raw_ids = [int(uid.strip()) for uid in os.environ["TELEGRAM_USER_ID"].split(",")]
TELEGRAM_ADMIN_ID = _raw_ids[0]  # First ID = admin, receives WaterGuru notifications
TELEGRAM_USER_IDS = set(_raw_ids)
HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.environ["HA_TOKEN"]
USER_NAMES: dict[int, str] = {}
for _pair in os.environ.get("TELEGRAM_USER_NAMES", "").split(","):
    if ":" in _pair:
        _uid, _name = _pair.strip().split(":", 1)
        USER_NAMES[int(_uid)] = _name.strip()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TRIGGER_DIR = os.environ.get("TRIGGER_DIR", "/trigger")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
WATERGURU_POLL_INTERVAL = int(os.environ.get("WATERGURU_POLL_INTERVAL", "1800"))
RATE_LIMIT_SECONDS = int(os.environ.get("RATE_LIMIT_SECONDS", "3"))
LOCAL_TZ = timezone(timedelta(hours=int(os.environ.get("TZ_OFFSET", "-3"))))
WEATHER_LAT = os.environ.get("WEATHER_LAT", "-34.46")   # Pilar, Buenos Aires
WEATHER_LON = os.environ.get("WEATHER_LON", "-58.91")
WEATHER_POLL_INTERVAL = int(os.environ.get("WEATHER_POLL_INTERVAL", "900"))  # 15 min
INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://influxdb:8086")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "zaino")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "homeassistant")
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

# HA Event Bus config
HA_WS_URL = HA_URL.replace("https://", "wss://").replace("http://", "ws://") + "/api/websocket"
HA_EVENT_TRACKED_DOMAINS = {
    # Actionable entities — full tracking, no filtering
    "light", "switch", "cover", "lock", "climate", "media_player",
    "alarm_control_panel", "fan", "vacuum", "input_boolean",
    # Observational entities — pattern-filtered to avoid noise
    "binary_sensor", "sensor", "camera",
    # Presence tracking
    "person", "device_tracker",
}

# Pattern filters for noisy domains (match against entity_id)
HA_SENSOR_PATTERNS = {
    "battery", "temperature", "temp_", "humidity",
    "power", "energy", "consumption",
    "air_quality", "pm25", "pm2_5", "pm10", "co2", "voc", "tvoc",
    "generac", "generador",
    "signal_strength", "rssi", "wireless",
    # Weather station (WeatherFlow Tempest)
    "wind", "precipitation", "lightning", "illuminance",
    "irradiance", "uv_index", "pressure", "dew_point",
    "feels_like", "air_density",
    # Pool chemistry
    "waterguru", "monitor_pileta",
    # Vacuum robots
    "roborock", "saros_z70", "doris",
    # Alarm
    "alarm_info",
    # LG ThinQ appliances (washers)
    "lavadora",
    # UniFi network infrastructure
    "usw",
}
HA_BINARY_SENSOR_PATTERNS = {
    "door", "window", "motion", "reja", "puerta", "gate",
    "overcurrent", "tamper", "smoke", "gas", "leak", "water",
    "occupancy", "vibration", "opening", "contact",
    # Spanish names (Z-Wave sensors)
    "movimiento", "ventana",
    # Shelly safety alerts
    "overheating", "overvoltage", "overpowering",
    # Generator alerts
    "generac",
    # Flood/freeze sensors
    "flood", "freeze",
    # AC filter maintenance
    "filter_clean",
    # Doorbell
    "ding",
    # LG ThinQ appliances
    "lavadora", "remote_start",
}
# Device tracker patterns — only track personal devices (phones/tablets),
# infrastructure devices (APs, switches, Sonos) rarely change state so
# they are tracked unfiltered via state_changed events naturally.
HA_DEVICE_TRACKER_PATTERNS = {
    "iphone", "ipad", "pixel", "galaxy", "phone", "mobile",
    "macbook", "laptop",
}

# Rate limiting for sensor domain (seconds between writes per entity)
HA_SENSOR_MIN_INTERVAL = 300  # 5 min — avoids flooding from oscillating values
_event_rate_limit: dict[str, float] = {}

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

# InfluxDB client (optional - for analytics, lazy-initialized)
_influx_client: InfluxDBClientAsync | None = None


def get_influx_client() -> InfluxDBClientAsync | None:
    """Lazy-init InfluxDB async client (must be called inside async context)."""
    global _influx_client
    if not INFLUXDB_TOKEN:
        return None
    if _influx_client is None:
        _influx_client = InfluxDBClientAsync(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG,
        )
        logger.info("InfluxDB client initialized")
    return _influx_client


if not INFLUXDB_TOKEN:
    logger.warning("INFLUXDB_TOKEN not set — analytics features disabled")


async def write_points_to_influx(*points):
    """Write points to InfluxDB (fire-and-forget, logs errors)."""
    client = get_influx_client()
    if not client:
        return
    try:
        write_api = client.write_api()
        await write_api.write(bucket=INFLUXDB_BUCKET, record=list(points))
    except Exception as e:
        logger.error(f"InfluxDB write error: {e}")


# Track tools used in last interaction (populated by chat_with_claude)
_last_tools_used: dict[int, list[str]] = {}


async def log_bot_interaction(user_id: int, msg_type: str, message: str, response_time_ms: int = 0):
    """Log a bot interaction to InfluxDB."""
    tools = _last_tools_used.pop(user_id, [])
    point = (
        Point("bot_interaction")
        .tag("user_id", str(user_id))
        .tag("user_name", USER_NAMES.get(user_id, str(user_id)))
        .tag("msg_type", msg_type)
        .field("message", message[:200])
        .field("response_time_ms", response_time_ms)
        .field("tools_used", ",".join(tools) if tools else "none")
        .field("tool_count", len(tools))
    )
    await write_points_to_influx(point)


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


# ─── InfluxDB Analytics Tools ────────────────────────────────────────────────


async def tool_query_entity_history(entity_id: str, days: int = 7) -> str:
    """Query long-term entity history from InfluxDB."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado. Agregá INFLUXDB_TOKEN en .env"})

    try:
        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{days}d)"
            f' |> filter(fn: (r) => r["entity_id"] == "{entity_id}")'
            ' |> sort(columns: ["_time"])'
            " |> limit(n: 500)"
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    "time": record.get_time().isoformat(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                })

        if not records:
            return json.dumps({"message": f"Sin datos para {entity_id} en los últimos {days} días"})

        if len(records) > 100:
            return json.dumps({
                "total_records": len(records),
                "first_10": records[:10],
                "last_10": records[-10:],
                "note": f"Mostrando primeros y últimos 10 de {len(records)} registros",
            }, ensure_ascii=False)

        return json.dumps(records, ensure_ascii=False)

    except Exception as e:
        logger.error(f"InfluxDB query_entity_history error: {e}")
        return json.dumps({"error": str(e)})


async def tool_query_entity_stats(entity_id: str, days: int = 30) -> str:
    """Get behavioral statistics for an entity: state distribution, hourly/daily patterns."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado. Agregá INFLUXDB_TOKEN en .env"})

    try:
        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{days}d)"
            f' |> filter(fn: (r) => r["entity_id"] == "{entity_id}")'
            ' |> sort(columns: ["_time"])'
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        records = []
        for table in tables:
            for record in table.records:
                records.append({
                    "time": record.get_time(),
                    "value": record.get_value(),
                    "field": record.get_field(),
                })

        if not records:
            return json.dumps({"message": f"Sin datos para {entity_id} en los últimos {days} días"})

        # State change counts
        state_counts = Counter(str(r["value"]) for r in records)

        # Hourly distribution (hour of day → number of state changes)
        hourly_activity = Counter(r["time"].hour for r in records)
        hourly_sorted = {f"{h:02d}:00": hourly_activity[h] for h in range(24) if h in hourly_activity}

        # Day of week distribution
        day_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        daily_activity = Counter(day_names[r["time"].weekday()] for r in records)

        # State duration calculation (how long in each state)
        state_durations = defaultdict(float)
        for i in range(len(records) - 1):
            state = str(records[i]["value"])
            duration_h = (records[i + 1]["time"] - records[i]["time"]).total_seconds() / 3600
            if duration_h < 720:  # ignore gaps > 30 days
                state_durations[state] += duration_h

        total_hours = sum(state_durations.values()) or 1
        state_pct = {s: round(h / total_hours * 100, 1) for s, h in state_durations.items()}

        stats = {
            "entity_id": entity_id,
            "period_days": days,
            "total_state_changes": len(records),
            "state_time_pct": state_pct,
            "state_change_counts": dict(state_counts),
            "hourly_pattern": hourly_sorted,
            "daily_pattern": dict(daily_activity),
        }

        # Numeric stats if applicable
        numeric_vals = [r["value"] for r in records if isinstance(r["value"], (int, float))]
        if numeric_vals:
            stats["numeric"] = {
                "min": round(min(numeric_vals), 2),
                "max": round(max(numeric_vals), 2),
                "avg": round(sum(numeric_vals) / len(numeric_vals), 2),
            }

        return json.dumps(stats, ensure_ascii=False, default=str)

    except Exception as e:
        logger.error(f"InfluxDB query_entity_stats error: {e}")
        return json.dumps({"error": str(e)})


async def tool_query_home_activity(hours: int = 24, domain: str = None) -> str:
    """Overview of home activity: most active entities, state changes by domain."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado. Agregá INFLUXDB_TOKEN en .env"})

    try:
        domain_filter = ""
        if domain:
            domain_filter = f' |> filter(fn: (r) => r["domain"] == "{domain}")'

        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{hours}h)"
            f"{domain_filter}"
            ' |> group(columns: ["entity_id", "domain"])'
            " |> count()"
            " |> group()"
            ' |> sort(columns: ["_value"], desc: true)'
            " |> limit(n: 50)"
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        entities = []
        domain_totals = defaultdict(int)
        for table in tables:
            for record in table.records:
                eid = record.values.get("entity_id", "")
                dom = record.values.get("domain", "")
                count = record.get_value()
                entities.append({"entity_id": eid, "domain": dom, "changes": count})
                domain_totals[dom] += count

        return json.dumps({
            "period_hours": hours,
            "domain_filter": domain,
            "changes_by_domain": dict(sorted(domain_totals.items(), key=lambda x: -x[1])),
            "most_active_entities": entities[:30],
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"InfluxDB query_home_activity error: {e}")
        return json.dumps({"error": str(e)})


# ─── HA Event Bus Listener ────────────────────────────────────────────────────


async def ha_event_listener():
    """Connect to HA WebSocket API and listen for state_changed events.
    Writes context-enriched events to InfluxDB for who-triggered-what analytics."""
    reconnect_delay = 5

    while True:
        try:
            async with websockets.connect(HA_WS_URL) as ws:
                # Phase 1: Authenticate
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_required":
                    logger.error(f"HA WS unexpected message: {msg.get('type')}")
                    await asyncio.sleep(reconnect_delay)
                    continue

                await ws.send(json.dumps({"type": "auth", "access_token": HA_TOKEN}))
                msg = json.loads(await ws.recv())
                if msg.get("type") != "auth_ok":
                    logger.error(f"HA WS auth failed: {msg}")
                    await asyncio.sleep(30)
                    continue

                logger.info("HA Event Bus: connected and authenticated")
                reconnect_delay = 5  # reset on successful connection

                # Phase 2: Subscribe to state_changed events
                await ws.send(json.dumps({
                    "id": 1,
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                }))

                # Phase 3: Process events
                async for raw_msg in ws:
                    try:
                        msg = json.loads(raw_msg)
                        if msg.get("type") != "event":
                            continue

                        event = msg["event"]
                        data = event["data"]
                        entity_id = data["entity_id"]
                        domain = entity_id.split(".")[0]

                        if domain not in HA_EVENT_TRACKED_DOMAINS:
                            continue

                        # Pattern filtering for noisy domains
                        if domain == "binary_sensor":
                            entity_lower = entity_id.lower()
                            if not any(p in entity_lower for p in HA_BINARY_SENSOR_PATTERNS):
                                continue
                        elif domain == "sensor":
                            entity_lower = entity_id.lower()
                            if not any(p in entity_lower for p in HA_SENSOR_PATTERNS):
                                continue
                        elif domain == "device_tracker":
                            # Only pattern-filter personal devices; infra devices
                            # rarely change state so they self-filter naturally
                            entity_lower = entity_id.lower()
                            if any(p in entity_lower for p in HA_DEVICE_TRACKER_PATTERNS):
                                pass  # always track personal devices
                            # Also let through any actual state change (home↔not_home)
                            # for infra devices — handled below

                        new_state = data.get("new_state") or {}
                        old_state = data.get("old_state") or {}

                        # Check if main state changed
                        old_val = old_state.get("state", "")
                        new_val = new_state.get("state", "")
                        state_changed = old_val != new_val

                        if not state_changed:
                            # For climate/cover, also log when key attributes
                            # change (temperature, position) even if state stays same
                            if domain == "climate":
                                old_temp = (old_state.get("attributes") or {}).get("current_temperature")
                                new_temp = (new_state.get("attributes") or {}).get("current_temperature")
                                if old_temp == new_temp:
                                    continue
                            elif domain == "cover":
                                old_pos = (old_state.get("attributes") or {}).get("current_position")
                                new_pos = (new_state.get("attributes") or {}).get("current_position")
                                if old_pos == new_pos:
                                    continue
                            else:
                                continue

                        # Rate limit sensor & attribute-only changes (max 1 write per 5 min per entity)
                        if domain == "sensor" or not state_changed:
                            now_ts = monotime()
                            if now_ts - _event_rate_limit.get(entity_id, 0) < HA_SENSOR_MIN_INTERVAL:
                                continue
                            _event_rate_limit[entity_id] = now_ts

                        context = event.get("context", {})
                        user_id = context.get("user_id") or ""
                        parent_id = context.get("parent_id") or ""
                        source = "user" if user_id else ("automation" if parent_id else "unknown")

                        point = (
                            Point("ha_context")
                            .tag("entity_id", entity_id)
                            .tag("domain", domain)
                            .tag("source", source)
                            .field("old_state", old_val)
                            .field("new_state", new_val)
                            .field("user_id", user_id)
                            .field("parent_id", parent_id)
                        )

                        # Domain-specific attribute enrichment
                        attrs = new_state.get("attributes", {})

                        if domain == "climate":
                            for k in ("current_temperature", "temperature", "hvac_action"):
                                v = attrs.get(k)
                                if v is not None:
                                    point.field(k, float(v) if isinstance(v, (int, float)) else str(v))
                        elif domain == "lock":
                            bl = attrs.get("battery_level")
                            if bl is not None:
                                point.field("battery", float(bl))
                        elif domain == "vacuum":
                            bl = attrs.get("battery_level")
                            if bl is not None:
                                point.field("battery", float(bl))
                            vstatus = attrs.get("status")
                            if vstatus:
                                point.field("vacuum_status", str(vstatus))
                        elif domain == "sensor":
                            unit = attrs.get("unit_of_measurement")
                            if unit:
                                point.tag("unit", unit)
                        elif domain == "cover":
                            pos = attrs.get("current_position")
                            if pos is not None:
                                point.field("position", float(pos))
                            dc = attrs.get("device_class", "")
                            if dc:
                                point.tag("cover_type", dc)
                        elif domain == "device_tracker":
                            src = attrs.get("source_type", "")
                            if src:
                                point.tag("source_type", src)
                            # For GPS trackers, capture location
                            if src == "gps":
                                lat = attrs.get("latitude")
                                lon = attrs.get("longitude")
                                gps_acc = attrs.get("gps_accuracy")
                                if lat is not None:
                                    point.field("latitude", float(lat))
                                if lon is not None:
                                    point.field("longitude", float(lon))
                                if gps_acc is not None:
                                    point.field("gps_accuracy", float(gps_acc))
                        elif domain == "person":
                            src = attrs.get("source", "")
                            if src:
                                point.tag("source_entity", src)

                        points = [point]

                        # For media_player playing, capture what's playing
                        if domain == "media_player" and new_val == "playing":
                            media_point = (
                                Point("media_history")
                                .tag("entity_id", entity_id)
                                .tag("source", source)
                                .field("title", attrs.get("media_title", ""))
                                .field("artist", attrs.get("media_artist", ""))
                                .field("album", attrs.get("media_album_name", ""))
                                .field("content_type", attrs.get("media_content_type", ""))
                                .field("volume", float(attrs.get("volume_level") or 0))
                            )
                            if user_id:
                                media_point.field("user_id", user_id)
                            points.append(media_point)

                        await write_points_to_influx(*points)

                    except Exception as e:
                        logger.error(f"HA Event Bus process error: {e}")

        except asyncio.CancelledError:
            logger.info("HA Event Bus: shutting down")
            break
        except Exception as e:
            logger.error(f"HA Event Bus error: {e}, reconnecting in {reconnect_delay}s")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)


# ─── Event Bus Analytics Tools ────────────────────────────────────────────────


async def tool_query_who_changed(entity_id: str = None, hours: int = 24) -> str:
    """Query who triggered state changes, with context (user/automation/unknown)."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado"})

    try:
        entity_filter = ""
        if entity_id:
            entity_filter = f' |> filter(fn: (r) => r["entity_id"] == "{entity_id}")'

        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{hours}h)"
            f' |> filter(fn: (r) => r["_measurement"] == "ha_context")'
            f"{entity_filter}"
            ' |> sort(columns: ["_time"], desc: true)'
            " |> limit(n: 200)"
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        events = defaultdict(dict)
        for table in tables:
            for record in table.records:
                key = f"{record.get_time().isoformat()}_{record.values.get('entity_id', '')}"
                events[key]["time"] = record.get_time().isoformat()
                events[key]["entity_id"] = record.values.get("entity_id", "")
                events[key]["domain"] = record.values.get("domain", "")
                events[key]["source"] = record.values.get("source", "")
                events[key][record.get_field()] = record.get_value()

        result = sorted(events.values(), key=lambda x: x["time"], reverse=True)

        if not result:
            return json.dumps({"message": f"Sin datos de contexto en las últimas {hours}h. "
                               "El event bus acumula datos desde que el bot se inicia."})

        source_counts = Counter(e.get("source", "unknown") for e in result)

        return json.dumps({
            "period_hours": hours,
            "entity_filter": entity_id,
            "total_changes": len(result),
            "by_source": dict(source_counts),
            "events": result[:50],
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"InfluxDB query_who_changed error: {e}")
        return json.dumps({"error": str(e)})


async def tool_query_media_history(entity_id: str = None, hours: int = 24) -> str:
    """Query what music/media was played on speakers."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado"})

    try:
        entity_filter = ""
        if entity_id:
            entity_filter = f' |> filter(fn: (r) => r["entity_id"] == "{entity_id}")'

        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{hours}h)"
            f' |> filter(fn: (r) => r["_measurement"] == "media_history")'
            f"{entity_filter}"
            ' |> sort(columns: ["_time"], desc: true)'
            " |> limit(n: 200)"
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        events = defaultdict(dict)
        for table in tables:
            for record in table.records:
                key = f"{record.get_time().isoformat()}_{record.values.get('entity_id', '')}"
                events[key]["time"] = record.get_time().isoformat()
                events[key]["entity_id"] = record.values.get("entity_id", "")
                events[key]["source"] = record.values.get("source", "")
                events[key][record.get_field()] = record.get_value()

        result = sorted(events.values(), key=lambda x: x["time"], reverse=True)

        if not result:
            return json.dumps({"message": f"Sin historial de media en las últimas {hours}h. "
                               "Se registra cada vez que un speaker empieza a reproducir."})

        return json.dumps({
            "period_hours": hours,
            "entity_filter": entity_id,
            "total_plays": len(result),
            "history": result[:50],
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"InfluxDB query_media_history error: {e}")
        return json.dumps({"error": str(e)})


async def tool_query_bot_usage(days: int = 7) -> str:
    """Query bot usage statistics."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado"})

    try:
        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{days}d)"
            f' |> filter(fn: (r) => r["_measurement"] == "bot_interaction")'
            ' |> sort(columns: ["_time"], desc: true)'
            " |> limit(n: 500)"
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        events = defaultdict(dict)
        for table in tables:
            for record in table.records:
                key = f"{record.get_time().isoformat()}_{record.values.get('user_name', '')}"
                events[key]["time"] = record.get_time().isoformat()
                events[key]["user_name"] = record.values.get("user_name", "")
                events[key]["msg_type"] = record.values.get("msg_type", "")
                events[key][record.get_field()] = record.get_value()

        interactions = sorted(events.values(), key=lambda x: x["time"], reverse=True)

        if not interactions:
            return json.dumps({"message": f"Sin interacciones en los últimos {days} días."})

        by_user = Counter(e.get("user_name", "?") for e in interactions)
        by_type = Counter(e.get("msg_type", "?") for e in interactions)

        all_tools = []
        for e in interactions:
            tools = e.get("tools_used", "none")
            if tools and tools != "none":
                all_tools.extend(tools.split(","))
        top_tools = Counter(all_tools).most_common(10)

        times = [e.get("response_time_ms", 0) for e in interactions
                 if isinstance(e.get("response_time_ms"), (int, float)) and e.get("response_time_ms", 0) > 0]
        avg_time = round(sum(times) / len(times)) if times else 0

        return json.dumps({
            "period_days": days,
            "total_interactions": len(interactions),
            "by_user": dict(by_user),
            "by_type": dict(by_type),
            "top_tools": dict(top_tools),
            "avg_response_time_ms": avg_time,
            "recent": interactions[:20],
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"InfluxDB query_bot_usage error: {e}")
        return json.dumps({"error": str(e)})


async def tool_query_weather_history(hours: int = 24) -> str:
    """Query weather history from InfluxDB."""
    if not get_influx_client():
        return json.dumps({"error": "InfluxDB no configurado"})

    try:
        query = (
            f'from(bucket: "{INFLUXDB_BUCKET}")'
            f" |> range(start: -{hours}h)"
            f' |> filter(fn: (r) => r["_measurement"] == "weather")'
            ' |> sort(columns: ["_time"], desc: true)'
            " |> limit(n: 500)"
        )

        query_api = get_influx_client().query_api()
        tables = await query_api.query(query, org=INFLUXDB_ORG)

        events = defaultdict(dict)
        for table in tables:
            for record in table.records:
                key = record.get_time().isoformat()
                events[key]["time"] = key
                events[key][record.get_field()] = record.get_value()

        result = sorted(events.values(), key=lambda x: x["time"], reverse=True)

        if not result:
            return json.dumps({"message": f"Sin datos meteorológicos en las últimas {hours}h. "
                               "Se recolectan cada 15 minutos."})

        # Current conditions (most recent)
        latest = result[0]

        # Compute min/max/avg for numeric fields
        numeric_fields = ["temperature", "apparent_temperature", "humidity",
                          "precipitation", "wind_speed", "uv_index", "pressure"]
        stats = {}
        for field in numeric_fields:
            values = [e.get(field) for e in result
                      if isinstance(e.get(field), (int, float))]
            if values:
                stats[field] = {
                    "min": round(min(values), 1),
                    "max": round(max(values), 1),
                    "avg": round(sum(values) / len(values), 1),
                }

        # Total precipitation
        precip_values = [e.get("precipitation", 0) for e in result
                         if isinstance(e.get("precipitation"), (int, float))]
        total_precip = round(sum(precip_values) * (WEATHER_POLL_INTERVAL / 3600), 1) if precip_values else 0

        return json.dumps({
            "period_hours": hours,
            "data_points": len(result),
            "current": latest,
            "stats": stats,
            "total_precipitation_mm_approx": total_precip,
            "samples": result[:10],  # Last 10 readings (~2.5h)
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"InfluxDB query_weather_history error: {e}")
        return json.dumps({"error": str(e)})


# ─── Tool Registry ───────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "list_entities": tool_list_entities,
    "get_entity": tool_get_entity,
    "control_entity": tool_control_entity,
    "call_service": tool_call_service,
    "search_entities": tool_search_entities,
    "get_history": tool_get_history,
    "get_ha_config": tool_get_ha_config,
    "query_entity_history": tool_query_entity_history,
    "query_entity_stats": tool_query_entity_stats,
    "query_home_activity": tool_query_home_activity,
    "query_who_changed": tool_query_who_changed,
    "query_media_history": tool_query_media_history,
    "query_bot_usage": tool_query_bot_usage,
    "query_weather_history": tool_query_weather_history,
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
    {
        "name": "query_entity_history",
        "description": "Query long-term entity history from InfluxDB (days/weeks/months, not limited to HA's 10-day retention). Use this for trend analysis over extended periods.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity to query history for"},
                "days": {"type": "integer", "description": "Days of history to query (default 7, max 365)"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "query_entity_stats",
        "description": "Get behavioral statistics for an entity from InfluxDB: percentage of time in each state, hourly patterns (when it's typically active), day-of-week patterns, and numeric stats. Use this to understand habits and detect anomalies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity to analyze"},
                "days": {"type": "integer", "description": "Days of data to analyze (default 30, max 365)"},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "query_home_activity",
        "description": "Overview of home activity from InfluxDB: most active entities, state changes grouped by domain. Use this to understand overall home behavior patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "description": "Hours of activity to analyze (default 24)"},
                "domain": {"type": "string", "description": "Optional domain filter (light, lock, climate, etc.)"},
            },
        },
    },
    {
        "name": "query_who_changed",
        "description": "Query who or what triggered state changes on entities (user, automation, or unknown). Shows context for each change from the HA event bus. Use for 'who turned on the lights?' or 'what changed recently?'",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Filter by entity (optional, shows all tracked domains if omitted)"},
                "hours": {"type": "integer", "description": "Hours of history (default 24)"},
            },
        },
    },
    {
        "name": "query_media_history",
        "description": "Query what music/media was played on speakers. Shows title, artist, album, speaker, and who started it. Use for 'what was playing earlier?' or 'what did we listen to today?'",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Filter by speaker entity (optional)"},
                "hours": {"type": "integer", "description": "Hours of history (default 24)"},
            },
        },
    },
    {
        "name": "query_bot_usage",
        "description": "Query bot usage statistics: who uses it, message types, most used tools, and average response time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days of history (default 7)"},
            },
        },
    },
    {
        "name": "query_weather_history",
        "description": "Query weather history for Pilar, Buenos Aires. Shows temperature, humidity, wind, precipitation, UV, pressure, cloud cover, and conditions. Use for correlating home behavior with weather, or answering 'how was the weather today/yesterday?'",
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "description": "Hours of history (default 24)"},
            },
        },
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

    # Build per-user system prompt with their name and current local time
    now_local = datetime.now(LOCAL_TZ)
    time_ctx = f"Fecha y hora actual: {now_local.strftime('%A %d/%m/%Y %H:%M')} (Argentina)."
    user_name = USER_NAMES.get(user_id)
    if user_name:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{time_ctx}\nEstás hablando con {user_name}."
    else:
        system_prompt = f"{SYSTEM_PROMPT}\n\n{time_ctx}"

    try:
        response = await asyncio.wait_for(
            claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            ),
            timeout=120,
        )

        max_iterations = 10
        iteration = 0
        tools_used = []

        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1
            assistant_content = response.content
            add_message(user_id, "assistant", assistant_content)

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
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
                    system=system_prompt,
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
        _last_tools_used[user_id] = tools_used
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
        if update.effective_user.id not in TELEGRAM_USER_IDS:
            await update.message.reply_text("⛔ No autorizado.")
            logger.warning(f"Unauthorized access from user {update.effective_user.id}")
            return
        name = USER_NAMES.get(update.effective_user.id, str(update.effective_user.id))
        logger.info(f"[{name}] /{func.__name__.removeprefix('cmd_')}")
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
        "/analytics - Análisis de actividad del hogar\n"
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
async def cmd_analytics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not get_influx_client():
        await update.message.reply_text("⚠️ InfluxDB no está configurado. Agregá INFLUXDB_TOKEN en .env")
        return
    response = await run_with_typing(update, chat_with_claude(
        update.effective_user.id,
        "Usá las herramientas de analytics (query_home_activity, query_entity_stats) para darme un resumen "
        "de la actividad de la casa en las últimas 24 horas. Incluí: dominios más activos, entidades con más "
        "cambios de estado, y cualquier patrón interesante que notes. Sé conciso y visual con emojis.",
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

    name = USER_NAMES.get(user_id, str(user_id))
    logger.info(f"[{name}] {update.message.text[:80]}")
    start = monotime()
    response = await run_with_typing(update, chat_with_claude(user_id, update.message.text))
    elapsed_ms = int((monotime() - start) * 1000)
    asyncio.create_task(log_bot_interaction(user_id, "text", update.message.text, elapsed_ms))
    await send_long_message(update, response)


@authorized
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = USER_NAMES.get(update.effective_user.id, str(update.effective_user.id))
    logger.info(f"[{name}] voice message")
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

        start = monotime()
        response = await run_with_typing(update, chat_with_claude(update.effective_user.id, transcribed_text))
        elapsed_ms = int((monotime() - start) * 1000)
        asyncio.create_task(log_bot_interaction(update.effective_user.id, "voice", transcribed_text, elapsed_ms))
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

        # Send WaterGuru report only to admin
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_ADMIN_ID,
                text=report,
                parse_mode="Markdown",
            )
        except Exception:
            plain = report.replace("*", "").replace("_", "")
            await telegram_bot.send_message(
                chat_id=TELEGRAM_ADMIN_ID,
                text=plain,
            )
        logger.info("WaterGuru report sent")

    except Exception as e:
        logger.error(f"WaterGuru report error: {e}")
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_ADMIN_ID,
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


# ─── Weather Data Collection ─────────────────────────────────────────────────

OPENMETEO_URL = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
    f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
    f"precipitation,weather_code,cloud_cover,pressure_msl,"
    f"wind_speed_10m,wind_direction_10m,uv_index"
    f"&daily=sunrise,sunset&timezone=America/Argentina/Buenos_Aires&forecast_days=1"
)

WEATHER_CODES = {
    0: "Despejado", 1: "Mayormente despejado", 2: "Parcialmente nublado",
    3: "Nublado", 45: "Niebla", 48: "Niebla con escarcha",
    51: "Llovizna leve", 53: "Llovizna moderada", 55: "Llovizna intensa",
    61: "Lluvia leve", 63: "Lluvia moderada", 65: "Lluvia intensa",
    71: "Nieve leve", 73: "Nieve moderada", 75: "Nieve intensa",
    80: "Chaparrón leve", 81: "Chaparrón moderado", 82: "Chaparrón intenso",
    95: "Tormenta", 96: "Tormenta con granizo leve", 99: "Tormenta con granizo",
}


async def weather_poll(context):
    """Poll Open-Meteo every 15 min and write weather data to InfluxDB."""
    if not get_influx_client():
        return

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(OPENMETEO_URL)
            if resp.status_code != 200:
                logger.error(f"Weather poll HTTP {resp.status_code}")
                return
            data = resp.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        weather_code = current.get("weather_code", 0)
        condition = WEATHER_CODES.get(weather_code, f"Código {weather_code}")

        point = (
            Point("weather")
            .tag("location", "pilar")
            .field("temperature", float(current.get("temperature_2m", 0)))
            .field("apparent_temperature", float(current.get("apparent_temperature", 0)))
            .field("humidity", float(current.get("relative_humidity_2m", 0)))
            .field("precipitation", float(current.get("precipitation", 0)))
            .field("cloud_cover", float(current.get("cloud_cover", 0)))
            .field("pressure", float(current.get("pressure_msl", 0)))
            .field("wind_speed", float(current.get("wind_speed_10m", 0)))
            .field("wind_direction", float(current.get("wind_direction_10m", 0)))
            .field("uv_index", float(current.get("uv_index", 0)))
            .field("weather_code", weather_code)
            .field("condition", condition)
        )

        # Add sunrise/sunset if available
        if daily.get("sunrise"):
            point.field("sunrise", daily["sunrise"][0])
        if daily.get("sunset"):
            point.field("sunset", daily["sunset"][0])

        await write_points_to_influx(point)
        logger.debug(f"Weather: {current.get('temperature_2m')}°C, {condition}")

    except Exception as e:
        logger.error(f"Weather poll error: {e}")


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
    app.add_handler(CommandHandler("analytics", cmd_analytics))
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
                chat_id=TELEGRAM_ADMIN_ID,
                text=text,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.error(f"Startup notification error: {e}")

    # Start HA Event Bus listener for context tracking
    async def start_event_listener(ctx: ContextTypes.DEFAULT_TYPE):
        if get_influx_client():
            asyncio.create_task(ha_event_listener())
            logger.info("HA Event Bus listener started")
        else:
            logger.info("HA Event Bus listener skipped (no InfluxDB)")

    app.job_queue.run_repeating(heartbeat, interval=30, first=10)
    app.job_queue.run_repeating(waterguru_poll, interval=WATERGURU_POLL_INTERVAL, first=60)
    app.job_queue.run_repeating(weather_poll, interval=WEATHER_POLL_INTERVAL, first=15)
    app.job_queue.run_once(startup_notify, when=2)
    app.job_queue.run_once(start_event_listener, when=5)

    # Graceful shutdown handler
    def graceful_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
