"""
WissensBot – Discord Bot mit Kanal-Indexierung und KI-gestützter Fragenbeantwortung
Läuft mit einer einzigen JSON-Datenbank (database.json)
APIs: Gemini (primär, mit Modell-Fallback-Kette) -> Groq (finaler Fallback)
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from io import BytesIO

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from flask import Flask
from threading import Thread
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

def clean_env(name: str) -> str | None:
    """Lädt eine Env-Var und entfernt Whitespace sowie versehentlich mitkopierte Anführungszeichen
    (häufigste Ursache für 'ACCESS_TOKEN_TYPE_UNSUPPORTED'-Fehler bei der Gemini API)."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value or None


DISCORD_TOKEN = clean_env("DISCORD_TOKEN")
GEMINI_API_KEY = clean_env("GEMINI_API_KEY")
GROQ_API_KEY = clean_env("GROQ_API_KEY")

OWNER_ID = 1211683189186105434

DB_PATH = "database.json"

BERLIN_TZ = timezone(timedelta(hours=2))  # Sommerzeit; Winterzeit +1 - für reine Timestamps ausreichend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wissensbot")

# Gemini-Modell-Kette: wird der Reihe nach probiert, bevor auf Groq zurückgefallen wird
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "gemma2-9b-it",
]

# Maximale Zeichenanzahl an Kontext, die an die KI geschickt wird (Kostenschutz)
MAX_CONTEXT_CHARS = 500_000
# Zeitabstand zwischen Fortschritts-Updates bei /index-build (Sekunden) - schützt vor Rate-Limits
PROGRESS_UPDATE_INTERVAL_SECONDS = 5

# Cooldown pro Nutzer zwischen zwei Fragen im Q&A-Kanal (Sekunden) - schützt die APIs vor Spam
QUESTION_COOLDOWN_SECONDS = 60

# ---------------------------------------------------------------------------
# ANPASSBAR: Statusanzeige während der Fragenbeantwortung
# ---------------------------------------------------------------------------
# Emoji-Frames für den Spinner im /index-build Fortschritts-Embed. Kann auch mit
# eigenen animierten Server-Emojis befüllt werden, z.B. ["<a:laden1:123>", "<a:laden2:456>"]
SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]

# Die Verarbeiten-Nachricht durchläuft diese drei Phasen (Emoji + Text), jeweils per
# Edit auf dieselbe Nachricht. Emojis können auch eigene animierte Server-Emojis sein,
# im Format <a:name:id>.
STAGE_THINKING = ("🕐", "Nachdenken...")
STAGE_GENERATING = ("✏️", "Antwort wird generiert...")
STAGE_PRESENTING = ("📄", "Präsentiere Antwort...")
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ANPASSBAR: /status Anzeige
# ---------------------------------------------------------------------------
DISCORD_ICON = "💬"
GEMINI_ICON = "✨"
GROQ_ICON = "⚡"
AI_STUDIO_URL = "https://aistudio.google.com/apikey"  # Fallback, falls kein Key gesetzt ist
# ---------------------------------------------------------------------------


EMBED_COLOR = 0x393A41
EMBED_COLOR_ERROR = 0xB33A3A
EMBED_COLOR_SUCCESS = 0x3A7D44

# ---------------------------------------------------------------------------
# Datenbank
# ---------------------------------------------------------------------------

DEFAULT_DB = {
    "indexed_channels": [],
    "qa_channel_id": None,
    "backup_channel_id": None,
    "messages": {}
}


def load_db() -> dict:
    if not os.path.exists(DB_PATH):
        save_db(DEFAULT_DB)
        return json.loads(json.dumps(DEFAULT_DB))
    with open(DB_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            log.error("database.json ist beschädigt, erstelle neue Datenbank.")
            data = json.loads(json.dumps(DEFAULT_DB))
    for key, value in DEFAULT_DB.items():
        if key not in data:
            data[key] = value
    return data


def save_db(data: dict) -> None:
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


db = load_db()

# Cooldown-Tracking pro Nutzer (nicht persistiert, reicht als In-Memory-State)
last_question_time: dict[int, float] = {}

# ---------------------------------------------------------------------------
# Discord Bot Setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!wb-unused!", intents=intents)


def is_owner():
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.user.id != OWNER_ID:
            await interaction.response.send_message(
                embed=error_embed("Keine Berechtigung", "Dieser Befehl ist nur für den Bot-Owner verfügbar."),
                ephemeral=True
            )
            return False
        return True
    return app_commands.check(predicate)


# ---------------------------------------------------------------------------
# Embed Helfer
# ---------------------------------------------------------------------------

def bullet(text: str) -> str:
    return f"> {text}"


def code_block(text: str, limit: int = 1000) -> str:
    """Formatiert Text als kopierbaren Codeblock, gekürzt aufs Discord-Feldlimit (1024 Zeichen)."""
    if len(text) > limit:
        text = text[:limit] + "\n... (gekürzt)"
    return f"```\n{text}\n```"


def base_embed(title: str, description: str = "", color: int = EMBED_COLOR) -> discord.Embed:
    embed = discord.Embed(title=title, description=description, color=color)
    return embed


def error_embed(title: str, description: str) -> discord.Embed:
    return base_embed(f"Fehler – {title}", description, EMBED_COLOR_ERROR)


def success_embed(title: str, description: str) -> discord.Embed:
    return base_embed(title, description, EMBED_COLOR_SUCCESS)


def with_executor(embed: discord.Embed, user: discord.abc.User) -> discord.Embed:
    embed.set_author(name=str(user), icon_url=user.display_avatar.url)
    embed.timestamp = datetime.now(timezone.utc)
    return embed


QA_BOT_DISPLAY_NAME = "InsuranceGuard Help"


def with_bot_branding(embed: discord.Embed) -> discord.Embed:
    """Setzt den Autor auf den Bot-Namen/Logo statt auf den fragenden User."""
    icon_url = bot.user.display_avatar.url if bot.user else None
    embed.set_author(name=QA_BOT_DISPLAY_NAME, icon_url=icon_url)
    embed.timestamp = datetime.now(timezone.utc)
    return embed


def progress_bar_indeterminate(tick: int, length: int = 20) -> str:
    """Wandernder Balken, der zeigt dass der Bot noch aktiv arbeitet (Gesamtzahl bei history() unbekannt)."""
    position = tick % length
    bar = ["░"] * length
    bar[position] = "█"
    if position > 0:
        bar[position - 1] = "▓"
    if position < length - 1:
        bar[position + 1] = "▓"
    return "`[" + "".join(bar) + "]`"


def progress_bar_percent(percent: float, length: int = 20) -> str:
    """Echter, füllender Fortschrittsbalken basierend auf einem bekannten Prozentwert."""
    percent = max(0.0, min(100.0, percent))
    filled = round((percent / 100) * length)
    bar = "█" * filled + "░" * (length - filled)
    return f"`[{bar}]` {percent:.0f}%"


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def latency_bar(ms: float | None, max_ms: float = 1000, length: int = 12) -> str:
    """Stellt eine Latenz in ms als Balken + Zahl dar. None = nicht erreichbar."""
    if ms is None:
        return "`[" + "░" * length + "]` nicht erreichbar"
    filled = min(length, max(1, round((ms / max_ms) * length)))
    bar = "█" * filled + "░" * (length - filled)
    return f"`[{bar}]` {ms:.0f} ms"


# ---------------------------------------------------------------------------
# Nachrichtenspeicherung
# ---------------------------------------------------------------------------

def embed_to_text(embed: discord.Embed) -> str:
    """Extrahiert lesbaren Text aus einem klassischen Embed (Titel, Beschreibung, Felder, Footer, Autor)."""
    parts = []
    if embed.author and embed.author.name:
        parts.append(f"[Embed-Autor] {embed.author.name}")
    if embed.title:
        parts.append(f"[Embed-Titel] {embed.title}")
    if embed.description:
        parts.append(f"[Embed-Beschreibung] {embed.description}")
    for field in embed.fields:
        parts.append(f"[Embed-Feld] {field.name}: {field.value}")
    if embed.footer and embed.footer.text:
        parts.append(f"[Embed-Footer] {embed.footer.text}")
    return "\n".join(parts)


def extract_message_text(message: discord.Message) -> str:
    """Kombiniert normalen Nachrichtentext mit Text aus klassischen Embeds (keine Components V2 Container)."""
    parts = []
    if message.content:
        parts.append(message.content)
    for embed in message.embeds:
        embed_text = embed_to_text(embed)
        if embed_text:
            parts.append(embed_text)
    return "\n".join(parts)


def store_message(channel_id: int, message: discord.Message) -> None:
    key = str(channel_id)
    if key not in db["messages"]:
        db["messages"][key] = []
    db["messages"][key].append({
        "id": message.id,
        "author": str(message.author),
        "content": extract_message_text(message),
        "timestamp": message.created_at.isoformat()
    })


def message_exists(channel_id: int, message_id: int) -> bool:
    key = str(channel_id)
    if key not in db["messages"]:
        return False
    return any(m["id"] == message_id for m in db["messages"][key])


def build_context(guild: discord.Guild) -> str:
    """Baut den kombinierten Textkontext aus allen indizierten Kanälen, gekappt auf MAX_CONTEXT_CHARS."""
    parts = []
    for channel_id in db["indexed_channels"]:
        key = str(channel_id)
        channel = guild.get_channel(channel_id)
        channel_name = channel.name if channel else f"unbekannt-{channel_id}"
        messages = db["messages"].get(key, [])
        if not messages:
            continue
        parts.append(f"\n--- Kanal: #{channel_name} ---")
        for m in messages:
            if m["content"].strip():
                parts.append(f"{m['author']}: {m['content']}")
    full_text = "\n".join(parts)
    if len(full_text) > MAX_CONTEXT_CHARS:
        full_text = full_text[-MAX_CONTEXT_CHARS:]  # neueste Nachrichten bevorzugen
    return full_text


# ---------------------------------------------------------------------------
# KI-Anbindung: Gemini (mit Modell-Fallback-Kette) -> Groq
# ---------------------------------------------------------------------------

QA_SYSTEM_INSTRUCTION = (
    "Du bist ein hilfreicher Assistent für einen Discord-Server. Beantworte die Frage des Nutzers "
    "ausschließlich basierend auf dem bereitgestellten Kontext aus den Server-Nachrichten. "
    "Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich, statt zu spekulieren. "
    "Antworte präzise, auf Deutsch und ohne den Kontext wörtlich zu wiederholen."
)


async def ask_gemini(session: aiohttp.ClientSession, model: str, context: str, question: str) -> tuple[str | None, int | None, str | None]:
    """Gibt (Antworttext_oder_None, HTTP-Status_oder_None, Fehlerdetails_oder_None) zurück."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": QA_SYSTEM_INSTRUCTION}]},
        "contents": [{
            "role": "user",
            "parts": [{"text": f"Kontext:\n{context}\n\nFrage: {question}"}]
        }]
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning(f"Gemini-Modell {model} fehlgeschlagen: HTTP {resp.status} – {body[:500]}")
                return None, resp.status, body[:900]
            data = await resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                detail = f"Keine Kandidaten in der Antwort (evtl. Safety-Block). Rohantwort: {json.dumps(data)[:800]}"
                log.warning(f"Gemini-Modell {model}: {detail}")
                return None, resp.status, detail
            return candidates[0]["content"]["parts"][0]["text"], resp.status, None
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log.warning(f"Gemini-Modell {model} Exception: {detail}")
        return None, None, detail


async def ask_groq(session: aiohttp.ClientSession, model: str, context: str, question: str) -> tuple[str | None, int | None, str | None]:
    """Gibt (Antworttext_oder_None, HTTP-Status_oder_None, Fehlerdetails_oder_None) zurück."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": QA_SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {question}"}
        ]
    }
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                body = await resp.text()
                log.warning(f"Groq-Modell {model} fehlgeschlagen: HTTP {resp.status} – {body[:500]}")
                return None, resp.status, body[:900]
            data = await resp.json()
            return data["choices"][0]["message"]["content"], resp.status, None
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log.warning(f"Groq Exception: {detail}")
        return None, None, detail


async def measure_gemini_latency() -> tuple[float | None, str | None]:
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY ist nicht gesetzt."
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    try:
        start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                body = await resp.text()
                if resp.status != 200:
                    log.warning(f"Gemini-Ping fehlgeschlagen: HTTP {resp.status} – {body[:500]}")
                    return None, f"HTTP {resp.status}: {body[:900]}"
        return (time.monotonic() - start) * 1000, None
    except asyncio.TimeoutError:
        return None, "Timeout nach 15 Sekunden – keine Antwort von Google erhalten."
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log.warning(f"Gemini-Ping Exception: {detail}")
        return None, detail


async def measure_groq_latency() -> tuple[float | None, str | None]:
    if not GROQ_API_KEY:
        return None, "GROQ_API_KEY ist nicht gesetzt."
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    try:
        start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                body = await resp.text()
                if resp.status != 200:
                    log.warning(f"Groq-Ping fehlgeschlagen: HTTP {resp.status} – {body[:500]}")
                    return None, f"HTTP {resp.status}: {body[:900]}"
        return (time.monotonic() - start) * 1000, None
    except asyncio.TimeoutError:
        return None, "Timeout nach 15 Sekunden – keine Antwort von Groq erhalten."
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log.warning(f"Groq-Ping Exception: {detail}")
        return None, detail


async def answer_question(context: str, question: str) -> tuple[str, str, str | None]:
    """Gibt (Antwort, verwendetes_Modell, Fehlerdetails_oder_None) zurück. Probiert Gemini-Modelle
    der Reihe nach, dann Groq. Bricht die Gemini-Kette bei einem Auth-Fehler (401/403) sofort ab,
    da weitere Modelle mit demselben ungültigen Key ohnehin scheitern würden."""
    AUTH_ERROR_CODES = {401, 403}
    error_log: list[str] = []
    async with aiohttp.ClientSession() as session:
        if not GEMINI_API_KEY:
            error_log.append("Gemini: GEMINI_API_KEY ist nicht gesetzt.")
        else:
            for model in GEMINI_MODELS:
                result, status, detail = await ask_gemini(session, model, context, question)
                if result:
                    return result, f"Gemini ({model})", None
                error_log.append(f"Gemini ({model}, HTTP {status}):\n{detail}")
                if status in AUTH_ERROR_CODES:
                    log.warning(f"Gemini-Auth-Fehler (HTTP {status}) – wechsle direkt zu Groq, überspringe restliche Gemini-Modelle.")
                    break

        if not GROQ_API_KEY:
            error_log.append("Groq: GROQ_API_KEY ist nicht gesetzt.")
        else:
            for model in GROQ_MODELS:
                result, status, detail = await ask_groq(session, model, context, question)
                if result:
                    return result, f"Groq ({model})", None
                error_log.append(f"Groq ({model}, HTTP {status}):\n{detail}")
                if status in AUTH_ERROR_CODES:
                    log.warning(f"Groq-Auth-Fehler (HTTP {status}) – überspringe restliche Groq-Modelle.")
                    break

    full_error = "\n\n".join(error_log) if error_log else "Keine weiteren Details verfügbar."
    return (
        "Ich konnte keine Antwort generieren – alle KI-Anbieter waren nicht erreichbar. Versuch es später erneut.",
        "keiner (Fehler)",
        full_error
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    log.info(f"Eingeloggt als {bot.user} (ID: {bot.user.id})")
    try:
        synced = await bot.tree.sync()
        log.info(f"{len(synced)} Slash-Befehle synchronisiert.")
    except Exception as e:
        log.error(f"Sync fehlgeschlagen: {e}")


@bot.event
async def on_message(message: discord.Message):
    if bot.user and message.author.id == bot.user.id:
        return  # eigene Nachrichten (z.B. Q&A-Antworten) nie indizieren

    # Live-Indexierung – bewusst auch für andere Bots (z.B. InsuranceGuard-Embeds mit wichtigen Infos)
    if message.channel.id in db["indexed_channels"]:
        store_message(message.channel.id, message)
        save_db(db)

    # Q&A-Kanal – nur echte Nutzer stellen Fragen, keine Bots
    if db["qa_channel_id"] and message.channel.id == db["qa_channel_id"] and not message.author.bot:
        if message.content.strip():
            await handle_question(message)

    await bot.process_commands(message)


async def log_error_to_backup(guild: discord.Guild, title: str, detail: str):
    """Schickt Fehlerdetails zusätzlich in den Backup-Kanal, falls einer gesetzt ist."""
    if not db["backup_channel_id"]:
        return
    backup_channel = guild.get_channel(db["backup_channel_id"])
    if not backup_channel:
        return
    embed = error_embed(title, code_block(detail, limit=3800))
    embed.timestamp = datetime.now(timezone.utc)
    try:
        await backup_channel.send(embed=embed)
    except discord.HTTPException:
        pass


async def handle_question(message: discord.Message):
    now = time.monotonic()
    last = last_question_time.get(message.author.id, 0.0)
    remaining = QUESTION_COOLDOWN_SECONDS - (now - last)
    if remaining > 0:
        await message.reply(
            embed=error_embed("Bitte kurz warten", bullet(f"Noch {remaining:.0f} Sekunden, bevor du die nächste Frage stellen kannst.")),
            delete_after=10
        )
        return
    last_question_time[message.author.id] = now

    def stage_text(stage: tuple[str, str]) -> str:
        emoji, text = stage
        return f"{emoji} {text}"

    processing_message = await message.channel.send(stage_text(STAGE_THINKING))

    async def safe_delete():
        try:
            await processing_message.delete()
        except discord.HTTPException:
            pass

    async def safe_edit(stage: tuple[str, str]):
        try:
            await processing_message.edit(content=stage_text(stage))
        except discord.HTTPException:
            pass

    try:
        context = build_context(message.guild)
        if not context.strip():
            await safe_delete()
            await message.reply(embed=error_embed(
                "Kein Kontext vorhanden",
                "Es sind noch keine Kanäle indiziert. Nutze `/index-add`, um Wissensquellen festzulegen."
            ))
            return

        await safe_edit(STAGE_GENERATING)
        answer, used_model, error_detail = await answer_question(context, message.content)

        await safe_edit(STAGE_PRESENTING)
        await asyncio.sleep(0.8)  # kurz sichtbar, bevor die finale Antwort ersetzt
        await safe_delete()

        if error_detail:
            embed = error_embed("Keine der KIs konnte antworten", answer)
            embed.add_field(name="Fehlerdetails", value=code_block(error_detail), inline=False)
            embed = with_bot_branding(embed)
            await message.reply(embed=embed)
            await log_error_to_backup(message.guild, f"Fehler bei Frage von {message.author}", error_detail)
            return

        embed = base_embed("Antwort", answer)
        embed = with_bot_branding(embed)
        await message.reply(embed=embed)
    except Exception as e:
        log.error(f"Fehler bei der Fragenbeantwortung: {e}")
        await safe_delete()
        await message.reply(embed=error_embed("Unerwarteter Fehler", "Bei der Bearbeitung deiner Frage ist etwas schiefgelaufen."))
        await log_error_to_backup(message.guild, f"Unerwarteter Fehler bei Frage von {message.author}", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Slash-Befehle: Indexierung
# ---------------------------------------------------------------------------

@bot.tree.command(name="index-add", description="Fügt einen Kanal als Wissensquelle hinzu (live indiziert)")
@is_owner()
@app_commands.describe(channel="Der Kanal, der indiziert werden soll")
async def index_add(interaction: discord.Interaction, channel: discord.TextChannel):
    if channel.id in db["indexed_channels"]:
        await interaction.response.send_message(
            embed=error_embed("Bereits indiziert", f"{channel.mention} ist bereits eine Wissensquelle."),
            ephemeral=True
        )
        return
    db["indexed_channels"].append(channel.id)
    save_db(db)
    embed = success_embed("Kanal hinzugefügt", bullet(f"{channel.mention} wird ab jetzt live indiziert."))
    await interaction.response.send_message(embed=with_executor(embed, interaction.user), ephemeral=True)


@bot.tree.command(name="index-remove", description="Entfernt einen Kanal aus den Wissensquellen")
@is_owner()
@app_commands.describe(channel="Der Kanal, der entfernt werden soll")
async def index_remove(interaction: discord.Interaction, channel: discord.TextChannel):
    if channel.id not in db["indexed_channels"]:
        await interaction.response.send_message(
            embed=error_embed("Nicht indiziert", f"{channel.mention} ist keine Wissensquelle."),
            ephemeral=True
        )
        return
    db["indexed_channels"].remove(channel.id)
    save_db(db)
    embed = success_embed("Kanal entfernt", bullet(f"{channel.mention} ist keine Wissensquelle mehr."))
    await interaction.response.send_message(embed=with_executor(embed, interaction.user), ephemeral=True)


@bot.tree.command(name="index-list", description="Zeigt alle indizierten Kanäle mit Nachrichtenanzahl")
@is_owner()
async def index_list(interaction: discord.Interaction):
    if not db["indexed_channels"]:
        await interaction.response.send_message(
            embed=base_embed("Wissensquellen", "Es sind aktuell keine Kanäle indiziert."),
            ephemeral=True
        )
        return
    lines = []
    for channel_id in db["indexed_channels"]:
        channel = interaction.guild.get_channel(channel_id)
        name = channel.mention if channel else f"unbekannt ({channel_id})"
        count = len(db["messages"].get(str(channel_id), []))
        lines.append(bullet(f"{name} – {count} Nachrichten"))
    embed = base_embed("Wissensquellen", "\n".join(lines))
    await interaction.response.send_message(embed=with_executor(embed, interaction.user), ephemeral=True)


@bot.tree.command(name="index-build", description="Lädt die komplette Historie eines indizierten Kanals nach")
@is_owner()
@app_commands.describe(channel="Kanal, dessen Historie nachgeladen werden soll (leer = alle indizierten Kanäle)")
async def index_build(interaction: discord.Interaction, channel: discord.TextChannel | None = None):
    targets = [channel.id] if channel else db["indexed_channels"]
    targets = [c for c in targets if c in db["indexed_channels"]]

    if not targets:
        await interaction.response.send_message(
            embed=error_embed("Keine Zielkanäle", "Der Kanal ist nicht indiziert oder es gibt keine Wissensquellen."),
            ephemeral=True
        )
        return

    await interaction.response.send_message(embed=base_embed("Indexierung gestartet", "Zähle Nachrichten..."), ephemeral=True)
    progress_message = await interaction.original_response()

    start_time = time.monotonic()
    last_update = 0.0
    update_count = 0

    # Phase 1: Zählen, damit die Fortschrittsanzeige in Phase 2 einen echten Prozentwert hat.
    # Kostet etwa genauso viel Zeit wie das Verarbeiten selbst, dafür läuft der Balken danach real durch.
    channel_totals: dict[int, int] = {}
    total_count = 0
    for idx, channel_id in enumerate(targets, start=1):
        target_channel = interaction.guild.get_channel(channel_id)
        if not target_channel:
            continue
        count = 0
        async for _ in target_channel.history(limit=None):
            count += 1
            now = time.monotonic()
            if now - last_update >= PROGRESS_UPDATE_INTERVAL_SECONDS:
                last_update = now
                update_count += 1
                spinner = SPINNER_FRAMES[update_count % len(SPINNER_FRAMES)]
                embed = base_embed(
                    f"{spinner} Zähle Nachrichten...",
                    bullet(f"Kanal {idx}/{len(targets)}: {target_channel.mention}") + "\n" +
                    bullet(f"{count} Nachrichten gefunden") + "\n" +
                    bullet(f"Laufzeit: {format_duration(now - start_time)}") + "\n\n" +
                    progress_bar_indeterminate(update_count)
                )
                try:
                    await progress_message.edit(embed=embed)
                except discord.HTTPException:
                    pass
        channel_totals[channel_id] = count
        total_count += count

    # Phase 2: Verarbeiten mit echtem Prozent-Fortschritt
    processed_total = 0
    total_new = 0
    last_update = 0.0
    for idx, channel_id in enumerate(targets, start=1):
        target_channel = interaction.guild.get_channel(channel_id)
        if not target_channel:
            continue
        new_in_channel = 0
        async for msg in target_channel.history(limit=None, oldest_first=True):
            processed_total += 1
            is_own_message = bot.user and msg.author.id == bot.user.id
            if not is_own_message and not message_exists(channel_id, msg.id):
                store_message(channel_id, msg)
                new_in_channel += 1

            now = time.monotonic()
            if now - last_update >= PROGRESS_UPDATE_INTERVAL_SECONDS:
                last_update = now
                elapsed = now - start_time
                rate = processed_total / elapsed if elapsed > 0 else 0
                percent = (processed_total / total_count * 100) if total_count > 0 else 100
                remaining = (total_count - processed_total) / rate if rate > 0 else 0
                embed = base_embed(
                    "Indexierung läuft",
                    bullet(f"Kanal {idx}/{len(targets)}: {target_channel.mention}") + "\n" +
                    bullet(f"{processed_total}/{total_count} Nachrichten verarbeitet, {new_in_channel} neu in diesem Kanal") + "\n" +
                    bullet(f"Tempo: {rate:.1f} Nachrichten/Sek · Verbleibend: ~{format_duration(remaining)}") + "\n\n" +
                    progress_bar_percent(percent)
                )
                try:
                    await progress_message.edit(embed=embed)
                except discord.HTTPException:
                    pass
        save_db(db)
        total_new += new_in_channel

    total_elapsed = time.monotonic() - start_time
    final_embed = success_embed(
        "Indexierung abgeschlossen",
        bullet(f"{len(targets)} Kanal/Kanäle verarbeitet") + "\n" +
        bullet(f"{total_new} neue Nachrichten gespeichert") + "\n" +
        bullet(f"Gesamtdauer: {format_duration(total_elapsed)}")
    )
    await progress_message.edit(embed=with_executor(final_embed, interaction.user))


@bot.tree.command(name="set-qa-channel", description="Legt fest, in welchem Kanal Fragen gestellt werden können")
@is_owner()
@app_commands.describe(channel="Der Kanal für Fragen")
async def set_qa_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    db["qa_channel_id"] = channel.id
    save_db(db)
    embed = success_embed("Q&A-Kanal gesetzt", bullet(f"Fragen können jetzt in {channel.mention} gestellt werden."))
    await interaction.response.send_message(embed=with_executor(embed, interaction.user), ephemeral=True)


# ---------------------------------------------------------------------------
# Slash-Befehle: Backup / Reload
# ---------------------------------------------------------------------------

@bot.tree.command(name="set-backup-channel", description="Legt den Kanal für automatische Backups fest")
@is_owner()
@app_commands.describe(channel="Der Kanal für Backups")
async def set_backup_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    db["backup_channel_id"] = channel.id
    save_db(db)
    embed = success_embed("Backup-Kanal gesetzt", bullet(f"Backups werden ab jetzt in {channel.mention} gepostet."))
    await interaction.response.send_message(embed=with_executor(embed, interaction.user), ephemeral=True)


@bot.tree.command(name="backup", description="Erstellt ein Backup der Datenbank im Backup-Kanal")
@is_owner()
async def backup(interaction: discord.Interaction):
    if not db["backup_channel_id"]:
        await interaction.response.send_message(
            embed=error_embed("Kein Backup-Kanal", "Nutze zuerst `/set-backup-channel`."),
            ephemeral=True
        )
        return
    backup_channel = interaction.guild.get_channel(db["backup_channel_id"])
    if not backup_channel:
        await interaction.response.send_message(
            embed=error_embed("Kanal nicht gefunden", "Der gesetzte Backup-Kanal existiert nicht mehr."),
            ephemeral=True
        )
        return

    save_db(db)
    buffer = BytesIO(json.dumps(db, ensure_ascii=False, indent=2).encode("utf-8"))
    timestamp = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    file = discord.File(buffer, filename=DB_PATH)  # gleicher Name wie die lokale DB -> /reload passt immer

    embed = base_embed("Datenbank-Backup", bullet(f"Erstellt am {timestamp} (Europe/Berlin)"))
    embed = with_executor(embed, interaction.user)
    await backup_channel.send(embed=embed, file=file)

    await interaction.response.send_message(
        embed=success_embed("Backup erstellt", bullet(f"Backup wurde in {backup_channel.mention} gepostet.")),
        ephemeral=True
    )


@bot.tree.command(name="reload", description="Lädt die Datenbank neu (aus Anhang oder letztem Backup)")
@is_owner()
@app_commands.describe(attachment="Optional: JSON-Datei zum Wiederherstellen. Leer = letztes Backup aus dem Backup-Kanal")
async def reload_db(interaction: discord.Interaction, attachment: discord.Attachment | None = None):
    global db
    await interaction.response.defer(ephemeral=True)

    json_bytes = None
    source_desc = ""

    if attachment:
        if not attachment.filename.endswith(".json"):
            await interaction.followup.send(embed=error_embed("Ungültige Datei", "Bitte eine .json-Datei anhängen."))
            return
        json_bytes = await attachment.read()
        source_desc = f"Anhang `{attachment.filename}`"
    else:
        if not db["backup_channel_id"]:
            await interaction.followup.send(embed=error_embed("Kein Backup-Kanal", "Nutze zuerst `/set-backup-channel`."))
            return
        backup_channel = interaction.guild.get_channel(db["backup_channel_id"])
        if not backup_channel:
            await interaction.followup.send(embed=error_embed("Kanal nicht gefunden", "Der Backup-Kanal existiert nicht mehr."))
            return

        found = None
        async for msg in backup_channel.history(limit=200):
            for att in msg.attachments:
                if att.filename.endswith(".json"):
                    found = att
                    break
            if found:
                break

        if not found:
            await interaction.followup.send(embed=error_embed("Kein Backup gefunden", "Im Backup-Kanal wurde keine .json-Datei gefunden."))
            return

        json_bytes = await found.read()
        source_desc = f"letztes Backup `{found.filename}`"

    try:
        new_data = json.loads(json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        await interaction.followup.send(embed=error_embed("Ungültiges Format", "Die Datei enthält kein gültiges JSON."))
        return

    for key, value in DEFAULT_DB.items():
        if key not in new_data:
            new_data[key] = value

    db = new_data
    save_db(db)

    embed = success_embed("Datenbank geladen", bullet(f"Quelle: {source_desc}"))
    await interaction.followup.send(embed=with_executor(embed, interaction.user), ephemeral=True)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@bot.tree.command(name="status", description="Zeigt den aktuellen Bot-Status inkl. API-Latenzen")
@is_owner()
async def status(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)

    total_messages = sum(len(v) for v in db["messages"].values())
    qa_channel = interaction.guild.get_channel(db["qa_channel_id"]) if db["qa_channel_id"] else None
    backup_channel = interaction.guild.get_channel(db["backup_channel_id"]) if db["backup_channel_id"] else None

    (gemini_ms, gemini_error), (groq_ms, groq_error) = await asyncio.gather(measure_gemini_latency(), measure_groq_latency())
    discord_ms = bot.latency * 1000

    lines = [
        bullet(f"Indizierte Kanäle: {len(db['indexed_channels'])}"),
        bullet(f"Gespeicherte Nachrichten: {total_messages}"),
        bullet(f"Q&A-Kanal: {qa_channel.mention if qa_channel else 'nicht gesetzt'}"),
        bullet(f"Backup-Kanal: {backup_channel.mention if backup_channel else 'nicht gesetzt'}"),
    ]
    latency_lines = [
        f"{DISCORD_ICON} Discord   {latency_bar(discord_ms, max_ms=300)}",
        f"{GEMINI_ICON} Gemini    {latency_bar(gemini_ms, max_ms=1500)}",
        f"{GROQ_ICON} Groq      {latency_bar(groq_ms, max_ms=1500)}",
    ]
    embed = base_embed("Bot-Status", "\n".join(lines))
    embed.add_field(name="API-Latenz", value="\n".join(latency_lines), inline=False)

    if gemini_error:
        embed.add_field(name=f"{GEMINI_ICON} Gemini-Fehlerdetails", value=code_block(gemini_error), inline=False)
        await log_error_to_backup(interaction.guild, "Gemini nicht erreichbar (/status)", gemini_error)
    if groq_error:
        embed.add_field(name=f"{GROQ_ICON} Groq-Fehlerdetails", value=code_block(groq_error), inline=False)
        await log_error_to_backup(interaction.guild, "Groq nicht erreichbar (/status)", groq_error)

    view = discord.ui.View()
    if GEMINI_API_KEY:
        gemini_test_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        button_label = "Gemini-Key direkt testen"
    else:
        gemini_test_url = AI_STUDIO_URL
        button_label = "Google AI Studio öffnen"
    view.add_item(discord.ui.Button(
        label=button_label,
        emoji=GEMINI_ICON,
        style=discord.ButtonStyle.link,
        url=gemini_test_url
    ))

    await interaction.followup.send(embed=with_executor(embed, interaction.user), view=view, ephemeral=True)


# ---------------------------------------------------------------------------
# Fehlerbehandlung für Slash-Befehle
# ---------------------------------------------------------------------------

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        return  # bereits im Check behandelt
    log.error(f"Command-Fehler: {error}")
    embed = error_embed("Unerwarteter Fehler", f"```{str(error)[:1000]}```")
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, ephemeral=True)


# ---------------------------------------------------------------------------
# Flask Keep-Alive (für Render.com)
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def home():
    return "WissensBot läuft."


def run_flask():
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))


def keep_alive():
    thread = Thread(target=run_flask)
    thread.daemon = True
    thread.start()


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

def mask_key(key: str | None) -> str:
    if not key:
        return "nicht gesetzt"
    if len(key) <= 8:
        return f"{key[:2]}... (Länge: {len(key)})"
    return f"{key[:6]}...{key[-4:]} (Länge: {len(key)})"


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN fehlt in der .env Datei.")
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY fehlt – Q&A wird direkt auf Groq zurückfallen.")
    else:
        log.info(f"Gemini-Key geladen: {mask_key(GEMINI_API_KEY)}")
    if not GROQ_API_KEY:
        log.warning("GROQ_API_KEY fehlt – kein Fallback verfügbar, falls Gemini ausfällt.")
    else:
        log.info(f"Groq-Key geladen: {mask_key(GROQ_API_KEY)}")

    keep_alive()
    bot.run(DISCORD_TOKEN)
