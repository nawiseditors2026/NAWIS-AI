"""
NAWIS AI — FastAPI Backend v3
Bilingual AI receptionist for New Al Wurood International School, Jeddah
"""
import os
import asyncio
import logging
import time
import csv
import re
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from groq import Groq

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_WHISPER_MODEL,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    CALLMEBOT_API_KEY, CALLMEBOT_PHONE,
    SCHOOL_DOCS_FOLDER, SERIAL_PORT, SERIAL_BAUD,
    TTS_VOICE_EN, TTS_VOICE_AR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nawis-ai")

# ── Globals ────────────────────────────────────────────────────────────────────
bm25_index   = None
doc_chunks: list[dict] = []
groq_client: Groq | None = None
serial_conn  = None
docs_count   = 0

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are NAWIS AI — the smart bilingual AI receptionist at the front desk of \
New Al Wurood International School (NAWIS), Jeddah, Saudi Arabia.

CORE RULES (follow every single one, every time):
1. ANSWER LENGTH: Maximum 2–3 short sentences. Never exceed this. No lengthy paragraphs.
2. USE THE DOCUMENTS: Read the SCHOOL DOCUMENTS section below carefully. \
   If the answer is there, use it directly and precisely. Do not paraphrase unnecessarily.
3. LANGUAGE: Detect the language of the user's message. \
   If English → reply in English only. If Arabic → reply in Arabic only. Never mix.
4. NO FILLER: Never start with "Certainly!", "Of course!", "Great question!", \
   "Sure!", or any similar opener. Start directly with the answer.
5. CONTACT: Whenever relevant, include: Phone +966 55 273 0945 | Email admin@alwuroodschool.org
6. ESCALATE: If the question involves private student records, individual finances, \
   disciplinary matters, complaints, or anything you cannot find in the documents, \
   write the single word ESCALATE on its own line at the very end of your response. \
   Do not announce that you are escalating — just add the word.
7. SCHOOL NAME: Always call it "NAWIS" or "Al Wurood International School".

SCHOOL DOCUMENTS — your authoritative knowledge base:
{context}

If the documents answer the question, cite from them. \
If the documents are silent and the topic is general (e.g. "what is CBSE?"), \
you may answer briefly from general knowledge, but keep it to 1–2 sentences."""

ESP_MAP = {"idle": "I", "listen": "L", "think": "T", "speak": "S", "error": "E"}


# ── Tokeniser (EN + AR) ────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u0600-\u06FF0-9]+", text.lower())


# ── Document loader ────────────────────────────────────────────────────────────
def _chunk_text(text: str, source: str, max_words: int = 400) -> list[dict]:
    words = text.split()
    chunks = []
    for i in range(0, max(len(words), 1), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if len(chunk) > 40:
            chunks.append({"text": chunk, "source": source})
    return chunks


def load_documents() -> list[dict]:
    global docs_count
    folder = Path(SCHOOL_DOCS_FOLDER)
    folder.mkdir(exist_ok=True)
    all_chunks: list[dict] = []

    for fp in sorted(folder.iterdir()):
        if not fp.is_file():
            continue
        try:
            text  = ""
            suffix = fp.suffix.lower()
            if suffix == ".txt":
                text = fp.read_text(encoding="utf-8", errors="ignore")
            elif suffix == ".pdf":
                try:
                    from pypdf import PdfReader
                    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(fp)).pages)
                except ImportError:
                    logger.warning("pypdf not installed — skipping %s", fp.name)
            elif suffix == ".docx":
                try:
                    from docx import Document
                    text = "\n".join(p.text for p in Document(str(fp)).paragraphs)
                except ImportError:
                    logger.warning("python-docx not installed — skipping %s", fp.name)
            elif suffix == ".csv":
                with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
                    text = "\n".join(", ".join(row) for row in csv.reader(f))

            if text.strip():
                all_chunks.extend(_chunk_text(text, fp.stem))
                logger.info("Loaded: %s (%d chars)", fp.name, len(text))
        except Exception as exc:
            logger.error("Error loading %s: %s", fp.name, exc)

    docs_count = len(all_chunks)
    logger.info("Total chunks indexed: %d", docs_count)
    return all_chunks


def build_bm25(chunks: list[dict]) -> None:
    global bm25_index
    if not chunks:
        return
    try:
        from rank_bm25 import BM25Okapi
        tokenized  = [tokenize(c["text"]) for c in chunks]
        bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index ready (%d chunks)", len(chunks))
    except Exception as exc:
        logger.error("BM25 build failed: %s", exc)
        bm25_index = None


def query_context(question: str, n: int = 8) -> str:
    """Return the top-n most relevant document chunks for the question."""
    if not doc_chunks:
        return ""
    tokens = tokenize(question)
    if not tokens:
        return ""

    if bm25_index is not None:
        try:
            scores  = bm25_index.get_scores(tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
            results = [(scores[i], doc_chunks[i]["text"]) for i in top_idx if scores[i] > 0]
        except Exception:
            results = []
    else:
        from collections import Counter
        from math import log
        q_terms = Counter(tokens)
        results  = []
        for chunk in doc_chunks:
            ct    = Counter(tokenize(chunk["text"]))
            total = sum(ct.values()) or 1
            score = sum((ct[t] / total) * log(1 + qf) for t, qf in q_terms.items() if t in ct)
            if score > 0:
                results.append((score, chunk["text"]))
        results.sort(reverse=True)
        results = results[:n]

    return "\n\n---\n\n".join(txt for _, txt in results)


# ── ESP32 ──────────────────────────────────────────────────────────────────────
def init_serial() -> None:
    global serial_conn
    try:
        import serial
        serial_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        logger.info("ESP32 connected on %s", SERIAL_PORT)
    except Exception as exc:
        logger.warning("ESP32 not available (%s)", exc)
        serial_conn = None


def send_esp32(char: str) -> None:
    global serial_conn
    if serial_conn is None:
        return
    try:
        if serial_conn.is_open:
            serial_conn.write(char.encode())
    except Exception as exc:
        logger.warning("ESP32 write error: %s", exc)
        serial_conn = None


# ── Alerts ─────────────────────────────────────────────────────────────────────

async def send_whatsapp_alert(question: str) -> None:
    """Send WhatsApp message via CallMeBot (free, no linked device required).
    Setup: headmaster adds +34 644 97 79 26 on WhatsApp, sends 'I allow callmebot
    to send me messages', receives API key. Set CALLMEBOT_API_KEY + CALLMEBOT_PHONE."""
    now  = datetime.now().strftime("%H:%M")
    text = (
        f"\U0001F514 NAWIS AI Alert\n\n"
        f"A visitor needs assistance:\n{question[:220]}\n\n"
        f"Please come to reception. Time: {now}"
    )

    if CALLMEBOT_API_KEY and CALLMEBOT_PHONE:
        try:
            async with httpx.AsyncClient(timeout=12) as client:
                r = await client.get(
                    "https://api.callmebot.com/whatsapp.php",
                    params={"phone": CALLMEBOT_PHONE, "text": text, "apikey": CALLMEBOT_API_KEY},
                )
                logger.info("CallMeBot WhatsApp: %s", r.status_code)
                return
        except Exception as exc:
            logger.error("CallMeBot error: %s — falling back to Telegram", exc)

    # Fallback: Telegram
    await send_telegram_alert(question)


async def send_telegram_alert(question: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("No alert channel configured (CallMeBot or Telegram).")
        return
    now  = datetime.now().strftime("%H:%M")
    text = (
        f"\U0001F514 *NAWIS AI \u2014 Escalation*\n\n"
        f"A visitor needs assistance with:\n_{question[:220]}_\n\n"
        f"\u23F0 *Time:* {now}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            )
            logger.info("Telegram alert: %s", r.status_code)
    except Exception as exc:
        logger.error("Telegram error: %s", exc)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, doc_chunks
    logger.info("=" * 56)
    logger.info("  NAWIS AI  v3  —  starting up…")
    logger.info("=" * 56)

    doc_chunks = load_documents()
    build_bm25(doc_chunks)

    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq ready  ·  LLM: %s  ·  STT: %s", GROQ_MODEL, GROQ_WHISPER_MODEL)
    else:
        logger.warning("GROQ_API_KEY not set — AI responses and STT disabled.")

    whatsapp_ok = bool(CALLMEBOT_API_KEY and CALLMEBOT_PHONE)
    telegram_ok = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    logger.info(
        "Alerts: WhatsApp/CallMeBot=%s  Telegram=%s",
        "\u2713" if whatsapp_ok else "\u2717",
        "\u2713" if telegram_ok else "\u2717",
    )

    init_serial()
    send_esp32("I")
    logger.info("NAWIS AI ready \u2713")
    yield

    if serial_conn and serial_conn.is_open:
        serial_conn.close()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="NAWIS AI", version="3.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    language: str = "en"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    send_esp32("T")
    context = query_context(req.message)
    prompt  = SYSTEM_PROMPT.format(
        context=context or "No relevant documents found — use general school knowledge."
    )

    if groq_client is None:
        send_esp32("I")
        return {
            "reply": "The AI service is not configured. Please contact the school office at +966 55 273 0945.",
            "escalate": False,
        }

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=220,       # short answers enforced at token level too
            temperature=0.4,
        )
        raw      = resp.choices[0].message.content or ""
        escalate = "ESCALATE" in raw
        clean    = raw.replace("ESCALATE", "").strip()

        if escalate:
            send_esp32("E")
            asyncio.create_task(send_telegram_alert(req.message))
        else:
            send_esp32("S")

        return {"reply": clean, "escalate": escalate}

    except Exception as exc:
        logger.error("Groq LLM error: %s", exc)
        send_esp32("I")
        return {
            "reply": "I'm having a moment — please try again or speak to our front desk staff.",
            "escalate": False,
        }


@app.post("/stt")
async def speech_to_text(audio: UploadFile = File(...), lang: str = Query("en")):
    """Transcribe audio using Groq Whisper large-v3 with school-specific priming."""
    if groq_client is None:
        return JSONResponse({"text": "", "error": "Groq not configured"}, status_code=503)

    data = await audio.read()
    if len(data) < 500:
        return JSONResponse({"text": "", "error": "Audio too short"})

    lang_code = "ar" if lang == "ar" else "en"
    fname     = audio.filename or "recording.webm"
    mime      = audio.content_type or "audio/webm"

    whisper_prompt = (
        "NAWIS, Al Wurood, Jeddah, Saudi Arabia, CBSE, admissions, school fees, "
        "Grade 10, Grade 12, principal, teacher, exam, result, transport, uniform, PTM."
        if lang_code == "en" else
        "مدرسة النورود، جدة، المملكة العربية السعودية، القبول، الرسوم، الحافلة، الامتحانات، المدير."
    )

    try:
        result = groq_client.audio.transcriptions.create(
            model       = GROQ_WHISPER_MODEL,
            file        = (fname, data, mime),
            language    = lang_code,
            response_format = "json",
            prompt      = whisper_prompt,
            temperature = 0.0,
        )
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()

        # Drop single-word noise results that are very short
        if len(text.split()) <= 1 and len(text) < 4:
            return {"text": ""}

        logger.info("STT [%s]: %s…", lang, text[:80])
        return {"text": text}
    except Exception as exc:
        logger.error("Whisper error: %s", exc)
        return JSONResponse({"text": "", "error": str(exc)}, status_code=500)


@app.get("/tts")
async def text_to_speech(text: str = Query(...), lang: str = Query("en")):
    """TTS via edge-tts (Microsoft Neural). Works perfectly when running locally."""
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(503, "edge-tts not installed")

    voice      = TTS_VOICE_AR if lang == "ar" else TTS_VOICE_EN
    clean_text = text[:600].strip()
    if not clean_text:
        raise HTTPException(400, "Empty text")

    buf = bytearray()
    try:
        communicate = edge_tts.Communicate(text=clean_text, voice=voice, rate="+5%")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
    except Exception as exc:
        logger.error("edge-tts error: %s", exc)
        raise HTTPException(502, f"TTS upstream error: {exc}")

    if not buf:
        raise HTTPException(502, "edge-tts returned no audio")

    return StreamingResponse(
        iter([bytes(buf)]),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Content-Length": str(len(buf)),
        },
    )


@app.get("/api/teachers")
async def get_teachers():
    """Return structured teacher list parsed from school_data/teachers.txt"""
    teachers_file = Path(SCHOOL_DOCS_FOLDER) / "teachers.txt"
    if not teachers_file.exists():
        return []
    text = teachers_file.read_text(encoding="utf-8", errors="ignore")
    teachers = []
    blocks = [b.strip() for b in text.split("---") if b.strip() and not b.strip().startswith("#")]
    for block in blocks:
        teacher: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                teacher[key.strip().lower().replace(" ", "_")] = val.strip()
        name = teacher.get("name", "")
        if name and not name.startswith("["):
            teachers.append(teacher)
    return teachers


@app.get("/status")
async def status():
    return {
        "status"            : "ready",
        "docs_loaded"       : docs_count,
        "rag_ready"         : bm25_index is not None,
        "groq_ready"        : groq_client is not None,
        "esp32_connected"   : bool(serial_conn and serial_conn.is_open),
        "model"             : GROQ_MODEL,
        "whisper_model"     : GROQ_WHISPER_MODEL,
        "whatsapp_ready"    : bool(CALLMEBOT_API_KEY and CALLMEBOT_PHONE),
        "telegram_ready"    : bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }


@app.post("/esp32/{state}")
async def esp32_state(state: str):
    char = ESP_MAP.get(state)
    if not char:
        raise HTTPException(400, f"Unknown state '{state}'")
    send_esp32(char)
    return {"ok": True}


# ── Static frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
