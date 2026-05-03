"""
NAWIS AI — FastAPI Backend v2
AI-powered school receptionist for New Al Wurood International School
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
    SCHOOL_DOCS_FOLDER, SERIAL_PORT, SERIAL_BAUD,
    TTS_VOICE_EN, TTS_VOICE_AR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nawis-ai")

# ── Global state ──────────────────────────────────────────────────────────────
bm25_index = None
doc_chunks: list[dict] = []
groq_client: Groq | None = None
serial_conn = None
docs_count = 0

SYSTEM_PROMPT = """\
You are NAWIS AI, the official AI receptionist of New Al Wurood International \
School (NAWIS) in Jeddah, Saudi Arabia. You speak on behalf of the school warmly \
and professionally.

YOUR JOB: Answer questions from parents and students about the school — admissions, \
academics, facilities, events, staff, rules, transport, results, and anything a \
school receptionist would know.

RULES:
1. Keep answers concise — 2 to 4 sentences. Never ramble.
2. Be warm, friendly, and professional at all times.
3. Use the SCHOOL CONTEXT provided below as your primary source. If the context \
has the answer, use it directly.
4. NEVER reveal specific student records, private teacher contact details, internal \
financial details, or anything that could embarrass the school.
5. If a question is about sensitive private information, or if you genuinely cannot \
find the answer in the context, end your entire response with the word ESCALATE on \
its own line. Do not announce you are escalating — just add the word at the end.
6. If the user's message is in Arabic, respond entirely in Arabic.
7. Always refer to the school as NAWIS or Al Wurood.

SCHOOL CONTEXT FROM DOCUMENTS:
{context}
"""

ESP_MAP = {"idle": "I", "listen": "L", "think": "T", "speak": "S", "error": "E"}


# ── Tokenizer (EN + AR) ───────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u0600-\u06FF0-9]+", text.lower())


# ── Document loading ──────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str, max_words: int = 350) -> list[dict]:
    words = text.split()
    chunks = []
    for i in range(0, max(len(words), 1), max_words):
        chunk = " ".join(words[i : i + max_words]).strip()
        if len(chunk) > 30:
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
            text = ""
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
    logger.info("Total chunks: %d", docs_count)
    return all_chunks


def build_bm25(chunks: list[dict]):
    global bm25_index
    if not chunks:
        return
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [tokenize(c["text"]) for c in chunks]
        bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index built (%d chunks)", len(chunks))
    except Exception as exc:
        logger.error("BM25 build failed: %s", exc)
        bm25_index = None


def query_context(question: str, n: int = 5) -> str:
    if not doc_chunks:
        return ""
    tokens = tokenize(question)
    if not tokens:
        return ""

    if bm25_index is not None:
        try:
            scores = bm25_index.get_scores(tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
            results = [(scores[i], doc_chunks[i]["text"]) for i in top_idx if scores[i] > 0]
        except Exception:
            results = []
    else:
        # Fallback: simple keyword scoring
        from collections import Counter
        from math import log
        q_terms = Counter(tokens)
        results = []
        for chunk in doc_chunks:
            ct = Counter(tokenize(chunk["text"]))
            total = sum(ct.values()) or 1
            score = sum((ct[t] / total) * log(1 + qf) for t, qf in q_terms.items() if t in ct)
            if score > 0:
                results.append((score, chunk["text"]))
        results.sort(reverse=True)
        results = results[:n]

    return "\n\n---\n\n".join(txt for _, txt in results)


# ── ESP32 serial ──────────────────────────────────────────────────────────────

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


# ── Telegram alert ────────────────────────────────────────────────────────────

async def send_telegram_alert(question: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured — skipping alert.")
        return
    now = datetime.now().strftime("%H:%M")
    text = (
        f"\U0001F514 *NAWIS AI \u2014 Escalation Alert*\n\n"
        f"A visitor needs assistance with:\n_{question[:250]}_\n\n"
        f"Please come to the reception desk.\n\u23F0 *Time:* {now}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            )
            logger.info("Telegram alert sent: %s", r.status_code)
    except Exception as exc:
        logger.error("Telegram error: %s", exc)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, doc_chunks

    logger.info("=" * 56)
    logger.info("  NAWIS AI  v2  —  starting up…")
    logger.info("=" * 56)

    doc_chunks = load_documents()
    build_bm25(doc_chunks)

    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq ready  |  LLM: %s  |  STT: %s", GROQ_MODEL, GROQ_WHISPER_MODEL)
    else:
        logger.warning("GROQ_API_KEY not set — AI responses and STT disabled.")

    init_serial()
    send_esp32("I")
    logger.info("NAWIS AI ready \u2713")
    yield

    if serial_conn and serial_conn.is_open:
        serial_conn.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NAWIS AI", version="2.0.0", lifespan=lifespan)


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    language: str = "en"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    send_esp32("T")

    context = query_context(req.message)
    prompt = SYSTEM_PROMPT.format(
        context=context or "No document context available — answer from general school knowledge."
    )

    if groq_client is None:
        send_esp32("I")
        return {
            "reply": "The AI service is not configured yet. Please contact the school office at +966 55 273 0945.",
            "escalate": False,
        }

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=400,
            temperature=0.6,
        )
        raw = resp.choices[0].message.content or ""
        escalate = "ESCALATE" in raw
        clean = raw.replace("ESCALATE", "").strip()

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
    """Transcribe audio using Groq Whisper."""
    if groq_client is None:
        return JSONResponse({"text": "", "error": "Groq not configured"}, status_code=503)

    data = await audio.read()
    if len(data) < 500:
        return JSONResponse({"text": "", "error": "Audio too short or empty"})

    lang_code = "ar" if lang == "ar" else "en"
    fname = audio.filename or "recording.webm"
    mime = audio.content_type or "audio/webm"

    # Domain-specific prompt dramatically improves Whisper accuracy for school vocab
    whisper_prompt = (
        "NAWIS, Al Wurood International School, Jeddah, Saudi Arabia, "
        "CBSE curriculum, admissions, school fees, transport, bus, "
        "Grade 10, Grade 12, Science stream, Commerce stream, "
        "principal, teacher, exam, result, uniform, PTM, parent teacher meeting."
        if lang != "ar" else
        "مدرسة النورود، جدة، المملكة العربية السعودية، القبول، الرسوم الدراسية، "
        "الحافلة، المنهج، الامتحانات، النتائج، المعلم، المدير."
    )

    try:
        result = groq_client.audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=(fname, data, mime),
            language=lang_code,
            response_format="json",
            prompt=whisper_prompt,
            temperature=0.0,   # Deterministic — reduces hallucination
        )
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()

        # Basic hallucination filter: reject suspiciously short or repeated results
        words = text.split()
        if len(words) <= 1 and len(text) < 3:
            logger.info("STT [%s]: result too short, treating as empty", lang)
            return {"text": ""}

        logger.info("STT [%s]: %s…", lang, text[:80])
        return {"text": text}
    except Exception as exc:
        logger.error("Whisper error: %s", exc)
        return JSONResponse({"text": "", "error": str(exc)}, status_code=500)


@app.get("/tts")
async def text_to_speech(text: str = Query(...), lang: str = Query("en")):
    """Stream natural TTS audio via edge-tts (Microsoft Neural voices).
    Used when running locally — in Replit the browser handles TTS directly."""
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(503, "edge-tts not installed")

    voice = TTS_VOICE_AR if lang == "ar" else TTS_VOICE_EN
    clean_text = text[:600].strip()
    if not clean_text:
        raise HTTPException(400, "Empty text")

    buf = bytearray()
    try:
        communicate = edge_tts.Communicate(text=clean_text, voice=voice, rate="+2%")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
    except Exception as exc:
        logger.error("edge-tts error: %s", exc)
        raise HTTPException(502, f"TTS upstream error: {exc}")

    if not buf:
        raise HTTPException(502, "edge-tts returned no audio data")

    return StreamingResponse(
        iter([bytes(buf)]),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Content-Length": str(len(buf))},
    )


@app.get("/status")
async def status():
    return {
        "status": "ready",
        "docs_loaded": docs_count,
        "rag_ready": bm25_index is not None,
        "esp32_connected": (serial_conn is not None and serial_conn.is_open) if serial_conn else False,
        "groq_ready": groq_client is not None,
        "model": GROQ_MODEL,
        "whisper_model": GROQ_WHISPER_MODEL,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
    }


@app.post("/esp32/{state}")
async def esp32_state(state: str):
    char = ESP_MAP.get(state)
    if not char:
        raise HTTPException(400, f"Unknown state '{state}'")
    send_esp32(char)
    return {"ok": True, "state": state}


# ── Serve frontend ────────────────────────────────────────────────────────────
# FastAPI router routes above always take priority over the static mount below.

FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
