"""
NAWIS AI — FastAPI Backend
AI-powered school receptionist for New Al Wurood International School
Uses in-memory document search (no heavy ML deps) + Groq LLM
"""
import os
import asyncio
import logging
import time
import csv
import json
import re
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from collections import Counter
from math import log

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    TEXTMEBOT_API_KEY, TEXTMEBOT_PHONE,
    SCHOOL_DOCS_FOLDER,
    SERIAL_PORT, SERIAL_BAUD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nawis-ai")

# ── Global state ──────────────────────────────────────────────────────────────
document_chunks: list[dict] = []
groq_client: Groq | None = None
serial_conn = None
docs_count = 0

SYSTEM_PROMPT = """\
You are NAWIS AI, the official AI receptionist of New Al Wurood International \
School (NAWIS) in Jeddah, Saudi Arabia. You speak on behalf of the school \
warmly and professionally.

YOUR JOB: Answer questions from parents and students about the school — \
admissions, academics, facilities, events, staff, rules, transport, results, \
and anything a school receptionist would know.

RULES:
1. Keep answers concise — 2 to 4 sentences. Never ramble.
2. Be warm, friendly, and professional.
3. Use the SCHOOL CONTEXT provided below as your primary source. \
If the context has the answer, use it.
4. NEVER reveal specific student records, private teacher contact details, \
internal financial details, or anything that could embarrass the school.
5. If a question is about sensitive private information, or if you genuinely \
cannot find the answer in the context, end your entire response with the word \
ESCALATE on its own line. Do not announce you are escalating — just add the \
word at the end.
6. If the user's message is in Arabic, respond in Arabic.
7. Always refer to the school as NAWIS or Al Wurood.

SCHOOL CONTEXT FROM DOCUMENTS:
{context}"""

ESP_STATE_MAP = {
    "idle":   "I",
    "listen": "L",
    "think":  "T",
    "speak":  "S",
    "error":  "E",
}


# ── Document loading ──────────────────────────────────────────────────────────

def _chunk_text(text: str, source: str, chunk_words: int = 400) -> list[dict]:
    words = text.split()
    chunks = []
    for i in range(0, max(len(words), 1), chunk_words):
        chunk = " ".join(words[i : i + chunk_words]).strip()
        if chunk:
            chunks.append({"text": chunk, "source": source, "id": f"{source}_{i}"})
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
                    reader = PdfReader(str(fp))
                    text = "\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    logger.warning("pypdf not installed — skipping %s", fp.name)

            elif suffix == ".docx":
                try:
                    from docx import Document
                    doc = Document(str(fp))
                    text = "\n".join(p.text for p in doc.paragraphs)
                except ImportError:
                    logger.warning("python-docx not installed — skipping %s", fp.name)

            elif suffix == ".csv":
                with open(fp, newline="", encoding="utf-8", errors="ignore") as f:
                    reader_obj = csv.reader(f)
                    text = "\n".join(", ".join(row) for row in reader_obj)

            if text.strip():
                all_chunks.extend(_chunk_text(text, fp.stem))
                logger.info("Loaded: %s (%d chars)", fp.name, len(text))

        except Exception as exc:
            logger.error("Error loading %s: %s", fp.name, exc)

    docs_count = len(all_chunks)
    logger.info("Documents loaded: %d chunks from %s", docs_count, SCHOOL_DOCS_FOLDER)
    return all_chunks


# ── Lightweight BM25-style retrieval ─────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u0600-\u06FF]+", text.lower())


def query_context(question: str, n: int = 5) -> str:
    if not document_chunks:
        return ""

    q_terms = Counter(_tokenize(question))
    if not q_terms:
        return ""

    scored: list[tuple[float, str]] = []
    for chunk in document_chunks:
        chunk_terms = Counter(_tokenize(chunk["text"]))
        total = sum(chunk_terms.values()) or 1
        score = 0.0
        for term, qf in q_terms.items():
            tf = chunk_terms.get(term, 0) / total
            # Simple TF × log(1 + query_freq) scoring
            score += tf * log(1 + qf)
        if score > 0:
            scored.append((score, chunk["text"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [text for _, text in scored[:n]]
    return "\n\n---\n\n".join(top)


# ── ESP32 serial ──────────────────────────────────────────────────────────────

def init_serial() -> None:
    global serial_conn
    try:
        import serial
        serial_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        logger.info("ESP32 connected on %s", SERIAL_PORT)
    except Exception as exc:
        logger.warning("ESP32 not available (%s) — continuing without it.", exc)
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


# ── WhatsApp alert via TextMeBot ──────────────────────────────────────────────

async def send_whatsapp_alert(question: str) -> None:
    if not TEXTMEBOT_PHONE or not TEXTMEBOT_API_KEY:
        logger.warning("TextMeBot not configured — skipping WhatsApp alert.")
        return
    now = datetime.now().strftime("%H:%M")
    text = (
        f"🔔 NAWIS AI Alert: A visitor needs help with: "
        f"{question[:100]}. Please come to reception. Time: {now}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.textmebot.com/send.php",
                params={"recipient": TEXTMEBOT_PHONE, "apikey": TEXTMEBOT_API_KEY, "text": text},
            )
            logger.info("TextMeBot: %s %s", r.status_code, r.text[:80])
    except Exception as exc:
        logger.error("TextMeBot error: %s", exc)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client, document_chunks

    logger.info("═" * 55)
    logger.info("  NAWIS AI  —  starting up…")
    logger.info("═" * 55)

    document_chunks = load_documents()

    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client ready  (%s)", GROQ_MODEL)
    else:
        logger.warning("GROQ_API_KEY not set — AI responses will be disabled.")

    init_serial()
    send_esp32("I")
    logger.info("NAWIS AI ready ✓")
    yield

    if serial_conn and serial_conn.is_open:
        serial_conn.close()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="NAWIS AI", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    language: str = "en"


# ── API routes ────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    send_esp32("T")

    context = query_context(req.message)
    prompt = SYSTEM_PROMPT.format(
        context=context if context else "No document context available — answer from general knowledge about the school."
    )

    if groq_client is None:
        send_esp32("I")
        return {
            "reply": (
                "I'm sorry, the AI service is not configured yet. "
                "Please contact the school office at +966 55 273 0945."
            ),
            "escalate": False,
        }

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": req.message},
            ],
            max_tokens=350,
            temperature=0.65,
        )
        raw = response.choices[0].message.content or ""
        should_escalate = "ESCALATE" in raw
        clean = raw.replace("ESCALATE", "").strip()

        if should_escalate:
            send_esp32("E")
            asyncio.create_task(send_whatsapp_alert(req.message))
            return {"reply": clean, "escalate": True}

        send_esp32("S")
        return {"reply": clean, "escalate": False}

    except Exception as exc:
        logger.error("Groq error: %s", exc)
        send_esp32("I")
        return {
            "reply": "I'm having a moment — please try again or speak to our front desk staff directly.",
            "escalate": False,
        }


@app.get("/status")
async def status():
    return {
        "status": "ready",
        "docs_loaded": docs_count,
        "rag_ready": len(document_chunks) > 0,
        "esp32_connected": (serial_conn is not None and serial_conn.is_open) if serial_conn else False,
        "groq_ready": groq_client is not None,
        "model": GROQ_MODEL,
    }


@app.post("/esp32/{state}")
async def esp32_state(state: str):
    char = ESP_STATE_MAP.get(state)
    if not char:
        raise HTTPException(status_code=400, detail=f"Unknown state '{state}'")
    send_esp32(char)
    return {"ok": True, "state": state, "char": char}


# ── Frontend ──────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)


@app.get("/")
async def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "NAWIS AI is running. Place index.html in the frontend/ folder."}


# Mount static files — API routes above take priority
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
