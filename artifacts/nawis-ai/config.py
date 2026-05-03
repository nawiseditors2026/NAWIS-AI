import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Primary model: llama-3.1-8b-instant ────────────────────────────────────────
# Chosen over llama-3.3-70b-versatile because:
#   • 500,000 tokens/day free limit (vs 100K for 70B — which was getting exhausted)
#   • ~5x faster response time — better kiosk UX
#   • Fully sufficient for short factual Q&A from documents
# Fallback chain is tried in order if the primary hits a rate limit.
GROQ_MODEL        = "llama-3.1-8b-instant"
GROQ_MODEL_CHAIN  = [
    "llama-3.1-8b-instant",   # primary — 500K TPD
    "gemma2-9b-it",            # first fallback — 15K RPD
    "llama3-8b-8192",          # last resort
]
GROQ_WHISPER_MODEL = "whisper-large-v3"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

CALLMEBOT_API_KEY = os.environ.get("CALLMEBOT_API_KEY", "")
CALLMEBOT_PHONE   = os.environ.get("CALLMEBOT_PHONE", "")

SCHOOL_DOCS_FOLDER = "school_data"

SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM3")
SERIAL_BAUD = 115200

TTS_VOICE_EN = "en-US-EmmaNeural"
TTS_VOICE_AR = "ar-SA-ZariyahNeural"
