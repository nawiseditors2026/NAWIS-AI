import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

TEXTMEBOT_API_KEY = os.environ.get("TEXTMEBOT_API_KEY", "")
TEXTMEBOT_PHONE = os.environ.get("TEXTMEBOT_PHONE", "")

SCHOOL_DOCS_FOLDER = "school_data"
CHROMA_PERSIST_DIR = "chroma_db"

SERIAL_PORT = os.environ.get("SERIAL_PORT", "COM3")
SERIAL_BAUD = 115200

APP_HOST = "0.0.0.0"
APP_PORT = int(os.environ.get("PORT", 8000))
