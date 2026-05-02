# NAWIS AI — School Receptionist Kiosk

AI-powered receptionist for New Al Wurood International School (NAWIS), Jeddah.

## Quick Start

```bash
cd artifacts/nawis-ai
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in Chrome (fullscreen with F11).

## Environment Variables / Secrets

Set these in your environment or `.env`:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (console.groq.com) |
| `TEXTMEBOT_API_KEY` | TextMeBot API key (textmebot.com) |
| `TEXTMEBOT_PHONE` | WhatsApp phone for alerts (e.g. +966501234567) |
| `SERIAL_PORT` | ESP32 port (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux) |

## Adding School Documents

Drop any of these file types into the `school_data/` folder:
- `.txt` — plain text files
- `.pdf` — PDF documents
- `.csv` — spreadsheets/tables
- `.docx` — Word documents

The app automatically indexes them on startup using ChromaDB. If documents haven't changed, it loads the existing index (fast). To force a re-index, delete the `chroma_db/` folder.

## ESP32 LED Controller

Upload `esp32_code/nawis_led.ino` to your ESP32 board using the Arduino IDE. Wire LEDs to pins 12, 13, 14, 15, 16 and a buzzer to pin 26.

## Project Structure

```
nawis-ai/
├── app.py              # FastAPI backend
├── config.py           # Configuration (reads from env vars)
├── requirements.txt    # Python dependencies
├── school_data/        # Drop school documents here
├── chroma_db/          # Auto-generated vector index (gitignore this)
├── frontend/
│   └── index.html      # Kiosk UI (served at /)
└── esp32_code/
    └── nawis_led.ino   # Arduino code for ESP32
```
