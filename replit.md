# NAWIS AI — Al Wurood International School Kiosk

AI-powered school receptionist kiosk for New Al Wurood International School (NAWIS), Jeddah, Saudi Arabia.

## Overview

A fullscreen kiosk web application powered by Python FastAPI + Groq LLM with in-memory document search, bilingual EN/AR support, voice I/O, animated orb UI, and ESP32 LED signaling.

## Stack

- **Backend**: Python 3.11 + FastAPI + Uvicorn
- **AI Model**: Groq API (`llama-3.3-70b-versatile`)
- **Document Search**: In-memory BM25-style retrieval (no heavy ML deps)
- **WhatsApp Alerts**: TextMeBot HTTP API
- **ESP32**: PySerial for LED status signaling
- **Frontend**: Vanilla HTML/CSS/JS (served as static file by FastAPI)
- **Monorepo tool**: pnpm workspaces (Node.js side)

## Project Structure

```
artifacts/nawis-ai/
├── app.py              # FastAPI backend (RAG, Groq, ESP32, WhatsApp)
├── config.py           # Configuration — reads from environment variables
├── requirements.txt    # Python dependencies
├── school_data/        # Drop school documents here (.txt, .pdf, .docx, .csv)
│   └── school_info.txt # Starter school data + STEAM info
├── frontend/
│   └── index.html      # Full kiosk UI (vanilla JS/CSS)
├── esp32_code/
│   └── nawis_led.ino   # Arduino code for ESP32 LED controller
└── README.md           # Setup and deployment guide
```

## Required Secrets

| Secret | Description |
|--------|-------------|
| `GROQ_API_KEY` | Groq API key — enables AI chat responses |
| `TEXTMEBOT_API_KEY` | TextMeBot API key — WhatsApp escalation alerts |
| `TEXTMEBOT_PHONE` | Phone number for WhatsApp alerts (e.g. +966501234567) |
| `SERIAL_PORT` | ESP32 port (e.g. COM3 on Windows, /dev/ttyUSB0 on Linux) |

## Key Commands

- **Start server**: `cd artifacts/nawis-ai && uvicorn app:app --host 0.0.0.0 --port 8000 --reload`
- **Install deps**: `cd artifacts/nawis-ai && pip install -r requirements.txt`

## Features

- **Animated state orb**: idle (breathing green), listening (blue ripple), thinking (gold spin), speaking (oscillate), escalating (red pulse)
- **Voice I/O**: Web Speech API for STT + TTS in EN and AR
- **Bilingual**: Full English/Arabic with RTL layout switching
- **FAQ chips**: 10 quick-tap questions including STEAM challenge
- **Escalation**: Auto WhatsApp alert + overlay when AI can't help
- **ESP32**: Serial LED control (I/L/T/S/E characters)
- **Graceful fallbacks**: Works without ESP32, without documents, warns without Groq key

## Adding School Documents

Drop any `.txt`, `.pdf`, `.docx`, or `.csv` files into `artifacts/nawis-ai/school_data/`. Restart the server — documents are indexed in memory on startup. No rebuild needed.
