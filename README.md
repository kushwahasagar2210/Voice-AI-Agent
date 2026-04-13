# 🎙️ Voice AI Agent

A local voice-controlled AI agent that accepts audio input, classifies user intent, executes local tools, and displays the full pipeline in a clean Streamlit UI.

Built for the **Mem0 AI/ML & Generative AI Developer Intern Assignment**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎤 Audio input | Microphone recording **or** file upload (WAV, MP3, M4A, OGG, FLAC) |
| 🔊 Speech-to-Text | Groq Whisper API (`whisper-large-v3`) |
| 🧠 Intent Classification | Groq LLM (`llama-3.3-70b-versatile`) |
| 📄 Create File | Creates any file type in the `output/` sandbox |
| 💻 Write Code | Generates & saves code in any language |
| 📋 Summarize | Bullet-point summary with TL;DR |
| 💬 General Chat | Multi-turn conversation with session memory |
| ✋ Human-in-the-Loop | Confirmation prompt before any file operation |
| 📜 Session Log | Timestamped history of all actions taken |

---

## 🏗️ Architecture

```
voice-ai-agent/
├── app.py                 # Streamlit UI — orchestrates the full pipeline
├── stt.py                 # Speech-to-Text via Groq Whisper API
├── intent_classifier.py   # Intent classification via Groq LLM (returns JSON)
├── tools.py               # Tool execution: create_file, generate_code, summarize, chat
├── config.py              # Centralised config (models, paths, env vars)
├── requirements.txt
├── .env.example
├── output/                # 🔒 All file writes are sandboxed here
└── README.md
```

### Pipeline Flow

```
User Audio
    │
    ▼
[STT — Groq Whisper]
    │  transcription (str)
    ▼
[Intent Classifier — Groq LLM]
    │  JSON: {primary_intent, filename, language, content_request, ...}
    ▼
 File op? ──Yes──► [HITL Confirmation] ──Confirmed──► [Tool Execution]
    │                                                        │
    No                                                       │
    ▼                                                        │
[Tool Execution (immediate)]                                 │
    │◄───────────────────────────────────────────────────────┘
    ▼
[Streamlit UI — display result]
```

### Intent → Tool Mapping

| Detected Intent | Tool Called | File Written? |
|---|---|---|
| `create_file` | `tools.create_file()` | ✅ (after confirmation) |
| `write_code` | `tools.generate_code()` | ✅ (after confirmation) |
| `summarize` | `tools.summarize_text()` | ❌ |
| `chat` | `tools.chat()` | ❌ |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/voice-ai-agent.git
cd voice-ai-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your Groq API Key

```bash
cp .env.example .env
# Open .env and replace gsk_your_key_here with your actual key
```

> **Get a free Groq API key** at [console.groq.com](https://console.groq.com).  
> Groq's free tier is generous — Whisper and LLaMA 3 calls are free up to rate limits.

Alternatively, you can enter the API key directly in the app sidebar (no `.env` needed).

### 5. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔧 Hardware Workaround

This project runs **fully on API** rather than local models. Here's why:

| Component | Local Option | Why API was chosen |
|---|---|---|
| STT | Whisper (HuggingFace) | Requires 4–8 GB VRAM; whisper-large-v3 takes ~30s on CPU — unacceptable UX |
| LLM | Ollama (LLaMA 3, Mistral) | 7B models need ~8 GB RAM and are slow on laptop CPUs |
| **Solution** | **Groq Cloud API** | Sub-second inference, free tier, identical model quality |

**Groq** uses custom LPU (Language Processing Unit) hardware to run the same open-source models (Whisper, LLaMA 3, Mixtral) at near-zero latency. The experience is indistinguishable from running locally, but without the hardware requirement.

If you **do** have a capable GPU, you can swap Groq for:
- STT → `openai/whisper-large-v3` via HuggingFace `transformers`
- LLM → Ollama (`ollama run llama3.2`) and point `tools.py` / `intent_classifier.py` at `http://localhost:11434/api/chat`

---

## 💡 Example Commands to Try

| Voice Command | Detected Intent | Action |
|---|---|---|
| "Create a Python file with a retry decorator" | `write_code` | Generates `retry.py` in `output/` |
| "Write a JavaScript function to debounce events" | `write_code` | Generates `debounce.js` in `output/` |
| "Create a new file called notes.txt" | `create_file` | Creates `output/notes.txt` |
| "Summarize this: [paste long text]" | `summarize` | Returns bullet-point summary |
| "What is the difference between TCP and UDP?" | `chat` | Conversational answer |

---

## 🎁 Bonus Features Implemented

- **Human-in-the-Loop:** Before any file is created or written, a confirmation panel shows exactly what will happen. The user must click **Confirm & Execute** or **Cancel**.
- **Session Memory:** Chat history is maintained across turns within a session. The LLM receives the last 5 exchanges as context.
- **Graceful Degradation:** All API calls are wrapped in try/except. Errors surface as friendly UI messages, not crashes. Unintelligible audio returns a transcription of background noise which is safely handled as a `chat` intent.
- **Session Action Log:** Every action is timestamped and logged in a collapsible panel.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `groq` | Groq Python SDK (STT + LLM) |
| `python-dotenv` | `.env` file loading |
| `audio-recorder-streamlit` | In-browser microphone recording |

---

## 📬 Submission

Submitted via [https://forms.gle/5x32P7zr4NvyRgK6A](https://forms.gle/5x32P7zr4NvyRgK6A)
