import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
STT_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"
OUTPUT_DIR = "output"
