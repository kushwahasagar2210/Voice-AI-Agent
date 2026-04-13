"""
tools.py — Tool execution layer
All file writes are sandboxed to the output/ directory.
"""

import os
import re
from groq import Groq
import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_path(filename: str) -> str:
    """Resolve a filename to the output/ sandbox. Prevents path traversal."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # Strip any directory components the LLM might have added
    safe_name = os.path.basename(filename)
    return os.path.join(config.OUTPUT_DIR, safe_name)


def _strip_code_fences(text: str) -> str:
    """Remove ```lang ... ``` markdown fences from generated code."""
    text = re.sub(r"^```[\w]*\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _groq_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    return Groq(api_key=config.GROQ_API_KEY)


# ── Tool: Create File ─────────────────────────────────────────────────────────

def create_file(filename: str, content: str = "") -> dict:
    """
    Create a file in the output/ sandbox.

    Returns:
        {success, filepath, message}
    """
    filepath = _safe_path(filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return {
            "success": True,
            "filepath": filepath,
            "message": f"File created at `{filepath}`.",
        }
    except OSError as exc:
        return {"success": False, "error": str(exc)}


# ── Tool: Generate & Save Code ────────────────────────────────────────────────

_LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "html": ".html",
    "css": ".css",
    "java": ".java",
    "c++": ".cpp",
    "cpp": ".cpp",
    "c": ".c",
    "go": ".go",
    "rust": ".rs",
    "ruby": ".rb",
    "php": ".php",
    "bash": ".sh",
    "shell": ".sh",
    "sql": ".sql",
}


def generate_code(
    request: str,
    language: str = "python",
    filename: str | None = None,
) -> dict:
    """
    Generate code with the LLM and save it to output/.

    Returns:
        {success, filepath, code, message}
    """
    lang_lower = (language or "python").lower()
    ext = _LANGUAGE_EXTENSIONS.get(lang_lower, ".txt")

    if not filename:
        filename = f"generated_code{ext}"
    elif not os.path.splitext(filename)[1]:
        filename = filename + ext

    client = _groq_client()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are an expert {language} developer. "
                    "Generate clean, well-commented, production-ready code. "
                    "Return ONLY the raw code — no markdown fences, no explanations."
                ),
            },
            {"role": "user", "content": request},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    code = _strip_code_fences(response.choices[0].message.content)
    filepath = _safe_path(filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        return {
            "success": True,
            "filepath": filepath,
            "code": code,
            "message": f"Code written to `{filepath}`.",
        }
    except OSError as exc:
        return {"success": False, "error": str(exc), "code": code}


# ── Tool: Summarize Text ──────────────────────────────────────────────────────

def summarize_text(text: str) -> dict:
    """
    Summarize provided text using the LLM.

    Returns:
        {success, summary}
    """
    client = _groq_client()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise summarizer. "
                    "Provide a clear, concise summary in 3–5 bullet points, "
                    "then a one-sentence TL;DR at the end."
                ),
            },
            {"role": "user", "content": f"Summarize the following:\n\n{text}"},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    summary = response.choices[0].message.content.strip()
    return {"success": True, "summary": summary}


# ── Tool: General Chat ────────────────────────────────────────────────────────

def chat(message: str, history: list | None = None) -> dict:
    """
    General-purpose chat with optional conversation history.

    Args:
        message: The latest user message.
        history: List of {"role": "user"|"assistant", "content": str} dicts.

    Returns:
        {success, reply}
    """
    client = _groq_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, concise AI assistant. "
                "Answer clearly and directly."
            ),
        }
    ]

    if history:
        messages.extend(history[-10:])  # keep last 5 exchanges to stay in context

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    reply = response.choices[0].message.content.strip()
    return {"success": True, "reply": reply}
