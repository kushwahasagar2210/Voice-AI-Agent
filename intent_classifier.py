"""
intent_classifier.py — Intent classification via Groq LLM
Parses transcribed text and returns structured intent data as JSON.
"""

import json
import re
from groq import Groq
import config


SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent.
Analyze the user's request and return ONLY a valid JSON object — no markdown, no extra text.

Supported intents:
- "create_file"  : User wants to create a new empty or simple text file/folder.
- "write_code"   : User wants to generate code and save it to a file.
- "summarize"    : User wants to summarize a piece of text they provide.
- "chat"         : General questions, conversation, or anything else.

Response JSON schema (all fields required):
{
  "primary_intent": "<one of the four intents above>",
  "intents": ["<list of all applicable intents, e.g. compound commands>"],
  "filename": "<suggested output filename including extension, or null>",
  "language": "<programming language if write_code, else null>",
  "content_request": "<the specific thing the user wants created/written/done>",
  "text_to_summarize": "<verbatim text to summarize if summarize intent, else null>",
  "explanation": "<one sentence explaining the detected intent>"
}

Rules:
- For "write_code", always infer a sensible filename from the request (e.g. retry.py, sort.js).
- For "create_file", use the name mentioned or invent one that makes sense.
- If the user wants multiple things (e.g. "generate code AND save it"), list all intents but pick the most actionable as primary_intent.
- Never return anything outside the JSON object.
"""


def classify_intent(text: str) -> dict:
    """
    Classify the intent of a transcribed user utterance.

    Args:
        text: Transcribed text from the STT module.

    Returns:
        Parsed intent dictionary.

    Raises:
        ValueError: If the LLM response cannot be parsed as JSON.
    """
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=config.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model added them despite instructions
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON response:\n{raw}\nError: {exc}"
        ) from exc

    # Normalise: ensure required keys exist with safe defaults
    data.setdefault("primary_intent", "chat")
    data.setdefault("intents", [data["primary_intent"]])
    data.setdefault("filename", None)
    data.setdefault("language", None)
    data.setdefault("content_request", text)
    data.setdefault("text_to_summarize", None)
    data.setdefault("explanation", "")

    return data
