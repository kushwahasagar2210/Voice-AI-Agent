"""
app.py — Voice AI Agent — Streamlit Frontend
Pipeline: Audio Input → STT → Intent Classification → (HITL confirm) → Tool Execution → Display
"""

import os
import tempfile
from datetime import datetime

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Voice AI Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .step-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; }
    .intent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        background: #1f77b4;
        color: white;
    }
    .log-entry { font-family: monospace; font-size: 0.85rem; color: #aaa; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=64
    )
    st.title("Voice AI Agent")
    st.caption("Powered by Groq + Streamlit")

    st.divider()
    st.subheader("🔑 API Key")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.divider()
    st.subheader("🤖 Models")
    stt_model = st.selectbox(
        "STT Model",
        ["whisper-large-v3", "whisper-large-v3-turbo"],
        help="whisper-large-v3-turbo is faster, large-v3 is more accurate",
    )
    llm_model = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        help="70b is most capable; 8b is faster for simple tasks",
    )

    st.divider()
    st.subheader("📁 Output Sandbox")
    st.code("./output/", language=None)
    st.caption("All file operations are restricted to this folder.")

    st.divider()
    if st.button("🗑️ Clear Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.divider()
    st.caption("Built for Mem0 AI/ML Internship Assignment")


# ── Load modules (after env might be set) ────────────────────────────────────
import config

config.STT_MODEL = stt_model
config.LLM_MODEL = llm_model

from stt import transcribe_audio
from intent_classifier import classify_intent
import tools


# ── Session State Init ────────────────────────────────────────────────────────
_DEFAULTS = {
    "transcription": None,
    "intent_data": None,
    "pending_action": False,
    "action_result": None,
    "chat_history": [],
    "action_log": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helper: intent display ────────────────────────────────────────────────────
INTENT_META = {
    "create_file": ("📄", "Create File",  "#27ae60"),
    "write_code":  ("💻", "Write Code",   "#2980b9"),
    "summarize":   ("📋", "Summarize",    "#8e44ad"),
    "chat":        ("💬", "Chat",         "#e67e22"),
}

def intent_badge(intent: str) -> str:
    icon, label, color = INTENT_META.get(intent, ("❓", intent, "#7f8c8d"))
    return (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:20px;font-weight:600;font-size:0.9rem;">'
        f"{icon} {label}</span>"
    )


# ── Main Title ────────────────────────────────────────────────────────────────
st.title("🎙️ Voice AI Agent")
st.caption(
    "**Pipeline:** Audio Input → Speech-to-Text → Intent Classification "
    "→ _(confirm)_ → Tool Execution → Result"
)

# ── STEP 1 — Audio Input ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Step 1 — Provide Audio")

tab_record, tab_upload = st.tabs(["🎤 Record Microphone", "📂 Upload Audio File"])

audio_bytes: bytes | None = None
audio_filename: str = "audio.wav"

with tab_record:
    st.info(
        "Click the microphone button below, speak your command, then click again to stop."
    )
    try:
        from audio_recorder_streamlit import audio_recorder

        recorded_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=2.5,
            sample_rate=16_000,
        )
        if recorded_bytes:
            audio_bytes = recorded_bytes
            audio_filename = "mic_recording.wav"
            st.audio(audio_bytes, format="audio/wav")
            st.success("✅ Recording captured — click **Process Audio** below.")
    except ImportError:
        st.warning(
            "🔧 `audio_recorder_streamlit` not installed. "
            "Run `pip install audio_recorder_streamlit` for microphone support, "
            "or use the file upload tab."
        )

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drop an audio file here",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC, WEBM",
    )
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        audio_filename = uploaded_file.name
        fmt = uploaded_file.name.rsplit(".", 1)[-1].lower()
        st.audio(audio_bytes, format=f"audio/{fmt}")
        st.success(f"✅ File loaded: `{uploaded_file.name}`")


# ── Process Button ────────────────────────────────────────────────────────────
st.markdown("---")
col_btn, col_warn = st.columns([1, 3])

api_key_present = bool(os.environ.get("GROQ_API_KEY") or config.GROQ_API_KEY)

with col_btn:
    process_clicked = st.button(
        "🚀 Process Audio",
        disabled=(audio_bytes is None or not api_key_present),
        type="primary",
        use_container_width=True,
    )

with col_warn:
    if not api_key_present:
        st.warning("⚠️ Please enter your **Groq API Key** in the sidebar to continue.")
    elif audio_bytes is None:
        st.info("☝️ Record or upload audio first, then click **Process Audio**.")


# ── Pipeline execution on button click ───────────────────────────────────────
if process_clicked and audio_bytes:
    # Reset state for fresh run
    for k in ("transcription", "intent_data", "pending_action", "action_result"):
        st.session_state[k] = _DEFAULTS[k]

    # -- STT ------------------------------------------------------------------
    with st.spinner("🎧 Transcribing audio with Whisper…"):
        try:
            st.session_state.transcription = transcribe_audio(audio_bytes, audio_filename)
        except Exception as exc:
            st.error(f"**STT Error:** {exc}")
            st.stop()

    # -- Intent ---------------------------------------------------------------
    with st.spinner("🧠 Classifying intent…"):
        try:
            st.session_state.intent_data = classify_intent(st.session_state.transcription)
        except Exception as exc:
            st.error(f"**Intent Classifier Error:** {exc}")
            st.stop()

    # -- Decide HITL vs immediate execution -----------------------------------
    primary = st.session_state.intent_data.get("primary_intent", "chat")
    FILE_INTENTS = {"create_file", "write_code"}

    if primary in FILE_INTENTS:
        # Pause and show confirmation UI
        st.session_state.pending_action = True
    else:
        # Execute immediately for safe (read-only / generation) intents
        with st.spinner("⚙️ Executing…"):
            try:
                if primary == "summarize":
                    text = (
                        st.session_state.intent_data.get("text_to_summarize")
                        or st.session_state.transcription
                    )
                    result = tools.summarize_text(text)
                else:  # chat / unknown
                    result = tools.chat(
                        st.session_state.transcription,
                        st.session_state.chat_history,
                    )
                    if result["success"]:
                        st.session_state.chat_history.append(
                            {"role": "user", "content": st.session_state.transcription}
                        )
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": result["reply"]}
                        )
                st.session_state.action_result = result
                st.session_state.action_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "intent": primary,
                    "detail": st.session_state.intent_data.get("explanation", ""),
                    "success": result.get("success", False),
                })
            except Exception as exc:
                st.session_state.action_result = {"success": False, "error": str(exc)}

    st.rerun()


# ── Display pipeline results ──────────────────────────────────────────────────
if st.session_state.transcription:

    # STEP 2 — Transcription
    st.markdown("---")
    st.markdown("### Step 2 — Transcription")
    st.info(f"🗣️ **You said:** {st.session_state.transcription}")

    # STEP 3 — Intent
    if st.session_state.intent_data:
        st.markdown("---")
        st.markdown("### Step 3 — Detected Intent")

        idata = st.session_state.intent_data
        primary = idata.get("primary_intent", "chat")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Primary Intent**")
            st.markdown(intent_badge(primary), unsafe_allow_html=True)

        with col_b:
            if idata.get("filename"):
                st.markdown("**Target File**")
                st.code(idata["filename"], language=None)
            if idata.get("language"):
                st.markdown("**Language**")
                st.code(idata["language"], language=None)

        with col_c:
            all_intents = idata.get("intents", [primary])
            if len(all_intents) > 1:
                st.markdown("**Compound Intents**")
                for i in all_intents:
                    st.markdown(intent_badge(i), unsafe_allow_html=True)
                    st.write("")

        st.caption(f"💡 {idata.get('explanation', '')}")

    # STEP 4 — Human-in-the-Loop Confirmation
    if st.session_state.pending_action and not st.session_state.action_result:
        st.markdown("---")
        st.markdown("### Step 4 — Confirm Action")

        idata = st.session_state.intent_data
        primary = idata.get("primary_intent", "")

        with st.container(border=True):
            st.warning("⚠️ **This action will write files to your local filesystem.**")
            st.markdown("**Proposed Operation:**")

            details = {
                "Operation": primary.replace("_", " ").title(),
                "Output file": idata.get("filename") or "_(auto-generated)_",
                "Folder": "`output/`",
            }
            if idata.get("language"):
                details["Language"] = idata["language"]
            if idata.get("content_request"):
                details["Request"] = idata["content_request"]

            for label, value in details.items():
                st.markdown(f"- **{label}:** {value}")

            col_yes, col_no, _ = st.columns([1, 1, 3])

            with col_yes:
                if st.button("✅ Confirm & Execute", type="primary", use_container_width=True):
                    with st.spinner("⚙️ Executing…"):
                        try:
                            if primary == "write_code":
                                result = tools.generate_code(
                                    request=idata.get("content_request") or st.session_state.transcription,
                                    language=idata.get("language") or "python",
                                    filename=idata.get("filename"),
                                )
                            elif primary == "create_file":
                                result = tools.create_file(
                                    filename=idata.get("filename") or "new_file.txt",
                                    content=idata.get("content_request") or "",
                                )
                            else:
                                result = {"success": False, "error": "Unrecognised file intent."}
                        except Exception as exc:
                            result = {"success": False, "error": str(exc)}

                    st.session_state.action_result = result
                    st.session_state.pending_action = False
                    st.session_state.action_log.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "intent": primary,
                        "detail": idata.get("filename") or "",
                        "success": result.get("success", False),
                    })
                    st.rerun()

            with col_no:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.pending_action = False
                    st.session_state.action_result = {
                        "success": False,
                        "error": "Action cancelled by user.",
                    }
                    st.rerun()

    # STEP 5 — Result
    if st.session_state.action_result:
        st.markdown("---")
        st.markdown("### Step 5 — Result")

        result = st.session_state.action_result

        if result.get("success"):
            st.success("✅ **Done!**")

            if "code" in result:
                lang = (
                    st.session_state.intent_data.get("language", "python")
                    if st.session_state.intent_data
                    else "python"
                )
                st.markdown(f"**Generated Code** — saved to `{result['filepath']}`")
                st.code(result["code"], language=lang)

            elif "summary" in result:
                st.markdown("**Summary**")
                st.markdown(result["summary"])

            elif "reply" in result:
                st.markdown("**Response**")
                st.markdown(result["reply"])

            else:
                st.markdown(result.get("message", "Action completed."))

        else:
            st.error(f"❌ **Error:** {result.get('error', 'Unknown error.')}")


# ── Session History ───────────────────────────────────────────────────────────
if st.session_state.action_log:
    st.markdown("---")
    with st.expander("📜 Session Action Log", expanded=False):
        for entry in reversed(st.session_state.action_log):
            icon = "✅" if entry["success"] else "❌"
            st.markdown(
                f'<span class="log-entry">[{entry["time"]}] {icon} '
                f'**{entry["intent"].replace("_"," ").title()}** — {entry["detail"]}</span>',
                unsafe_allow_html=True,
            )

if st.session_state.chat_history:
    st.markdown("---")
    with st.expander("💬 Conversation History", expanded=False):
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
