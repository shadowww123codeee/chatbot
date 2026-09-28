"""Gemini + voice helpers for the English Learning Chatbot (Streamlit)."""
import io
import os
import re

import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS

# Change this if Google retires the model; any current free-tier Flash model works.
MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a friendly English-learning buddy for school students in rural India. "
    "Use simple words and short sentences. When it helps, add a Hindi translation. "
    "Gently correct grammar mistakes. Keep answers under 120 words."
)


def get_api_key():
    """Key priority: sidebar box -> Streamlit secrets -> environment variable."""
    key = st.session_state.get("gemini_key_input", "").strip()
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


def _client():
    key = get_api_key()
    return genai.Client(api_key=key) if key else None


def ask_gemini(user_text, history=None):
    """Send the question (plus a little chat history) to Gemini."""
    client = _client()
    if client is None:
        return "⚠️ Please add your Gemini API key in the sidebar."

    contents = []
    for m in (history or [])[-8:]:
        role = "user" if m["role"] == "user" else "model"
        if not contents and role == "model":
            continue  # history should start with a user turn
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return resp.text or "Sorry, I could not answer that."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"


def transcribe(audio_bytes, mime_type="audio/wav"):
    """Speech -> text using Gemini's native audio input."""
    client = _client()
    if client is None:
        return ""
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                "Transcribe this audio exactly. Reply with only the transcription.",
            ],
        )
        return (resp.text or "").strip()
    except Exception:
        return ""


def speak(text, lang="en"):
    """Text -> MP3 bytes using gTTS (free, no key needed)."""
    clean = re.sub(r"[*_`#>]", "", text)[:600]
    buf = io.BytesIO()
    gTTS(clean, lang=lang).write_to_fp(buf)
    return buf.getvalue()
