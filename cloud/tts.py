# This runs on the SERVER side if you want audio bytes back.
# For laptop testing, we use edge-side TTS instead (simpler).
# This file is a placeholder for future cloud TTS (e.g. gTTS).

def text_to_speech_bytes(text: str) -> bytes | None:
    """
    Optional: return MP3 bytes from gTTS for client to play.
    Requires: pip install gTTS
    """
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"[tts] Failed: {e}")
        return None