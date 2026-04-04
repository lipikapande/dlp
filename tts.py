import pyttsx3

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 160)   # Words per minute
        _engine.setProperty("volume", 0.9)
    return _engine

def speak(text: str):
    """Speak text out loud using system TTS. Blocking call."""
    print(f"[tts] Speaking: {text}")
    engine = get_engine()
    engine.say(text)
    engine.runAndWait()