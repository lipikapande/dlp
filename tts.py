# import subprocess

# def speak(text: str):
#     """Speak text out loud using macOS native 'say' command. Blocking call."""
#     print(f"[tts] Speaking: {text}")
#     subprocess.run(["say", "-r", "160", text])

## WINDOWS

import pyttsx3

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 160)
        _engine.setProperty("volume", 0.9)
    return _engine

def speak(text: str):
    print(f"[tts] Speaking: {text}")
    try:
        engine = get_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[tts] Error: {e} — printing instead")