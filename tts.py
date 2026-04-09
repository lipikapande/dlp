import subprocess

def speak(text: str):
    """Speak text out loud using macOS native 'say' command. Blocking call."""
    print(f"[tts] Speaking: {text}")
    subprocess.run(["say", "-r", "160", text])