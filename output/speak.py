"""
speak.py - Text-to-Speech Output Module
DysVoice | Person 3 - Output & Integration

Converts transcribed text to spoken audio using pyttsx3.
Works offline on Windows (SAPI5), Mac (nsss), and Linux/Pi (espeak).
"""

import pyttsx3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _get_engine():
    """
    Initialise and configure the pyttsx3 TTS engine with settings from config.py.

    Returns:
        pyttsx3.Engine: A configured TTS engine instance.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', config.TTS_RATE)
    engine.setProperty('volume', config.TTS_VOLUME)
    return engine


def speak(text: str) -> None:
    """
    Convert a text string to spoken audio output.

    Speaks the given text aloud through the system's default audio output.
    Does nothing if the text is empty or whitespace only.

    Args:
        text (str): The transcript text to speak aloud.

    Returns:
        None
    """
    if not text or not text.strip():
        return

    engine = _get_engine()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def save_audio(text: str, output_path: str) -> str:
    """
    Save TTS output to a .wav file instead of playing it aloud.

    Useful for pre-generating demo audio clips or as a backup
    if live TTS fails on demo day.

    Args:
        text (str):        The text to convert to speech.
        output_path (str): File path where the .wav will be saved.
                           Example: 'output/demo_clip.wav'

    Returns:
        str: The path of the saved .wav file.
    """
    if not text or not text.strip():
        print("save_audio: empty text provided, nothing saved.")
        return output_path

    engine = _get_engine()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    print(f"Audio saved to: {output_path}")
    return output_path


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== speak.py test ===\n")

    print("Testing speak()...")
    speak("please bring me water")
    speak("turn off the lights")
    print("  speak() tests done.\n")

    print("Testing save_audio()...")
    path = save_audio("This is a saved audio clip for the demo.", "test_output.wav")
    if os.path.exists(path):
        print(f"  ✓ File saved: {path} ({os.path.getsize(path)} bytes)")
    else:
        print("  ✗ File not created — known issue on some macOS versions.")

    print("\n=== speak.py tests complete ===")