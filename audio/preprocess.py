import numpy as np
import librosa

TARGET_SR = 16000

def preprocess_audio(audio: np.ndarray, sr: int):
    """
    Convert audio to 16kHz float32 mono for Whisper
    """

    # Ensure float32
    audio = audio.astype(np.float32)

    # Normalize (avoid clipping / low volume)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Resample if needed
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return audio, TARGET_SR
