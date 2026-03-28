import os
import numpy as np
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

try:
    import config
    SAMPLE_RATE    = config.SAMPLE_RATE
    CT2_MODEL_PATH = config.CT2_MODEL_PATH
    DEVICE         = config.DEVICE
    COMPUTE_TYPE   = config.COMPUTE_TYPE
except ModuleNotFoundError:
    SAMPLE_RATE    = 16000
    CT2_MODEL_PATH = "model/dysvoice_ct2"
    DEVICE         = "cpu"
    COMPUTE_TYPE   = "int8"

from faster_whisper import WhisperModel

print(f"[transcribe] Loading faster-whisper model from {CT2_MODEL_PATH}")
_model = WhisperModel(CT2_MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"[transcribe] Model loaded successfully")


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a numpy audio array to text using faster-whisper.
    Parameters
    ----------
    audio : np.ndarray
        1-D float32 array of audio samples at 16000 Hz.
    Returns
    -------
    str
        Transcribed text e.g. 'bring me water'.
        Returns empty string if audio is empty or transcription fails.
    """
    if audio.size == 0:
        return ""
    try:
        segments, _ = _model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=5,
        )
        text = " ".join(segment.text for segment in segments).strip()
        return text
    except Exception as e:
        print(f"[transcribe] Error: {e}")
        return ""


if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("transcribe.py  -  faster-whisper transcription test")
    print("=" * 50)
    if len(sys.argv) > 1:
        import librosa
        wav_path = sys.argv[1]
        print(f"Loading file: {wav_path}")
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        print(f"Loaded {audio.size} samples ({audio.size / SAMPLE_RATE:.2f}s)")
    else:
        from audio.record import record_audio
        from audio.denoise import denoise_audio
        print("No file given - recording from mic...")
        audio = record_audio()
        audio = denoise_audio(audio)

    if audio.size == 0:
        print("No audio to transcribe.")
        sys.exit(1)

    print("Transcribing...")
    result = transcribe(audio)
    print(f"\nTranscript: '{result}'")
    print("\nTest PASSED - transcribe() returned a string successfully.")
