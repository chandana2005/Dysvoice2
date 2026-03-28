"""
audio/denoise.py
================
DysVoice – Audio Pipeline (Person 2)

Takes the raw numpy audio array from record.py, removes background noise,
and normalises the volume so quiet speakers are amplified consistently.

Public API (called by Person 3's main.py):
    clean_audio = denoise_audio(audio_array)
"""

import numpy as np
import noisereduce as nr

try:
    import config
    SAMPLE_RATE = config.SAMPLE_RATE
except ModuleNotFoundError:
    SAMPLE_RATE = 16000


def denoise_audio(audio: np.ndarray) -> np.ndarray:
    """
    Remove background noise and normalise volume from a raw audio array.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 array of audio samples at 16000 Hz (from record_audio()).

    Returns
    -------
    np.ndarray
        Cleaned and normalised float32 audio array, same sample rate.
    """
    if audio.size == 0:
        return audio

    # Step 1: Noise reduction
    # noisereduce estimates the noise profile from the audio itself
    reduced = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=1.0, stationary=True)

    # Step 2: Normalise amplitude to [-1.0, 1.0]
    # This ensures quiet speakers are amplified and loud speakers don't clip
    max_val = np.max(np.abs(reduced))
    if max_val > 0:
        reduced = reduced / max_val

    return reduced.astype(np.float32)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 50)
    print("denoise.py  –  Noise reduction test")
    print("=" * 50)

    # Test using a TORGO .wav file if provided, otherwise use mic
    if len(sys.argv) > 1:
        import librosa
        wav_path = sys.argv[1]
        print(f"Loading: {wav_path}")
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        print(f"Loaded {audio.size} samples ({audio.size / SAMPLE_RATE:.2f}s)")
    else:
        from audio.record import record_audio
        print("No file given — recording from mic instead...")
        audio = record_audio()

    if audio.size == 0:
        print("No audio to process.")
        sys.exit(1)

    print(f"\nBefore denoising — max amplitude: {np.max(np.abs(audio)):.4f}")
    clean = denoise_audio(audio)
    print(f"After denoising  — max amplitude: {np.max(np.abs(clean)):.4f}")
    print(f"Output shape: {clean.shape}, dtype: {clean.dtype}")
    print("\nTest PASSED — denoised audio array returned successfully.")

    # Save both versions so you can compare them
    try:
        import soundfile as sf
        sf.write("test_raw.wav", audio, SAMPLE_RATE)
        sf.write("test_clean.wav", clean, SAMPLE_RATE)
        print("Saved test_raw.wav and test_clean.wav — compare them to hear the difference.")
    except ImportError:
        print("(Install soundfile to save output as .wav)")