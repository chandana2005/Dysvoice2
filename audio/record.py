"""
audio/record.py
===============
DysVoice Audio Pipeline (Person 2)

Records audio from the microphone using Voice Activity Detection (VAD).

How VAD works here:
  1. Continuously reads short chunks (frames) from the mic.
  2. Waits in WAITING state until a loud-enough chunk is detected
     (RMS energy above SILENCE_THRESHOLD).
  3. Once speech starts, switches to RECORDING state and collects all
     chunks.
  4. If 1.5 seconds of consecutive silence is detected while recording,
     stops and returns the collected audio as a numpy array at 16 kHz.

Public API (called by Person 3's main.py):
    audio = record_audio()   # returns np.ndarray, dtype=float32, 16 kHz
"""

import numpy as np
import pyaudio
import sys

# ── Config ────────────────────────────────────────────────────────────────────
# These match config.py values; imported directly here so this file works
# standalone during testing without needing the full project on sys.path.
try:
    import config
    SAMPLE_RATE = config.SAMPLE_RATE          # 16000 Hz
    MAX_DURATION = config.MAX_DURATION_SECONDS  # 10 s hard cap
except ModuleNotFoundError:
    SAMPLE_RATE = 16000
    MAX_DURATION = 10

CHANNELS      = 1           # mono
CHUNK         = 512         # frames per buffer (~32 ms at 16 kHz)
FORMAT        = pyaudio.paInt16

# ── VAD tuning ────────────────────────────────────────────────────────────────
# Raise SILENCE_THRESHOLD if the mic picks up too much background noise.
# Lower it if speech is not being detected.
SILENCE_THRESHOLD  = 50   # RMS amplitude below this = silence (0–32767 scale)
SILENCE_DURATION   = 3.0   # seconds of silence before recording stops
SILENCE_CHUNKS     = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK)  # ≈ 47 chunks

# Minimum speech length to be returned (avoids returning a tiny noise burst)
MIN_SPEECH_CHUNKS  = int(0.3 * SAMPLE_RATE / CHUNK)               # ≈ 9 chunks


def _rms(chunk_bytes: bytes) -> float:
    """Return the root-mean-square energy of a raw PCM int16 chunk."""
    samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def record_audio() -> np.ndarray:
    """
    Record from the default microphone using VAD and return the audio.

    Behaviour
    ---------
    - Prints "Listening..." and waits silently until speech is detected.
    - Prints "Recording..." once speech starts.
    - Stops automatically after SILENCE_DURATION seconds of silence.
    - Hard-caps at MAX_DURATION seconds to prevent runaway recordings.
    - Prints "Done." when finished.

    Returns
    -------
    np.ndarray
        1-D float32 array of audio samples at SAMPLE_RATE (16 000 Hz).
        The array is normalised to the range [-1.0, 1.0].
        Returns an empty array (shape (0,)) if no speech was captured.
    """
    pa = pyaudio.PyAudio()

    
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("Listening...", flush=True)

    collected_chunks: list[bytes] = []
    silence_counter  = 0
    recording        = False
    max_chunks       = int(MAX_DURATION * SAMPLE_RATE / CHUNK)

    try:
        for _ in range(max_chunks):
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            energy = _rms(chunk)

            if not recording:
                # ── WAITING state: listen for speech onset ────────────────
                if energy > SILENCE_THRESHOLD:
                    recording = True
                    silence_counter = 0
                    print("Recording...", flush=True)
                    collected_chunks.append(chunk)
            else:
                # ── RECORDING state: collect until silence ────────────────
                collected_chunks.append(chunk)
                if energy <= SILENCE_THRESHOLD:
                    silence_counter += 1
                    if silence_counter >= SILENCE_CHUNKS:
                        break          # enough silence → stop
                else:
                    silence_counter = 0  # reset on any speech

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if len(collected_chunks) < MIN_SPEECH_CHUNKS:
        print("No speech detected.", flush=True)
        return np.zeros(0, dtype=np.float32)

    print("Done.", flush=True)

    # Concatenate raw PCM bytes → int16 numpy array → normalised float32
    raw = b"".join(collected_chunks)
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0   # normalise to [-1, 1]
    return audio_float


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("record.py  –  VAD microphone test")
    print("Speak into your mic. Recording stops after 1.5 s of silence.")
    print("=" * 50)

    audio = record_audio()

    if audio.size == 0:
        print("Nothing recorded. Check your microphone and SILENCE_THRESHOLD.")
        sys.exit(1)

    print(f"\nCaptured {audio.size} samples  "
          f"({audio.size / SAMPLE_RATE:.2f} s at {SAMPLE_RATE} Hz)")
    print(f"dtype  : {audio.dtype}")
    print(f"min    : {audio.min():.4f}")
    print(f"max    : {audio.max():.4f}")
    print(f"shape  : {audio.shape}")
    print("\nTest PASSED — numpy audio array returned successfully.")

    # Optionally save to disk so you can listen back
    try:
        import soundfile as sf
        out_path = "test_recording.wav"
        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"Saved to {out_path}  (play it to verify quality)")
    except ImportError:
        print("(Install soundfile to also save the recording as a .wav)")