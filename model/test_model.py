import torch
import librosa
import config
from transformers import WhisperProcessor, WhisperForConditionalGeneration

try:
    # ── Load model ──────────────────────────────────────────────────────────
    processor = WhisperProcessor.from_pretrained(config.MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.to(config.DEVICE)
    model.eval()
    model.config.forced_decoder_ids = None

    print("✅ Model loaded successfully")

    # ── Load a TORGO .wav file ───────────────────────────────────────────────
    WAV_PATH = r"C:\Users\chand\Desktop\Dysvoice\F_dys\F01\Session1\wav_headMic\0001.wav"

    audio, _ = librosa.load(WAV_PATH, sr=config.SAMPLE_RATE)
    print("✅ Audio loaded successfully")

    # ── Transcribe ───────────────────────────────────────────────────────────
    inputs = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt", language="en")
    input_features = inputs.input_features.to(config.DEVICE)
    attention_mask = torch.ones_like(input_features)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features, attention_mask=attention_mask, language="en")

    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"🎙  Input file : {WAV_PATH}")
    print(f"📝 Transcript : {transcript}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()