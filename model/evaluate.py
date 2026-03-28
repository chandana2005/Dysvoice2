import sys
sys.path.append(r"C:\Users\chand\Desktop\Dysvoice")
import os
import re
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
import config

def load_model():
    print("Loading trained model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    model.config.forced_decoder_ids = None
    processor.tokenizer.forced_decoder_ids = None
    return model, processor

def transcribe_audio(wav_path, model, processor):
    audio, _ = librosa.load(wav_path, sr=config.SAMPLE_RATE)
    inputs = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        predicted_ids  = model.generate(
    inputs.input_features,
    language="en",
    attention_mask=torch.ones(inputs.input_features.shape[:2], dtype=torch.long)
)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript.strip()

def evaluate_speaker(speaker_path, model, processor):
    results = []
    wav_path = os.path.join(speaker_path, "wav_headMic")
    prompt_path = os.path.join(speaker_path, "prompts")
    
    if not os.path.exists(wav_path) or not os.path.exists(prompt_path):
        print(f"Missing wav or prompts in {speaker_path}")
        return results
    
    for txt_file in os.listdir(prompt_path):
        if not txt_file.endswith(".txt"):
            continue
        
        txt_full = os.path.join(prompt_path, txt_file)
        with open(txt_full, "r") as f:
            transcript = f.read().strip()
        
        if transcript.startswith("["):
            continue
        transcript = re.sub(r'\[.*?\]', '', transcript).strip()
        if not transcript:
            continue
        
        wav_file = txt_file.replace(".txt", ".wav")
        wav_full = os.path.join(wav_path, wav_file)
        if not os.path.exists(wav_full):
            continue
        
        try:
            predicted = transcribe_audio(wav_full, model, processor)
            results.append((transcript, predicted))
            if len(results) >= 50:
                break
        except:
            continue
    

    return results

def calculate_accuracy(results):
    references = [r[0].lower() for r in results]
    hypotheses = [r[1].lower() for r in results]
    error_rate = wer(references, hypotheses)
    accuracy = (1 - error_rate) * 100
    return accuracy, error_rate * 100

if __name__ == "__main__":
    model, processor = load_model()
    
    test_speakers = {
        "M04": "C:\\Users\\chand\\Desktop\\Dysvoice\\M_dys\\M04\\Session2",
        "F03": "C:\\Users\\chand\\Desktop\\Dysvoice\\F_dys\\F03\\Session1"
    }
    
    for speaker, path in test_speakers.items():
        print(f"\nEvaluating {speaker}...")
        results = evaluate_speaker(path, model, processor)
        if results:
            accuracy, error_rate = calculate_accuracy(results)
            print(f"Speaker {speaker}:")
            print(f"  WRA: {accuracy:.2f}%")
            print(f"  WER: {error_rate:.2f}%")
            print(f"  Samples tested: {len(results)}")
            print(f"\nExample:")
            print(f"  Actual:    {results[0][0]}")
            print(f"  Predicted: {results[0][1]}")