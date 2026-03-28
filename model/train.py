import os
import re
import config
import librosa
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

TORGO_PATH = config.TORGO_PATH
SAMPLE_RATE = config.SAMPLE_RATE

def load_torgo_data(base_path):
    data = []
    
    for dys_folder in ["F_dys", "M_dys"]:
        dys_path = os.path.join(base_path, dys_folder)
        print(f"Checking: {dys_path}")
        
        if not os.path.exists(dys_path):
            print(f"  NOT FOUND - skipping")
            continue
        
        print(f"  Found! Scanning speakers...")
            
        for speaker in os.listdir(dys_path):
            speaker_path = os.path.join(dys_path, speaker)
            if not os.path.isdir(speaker_path):
                continue
            print(f"    Speaker: {speaker}")
                
            for session in os.listdir(speaker_path):
                session_path = os.path.join(speaker_path, session)
                wav_path = os.path.join(session_path, "wav_headMic")
                prompt_path = os.path.join(session_path, "prompts")
                
                if not os.path.exists(wav_path) or not os.path.exists(prompt_path):
                    print(f"      Session {session}: missing wav or prompts - skipping")
                    continue
                
                print(f"      Session {session}: found wav + prompts")
                
                for txt_file in os.listdir(prompt_path):
                    if not txt_file.endswith(".txt"):
                        continue
                    
                    txt_full = os.path.join(prompt_path, txt_file)
                    with open(txt_full, "r") as f:
                        transcript = f.read().strip()
                    
                    if transcript.startswith("["):
                        continue
                    import re
                    transcript = re.sub(r'\[.*?\]', '', transcript).strip()
                    
                    if not transcript:
                        continue
                    
                    wav_file = txt_file.replace(".txt", ".wav")
                    wav_full = os.path.join(wav_path, wav_file)
                    
                    if not os.path.exists(wav_full):
                        continue
                    
                    data.append((wav_full, transcript))
    
    print(f"\nTotal samples loaded: {len(data)}")
    return data
def preprocess_audio(wav_path, processor):
    # Load the audio file
    audio, sr = librosa.load(wav_path, sr=16000)
    
    # Extract features using Whisper's own processor
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    return inputs.input_features


def train_model(data, output_path="model/dysvoice_whisper.pt"):
    print("Loading Whisper model and processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    print(f"Starting training on {len(data)} samples...")
    model.train()
    
    for epoch in range(10):
        total_loss = 0
        count = 0
        
        for wav_path, transcript in data:
            try:
                # Load and preprocess audio
                audio, _ = librosa.load(wav_path, sr=16000)
                inputs = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )
                input_features = inputs.input_features.to(device)
                
                # Tokenize transcript
                labels = processor.tokenizer(
                    transcript,
                    return_tensors="pt"
                ).input_ids.to(device)
                
                # Forward pass
                outputs = model(
                    input_features=input_features,
                    labels=labels
                )
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
                
                if count % 100 == 0:
                    print(f"Epoch {epoch+1}, Sample {count}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                continue
        
        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), output_path)
            print(f"Checkpoint saved to {output_path}")
    
    # Save final model
    torch.save(model.state_dict(), output_path)
    print(f"Training complete! Model saved to {output_path}")
    return model


if __name__ == "__main__":
    print("Starting TORGO data loading...\n")
    data = load_torgo_data(TORGO_PATH)
    for wav, text in data[:5]:
        print(f"Audio: {wav}")
        print(f"Transcript: {text}")
        print("---")