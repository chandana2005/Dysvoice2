import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import ctranslate2

# Paths
base      = os.path.dirname(os.path.abspath(__file__))
pt_path   = os.path.join(base, "model", "dysvoice_whisper.pt")
hf_path   = os.path.join(base, "model", "dysvoice_hf_temp")
ct2_path  = os.path.join(base, "model", "dysvoice_ct2")

# Step 1: Load fine-tuned weights
print("Loading fine-tuned model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
state_dict = torch.load(pt_path, map_location="cpu")
model.load_state_dict(state_dict)
model.config.forced_decoder_ids = None

# Step 2: Save as HF format
print("Saving HF format...")
os.makedirs(hf_path, exist_ok=True)
model.save_pretrained(hf_path)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
processor.save_pretrained(hf_path)

# Step 3: Convert using ctranslate2 directly
print("Converting to CTranslate2...")
converter = ctranslate2.converters.TransformersConverter(hf_path, low_cpu_mem_usage=True)
converter.convert(ct2_path, quantization="int8", force=True)

print("Done! Model saved to:", ct2_path)