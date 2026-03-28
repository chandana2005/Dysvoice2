import torch
import os
import shutil
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ORIGINAL_MODEL  = "model/dysvoice_whisper.pt"
TEMP_HF_PATH    = "model/dysvoice_hf_temp"
OUTPUT_CT2_PATH = "model/dysvoice_ct2"
BASE_MODEL_NAME = "openai/whisper-small"

print("Step 1: Loading your fine-tuned weights...")
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
state_dict = torch.load(ORIGINAL_MODEL, map_location="cpu")
model.load_state_dict(state_dict)
model.config.forced_decoder_ids = None
print("Fine-tuned weights loaded successfully")

print("Step 2: Saving as HuggingFace format temporarily...")
os.makedirs(TEMP_HF_PATH, exist_ok=True)
model.save_pretrained(TEMP_HF_PATH)
processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)
processor.save_pretrained(TEMP_HF_PATH)
print(f"Saved to {TEMP_HF_PATH}")

print("Step 3: Converting to CTranslate2 format with int8 quantization...")
os.system(
    f"ct2-whisper-converter --model {TEMP_HF_PATH} "
    f"--output_dir {OUTPUT_CT2_PATH} "
    f"--quantization int8 "
    f"--force"
)

print("Step 4: Cleaning up temporary folder...")
shutil.rmtree(TEMP_HF_PATH)

print("")
print("Conversion complete!")
print(f"Your converted model is at: {OUTPUT_CT2_PATH}")