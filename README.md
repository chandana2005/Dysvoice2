# DysVoice
### AI-Based Dysarthric Speech Assistance System
 
DysVoice is an AI system that listens to dysarthric speech (speech from people with conditions like cerebral palsy that affect how they talk) and converts it into clear text and spoken output. Think of it like a translator between dysarthric speech and normal communication.
 
---
 
## Project Structure
 
```
Dysvoice/
├── config.py                  ← Shared settings used by all 3 developers
├── requirements.txt           ← All libraries the project needs
├── main.py                    ← Master file that connects everything together
├── model/
│   ├── train.py               ← Code to train the AI model
│   ├── evaluate.py            ← Code to test how accurate the model is
│   ├── test_model.py          ← Script to verify the model loads and transcribes correctly
│   └── dysvoice_whisper.pt    ← Trained model file (download from Google Drive — see below)
├── audio/
│   ├── record.py              ← Code to record from microphone
│   └── denoise.py             ← Code to clean up background noise
├── inference/
│   └── transcribe.py          ← Code to convert speech to text
├── output/
│   ├── speak.py               ← Code to convert text to speech
│   └── display.py             ← Code to show text on screen
└── hardware/
    └── setup.sh               ← Setup script for Raspberry Pi
```
 
---
 
## Model Download
 
The trained model file `dysvoice_whisper.pt` is 967MB — too large for GitHub. Download it from Google Drive:
 
🔗 [Download dysvoice_whisper.pt](https://drive.google.com/drive/folders/1rtKe_JsFFLvp0zqZ8ohyRvxXvRcqMYD2?usp=sharing)
 
Place it in the `model/` folder before running any code:
 
```
Dysvoice/
└── model/
    └── dysvoice_whisper.pt  ← place here
```
 
---
 
## Accuracy Results
 
Model evaluated on TORGO speakers not seen during training:
 
| Speaker | Severity | WRA | WER | Samples Tested |
|---------|----------|-----|-----|----------------|
| M04 | Mild/Moderate | 96.30% | 3.70% | 10 |
| F03 | Moderate | 100.00% | 0.00% | 10 |
 
**Target was 85% — model achieved 96–100%** ✅
 
---
 
## Setup
 
```bash
git clone <repo_url>
cd Dysvoice
pip install -r requirements.txt
```
 
Then download `dysvoice_whisper.pt` from the link above and place it in `model/`.
 
---
 
## Run
 
```bash
# Live microphone mode
python main.py
 
# Demo backup mode (file input)
python main.py --file path/to/audio.wav
 
# Verify model is working
python -m model.test_model
```
 
---
 
## Key Libraries
 
| Library | Purpose |
|---------|---------|
| `torch` | Deep learning engine |
| `transformers` | Loads and runs the Whisper model |
| `librosa` | Loads audio files and resamples to 16kHz |
| `noisereduce` | Removes background noise from audio |
| `pyaudio` | Microphone access and recording |
| `pyttsx3` | Text-to-speech output (works offline) |
| `jiwer` | Calculates Word Error Rate (WER) |
 
---
 
## Developer Log
 
---
 
# Developer 1 — AI & Model Training
 
## Day 1
 
### Goal
Set up the shared GitHub repository, create the project folder structure, and write the shared configuration files so all three developers can start working immediately from Day 1 with no blockers.
 
### What is this project?
DysVoice is an AI system that listens to dysarthric speech and converts it into clear text and spoken output. Think of it like a translator between dysarthric speech and normal communication.
 
### Steps Completed
 
- **Step 1:** Created the GitHub repository and shared the link with Developers 2 and 3
- **Step 2:** Created the project folder structure with placeholder files so each developer knows exactly where their files go:
  1. `model/` — Developer 1's folder, contains all AI training code
  2. `audio/` — Developer 2's folder, contains microphone recording and noise cleaning code
  3. `inference/` — Developer 2's folder, contains the code that runs speech through the AI model
  4. `output/` — Developer 3's folder, contains text-to-speech and display code
  5. `hardware/` — Developer 3's folder, contains setup scripts for the Raspberry Pi
- **Step 3:** Wrote `config.py` with shared settings used by all 3 developers:
  1. `MODEL_NAME` — which AI model we are using (Whisper Small from OpenAI)
  2. `SAMPLE_RATE` — audio quality setting, 16000 means 16000 audio samples per second
  3. `MODEL_PATH` — where the trained model file will be saved
  4. `DEVICE` — whether to use CPU (normal laptop) or CUDA (GPU for faster training)
  5. `MAX_DURATION_SECONDS` — maximum length of audio the system will process
  6. `TTS_RATE` — how fast the text-to-speech voice speaks
  7. `TTS_VOLUME` — how loud the text-to-speech voice is
- **Step 4:** Wrote `requirements.txt` listing every library the project needs. When teammates clone the repo they run `pip install -r requirements.txt` and Python automatically installs everything:
  1. `torch` — deep learning engine
  2. `transformers` — loads the Whisper model from HuggingFace
  3. `datasets` — helps load and organise audio data
  4. `librosa` — loads audio files and converts sample rates
  5. `soundfile` — reads and writes audio files
  6. `noisereduce` — removes background noise from audio
  7. `pyaudio` — accesses the microphone
  8. `pyttsx3` — converts text to speech
  9. `evaluate` — calculates how accurate the model is
  10. `jiwer` — calculates Word Error Rate
- **Step 5:** Created `__init__.py` files in each folder so Python recognises them as importable packages
- **Step 6:** Pushed everything to GitHub and added teammates as collaborators
 
---
 
## Day 2
 
### Goal
Write the data loading function — the code that reads all the audio files and their matching transcripts from the TORGO dataset and prepares them as clean pairs ready for AI training.
 
### What is the TORGO Dataset?
TORGO is a research dataset created by the University of Toronto. It contains recordings of real people with dysarthria speaking words and sentences into a microphone. Each audio file has a matching text file that says exactly what the person said. This is called a labelled dataset — the AI needs both the audio AND the correct text to learn from.
 
### Steps Completed
 
- **Step 1:** Downloaded the TORGO dataset from the official University of Toronto website:
  - `F_dys.bz2` — female dysarthric speakers
  - `M_dys.bz2` — male dysarthric speakers
  - Skipped `F_con` and `M_con` — these are control speakers (normal speech), not needed for training
  - `.bz2` is a compressed format like `.zip` — it makes large files smaller for downloading
- **Step 2:** Extracted both files to create `F_dys/` and `M_dys/` folders inside the project. Windows cannot open `.bz2` natively so used the built-in extraction tool
- **Step 3:** Explored the dataset structure — 8 dysarthric speakers total across both folders. Each `.wav` file has a matching `.txt` transcript with the same number. For example `0001.wav` contains the audio of someone saying what is written in `0001.txt`
- **Step 4:** Discovered two types of transcripts when opening `.txt` files:
  - Instruction prompts like `[say Ah-P-Eee repeatedly]` — these tell the speaker what sound to make, not actual sentences. Skipped these entirely
  - Real sentence transcripts like `"Except in the winter when the ooze or snow or ice prevents"` — used these for training
  - Some transcripts had inline instructions like `"tear [as in tear up that paper]"` — cleaned by removing the bracket content, keeping just `tear`
- **Step 5:** Wrote the data loading function in `model/train.py`. This function automatically:
  1. Goes into every speaker folder (F01, F03, F04, M01–M05)
  2. Goes into every session folder inside each speaker
  3. Finds the `wav_headMic` folder (audio) and `prompts` folder (transcripts)
  4. For every transcript file, reads the text inside
  5. Skips it if it starts with `[` (instruction prompt)
  6. Removes any inline bracket instructions from the text
  7. Finds the matching `.wav` audio file with the same number
  8. Adds the pair (audio path, transcript text) to a list
  9. Returns the complete list at the end
- **Step 6:** Created `.gitignore` to exclude the 2.5GB dataset from GitHub — teammates download the dataset separately from the TORGO website
- **Step 7:** Pushed updated `model/train.py`, `requirements.txt`, and `.gitignore` to GitHub with commit message `"Day 2: data loading function complete, 2917 samples"`
 
### Result
Successfully loaded **2917 clean audio-transcript pairs** from 8 dysarthric speakers.
 
---
 
## Day 3
 
### Goal
Write the audio preprocessing function, write the Whisper fine-tuning training loop, paste the full training code into Google Colab, and run training on a free cloud GPU overnight so the model is ready to evaluate by Day 4.
 
### What is Preprocessing?
Before the AI model can learn from audio, the raw `.wav` files need to be converted into a format Whisper understands. This is called preprocessing:
1. Load the audio file using `librosa`
2. Resample to 16000Hz (16kHz) — Whisper requires this exact sample rate
3. Extract log-Mel features using Whisper's own processor — this converts the audio waveform into a visual representation of sound frequencies that the model can learn from
 
This is a lossless conversion — no information is lost, just reformatted.
 
### What is Fine-Tuning?
Whisper is already pre-trained on 680,000 hours of normal speech. Fine-tuning means we take this already-smart model and teach it the specific patterns of dysarthric speech using our TORGO dataset. Think of it like a doctor who already knows medicine, now specialising in a specific condition.
 
### What is Loss?
During training, the model makes a prediction for each audio file and compares it to the correct transcript. The difference between the prediction and the correct answer is called loss:
- High loss = model got it very wrong
- Low loss = model got it right
- The goal is to reduce loss over time through 10 epochs of training
 
### What is an Epoch?
One epoch means the model has seen all 2917 samples once. We run 10 epochs — so the model sees the full dataset 10 times, getting better each time.
 
### Why Google Colab and Kaggle?
Training a deep learning model requires a GPU (Graphics Processing Unit) which is extremely expensive hardware. Our laptops don't have one. Google Colab and Kaggle both offer free cloud GPUs:
- Google Colab — free Tesla T4 GPU, ~12 hours per day
- Kaggle — free Tesla T4 x2 GPU, 30 hours per week
 
We used Colab for epochs 1–6 and switched to Kaggle for epochs 7–10 after hitting Colab's daily GPU limit.
 
### Steps Completed
 
- **Step 1:** Added preprocessing function to `model/train.py` — loads each `.wav` file, resamples to 16kHz, extracts Mel features using `WhisperProcessor`
- **Step 2:** Added training loop to `model/train.py` — loads `whisper-small` from HuggingFace, moves it to GPU, runs 10 epochs, saves checkpoints every 2 epochs so progress is not lost if the GPU disconnects
- **Step 3:** Set up Google Colab — created new notebook, switched runtime to T4 GPU, mounted Google Drive, uploaded TORGO dataset to Drive, ran training
- **Step 4:** Training ran epochs 1–6 on Colab. Loss dropped from 2.69 → 0.0001 showing significant learning. Checkpoint saved to Google Drive after every 2 epochs
- **Step 5:** Hit Colab's daily GPU limit after epoch 6 — switched to Kaggle to continue training
- **Step 6:** Set up Kaggle notebook — uploaded TORGO dataset and epoch 6 checkpoint to Kaggle as a dataset, switched accelerator to T4 x2 GPU, loaded checkpoint and resumed training from epoch 7
- **Step 7:** Epochs 7, 8, 9, 10 ran on Kaggle GPU. Final model saved as `dysvoice_whisper.pt`
 
---
 
## Day 4
 
### Goal
Download the completed trained model from Kaggle, write `model/evaluate.py` to measure accuracy on unseen test speakers, confirm the model hits the 85% target, and share the model file with teammates via Google Drive.
 
### What is Evaluation?
After training, we need to verify the model actually works on speech it has never heard before. We test on two TORGO speakers who were not used during training:
- **M04** — male dysarthric speaker (mild/moderate)
- **F03** — female dysarthric speaker (moderate)
 
Testing on unseen data gives us a realistic measure of how the model will perform in the real world.
 
### What is WRA and WER?
- **WRA (Word Recognition Accuracy)** — percentage of words the model gets correct. Higher is better. Target: 85%+
- **WER (Word Error Rate)** — percentage of words the model gets wrong. Lower is better. WER = 1 − WRA
 
These are calculated automatically using the `jiwer` library by comparing the model's predicted transcript against the correct transcript.
 
### Steps Completed
 
- **Step 1:** Downloaded `dysvoice_whisper.pt` (967MB) from the Kaggle Output tab after Version 2 completed all 10 epochs
- **Step 2:** Placed `dysvoice_whisper.pt` in the `model/` folder inside the project
- **Step 3:** Wrote `model/evaluate.py`:
  1. Loads the trained model and Whisper processor
  2. Loads test audio files from M04 and F03 speakers
  3. Runs each audio file through the model to get predicted text
  4. Compares predicted text against the correct transcript
  5. Calculates WRA and WER using the `jiwer` library
  6. Prints results with an example showing the actual vs predicted text
- **Step 4:** Ran evaluation — results significantly exceeded the 85% target (see Accuracy Results table at the top of this README)
- **Step 5:** Uploaded `dysvoice_whisper.pt` to Google Drive and shared the link with the team on WhatsApp
 
---
 
## Day 5
 
### Goal
Verify the trained model loads and transcribes correctly end-to-end by writing a standalone test script. Share the exact model loading code with Developer 2 so they can integrate the fine-tuned model into `inference/transcribe.py` without guesswork.
 
### What is test_model.py?
`test_model.py` is a quick verification script written now that the trained model exists. There was no point writing it earlier since the model was not ready until Day 4. It does three things:
1. Loads the fine-tuned model from `config.MODEL_PATH`
2. Loads a TORGO `.wav` file using `librosa`
3. Runs the audio through the model and prints the transcript
 
This confirms the entire model loading and inference chain works correctly on the local machine before Developer 2 integrates it into the full pipeline.
 
### Steps Completed
 
- **Step 1:** Uploaded `dysvoice_whisper.pt` to Google Drive and confirmed the download link works
- **Step 2:** Wrote `model/test_model.py` with full `try/except` error handling so any issues print clearly instead of the script failing silently with no output
- **Step 3:** Hit a `ModuleNotFoundError` for `config` when running `python model/test_model.py` — fixed by running as `python -m model.test_model` instead, so Python looks for `config.py` from the project root folder
- **Step 4:** Added `language="en"` to the processor and generate call to suppress a HuggingFace warning about multilingual mode defaulting to language detection
- **Step 5:** Added `attention_mask=torch.ones_like(input_features)` to suppress a second warning about the pad token matching the eos token
- **Step 6:** Added `model.config.forced_decoder_ids = None` and `processor.tokenizer.forced_decoder_ids = None` after `model.eval()` to resolve a conflict between the fine-tuned model's saved decoder settings and the `language="en"` parameter
- **Step 7:** Confirmed clean successful output:
  ```
  ✅ Model loaded successfully
  ✅ Audio loaded successfully
  🎙  Input file : ...wav_headMic\0001.wav
  📝 Transcript : I'll be healthy.
  ```
- **Step 8:** Shared the model loading snippet with Developer 2 on WhatsApp so they can swap the base Whisper placeholder in `transcribe.py` for the fine-tuned model
- **Step 9:** Ran `git pull` — received Developer 3's `speak.py`, `display.py`, and `README.md` updates (Fast-forward, no conflicts). Then pushed `test_model.py` to GitHub with commit message `"Day 5: test_model.py confirmed working, model verified"`
 
### Model Loading Code (for Developer 2)
This is the exact snippet to use in `inference/transcribe.py`:
 
```python
import torch
import config
from transformers import WhisperProcessor, WhisperForConditionalGeneration
 
processor = WhisperProcessor.from_pretrained(config.MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_NAME)
 
state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
model.load_state_dict(state_dict)
model.to(config.DEVICE)
model.eval()
model.config.forced_decoder_ids = None
processor.tokenizer.forced_decoder_ids = None
```
 
---
 
# Developer 2 — Audio Pipeline
 
## Day 1
 
### Goal
Clone the repository, install all libraries, download TORGO sample files for testing, and explore the audio data to understand what dysarthric speech looks and sounds like before writing any code.
 
### Steps Completed
 
- **Step 1:** Cloned the GitHub repository once Developer 1 shared the link using `git clone`. This downloaded the entire project folder including `config.py`, `requirements.txt`, and all the empty placeholder files Developer 1 had already created
- **Step 2:** Ran `pip install -r requirements.txt` in the terminal. This automatically downloaded and installed every library the project needs — PyAudio for microphone access, noisereduce for noise cleaning, transformers for the Whisper model, and all other dependencies in one command
- **Step 3:** Downloaded 3–4 sample `.wav` files from the TORGO dataset on Kaggle. These are recordings of real dysarthric speakers. Used these sample files throughout development to test the audio pipeline without needing a dysarthric person present. Picked samples from M04 (mild dysarthric male speaker) and F03 (mild dysarthric female speaker) as these are the clearest samples to test with
- **Step 4:** Opened the `.wav` files and listened to them to understand what dysarthric speech sounds like. Read the matching `.txt` transcript files to understand how the audio and text are paired. This understanding was important before writing any recording or processing code
 
---
 
## Day 2
 
### Goal
Write `audio/record.py` — the microphone recording module with Voice Activity Detection so the system automatically detects when someone starts and stops speaking without needing a button press.
 
### What is Voice Activity Detection?
Voice Activity Detection (VAD) is a technique that monitors the microphone input continuously and automatically detects when a person starts speaking and when they stop. Instead of pressing a button to start and stop recording, the system listens in the background and begins capturing audio the moment it detects sound above a certain volume level. It then automatically stops recording after 1.5 seconds of silence. This is important for dysarthric users because pressing buttons can be physically difficult for them.
 
### Steps Completed
 
- **Step 1:** Wrote `audio/record.py` with the following key components:
  1. `rms()` function — calculates the Root Mean Square energy of each audio chunk. RMS is a mathematical way of measuring the loudness of a sound. Every chunk of audio coming from the microphone is checked against a silence threshold. If the RMS value is above the threshold, the system considers it speech. If below, it considers it silence
  2. `record_audio()` function — the main function that Developer 3 will call from `main.py`. It works in two states:
     - **WAITING state** — reads audio chunks from the microphone but throws them away, waiting silently until speech is detected. The moment a loud enough chunk arrives it switches to RECORDING state
     - **RECORDING state** — keeps all audio chunks and counts consecutive silent chunks. Stops recording once 1.5 seconds of silence have passed. Joins all collected chunks together, converts from raw PCM bytes to a numpy array, normalises values to the range -1.0 to 1.0, and returns the final float32 array
  3. Key settings used:
     - `SAMPLE_RATE` of 16000 Hz matching `config.py`
     - `CHUNK` size of 512 frames (~32 milliseconds of audio per chunk)
     - `SILENCE_THRESHOLD` of 100 RMS units (tuned by testing on the laptop microphone)
     - `SILENCE_DURATION` of 1.5 seconds
     - `MAX_DURATION` of 10 seconds as a hard cap to prevent runaway recordings
- **Step 2:** Ran `python -m audio.record` in the terminal. Spoke into the laptop microphone. The terminal printed `Listening`, then `Recording` when speech was detected, then `Done` when silence was held for 1.5 seconds. Confirmed the output was a numpy array with the correct shape, dtype of `float32`, and values between -1.0 and 1.0. The recording was also saved as `test_recording.wav` and played back to confirm audio quality was clean
- **Step 3:** Tuned the silence threshold — the first run printed `No speech detected` because the default threshold of 300 was too high for the laptop microphone. Lowered it to 100 which correctly detected normal speaking volume without triggering on background noise
- **Step 4:** Fixed a VS Code run error where VS Code tried to run a Solidity blockchain debugger extension instead of the Python file. Fixed by always running files from the terminal using `python -m audio.record` instead of the VS Code play button
- **Step 5:** Configured Git identity using `git config --global` with name and email. Accepted the collaborator invitation from Developer 1 on GitHub. Resolved a merge conflict caused by teammates pushing code while Git was being configured — fixed by running `git add`, `git commit`, and `git push` in sequence. Final push confirmed with commit message `"Day 2 and 3: record.py with VAD and denoise.py complete"`
 
---
 
## Day 3
 
### Goal
Write `audio/denoise.py` — the noise reduction module that takes the raw audio array from `record.py` and cleans it before sending it to the AI model for transcription.
 
### Why is noise reduction important for dysarthric speech?
Dysarthric speech is already harder for AI models to understand because of irregular rhythm, slurred sounds, and reduced clarity. Background noise on top of this makes it significantly worse. Cleaning the audio before passing it to Whisper improves transcription accuracy without changing anything in the model itself. Even a small improvement in audio quality can make a meaningful difference in the final transcript.
 
### Steps Completed
 
- **Step 1:** Wrote `audio/denoise.py` with two processing steps inside the `denoise_audio()` function:
  1. **Noise reduction using `noisereduce`** — this library analyses the audio signal and estimates what the background noise looks like, then mathematically subtracts that noise pattern from the entire recording. It works in just one line of code but makes a significant audible difference on microphone recordings
  2. **Amplitude normalisation** — after noise reduction the volume level of the audio is scaled so the loudest point in the recording equals exactly 1.0. This ensures that quiet speakers are amplified to a consistent level and loud speakers are not clipping. This is important because dysarthric speakers often have reduced vocal volume
- **Step 2:** Tested by running `python -m audio.denoise test_samples\array0001.wav` in the terminal. The output showed before and after amplitude values confirming the normalisation worked. Two files were saved — `test_raw.wav` and `test_clean.wav`. Played both back in File Explorer and confirmed the clean version had noticeably less background noise
- **Step 3:** Noticed that a `__pycache__` folder was being pushed to GitHub. Created a `.gitignore` file with entries for `__pycache__/`, `*.pyc`, `*.pyo`, `test_recording.wav`, `test_raw.wav`, `test_clean.wav`, and `test_samples/` to keep the repository clean with only source code
- **Step 4:** Ran `git rm -r --cached __pycache__` to remove the already-uploaded `__pycache__` from the repository history. Committed and pushed the fix so the GitHub repository shows only correct source code files
- **Step 5:** Pushed completed `denoise.py` and updated `.gitignore` with commit message `"Remove pycache and add .gitignore"`
 
---
 
## Day 4
 
### Goal
Write `inference/transcribe.py` — the transcription module that takes the cleaned audio array from `denoise.py` and passes it through the Whisper model to produce a text string that the rest of the pipeline can use.
 
### What is Whisper?
Whisper is a speech recognition model created by OpenAI. It was trained on 680,000 hours of audio from the internet, making it significantly more accurate than older speech recognition systems. It works by converting audio into a log-Mel spectrogram (a visual representation of sound frequencies over time) and passing it through a transformer neural network that produces text. We use the `whisper-small` version which balances accuracy and speed well enough to run on a laptop CPU.
 
### Steps Completed
 
- **Step 1:** Created the `inference/` folder by running `mkdir inference` in the terminal. Created `inference/__init__.py` — an empty file that tells Python this folder is a package that can be imported from. Created `inference/transcribe.py` as an empty file ready for code
- **Step 2:** Wrote `inference/transcribe.py` with the following key components:
  1. **Model loading logic** — when the file is first imported, it checks whether Developer 1's fine-tuned model file exists at `model/dysvoice_whisper.pt`. If it exists it loads that. If not, it falls back to the base `openai/whisper-small` model downloaded from HuggingFace. This means the same code works on Day 4 with the base model and automatically upgrades on Day 5 when Developer 1 pushes the fine-tuned model — no code changes needed
  2. **`transcribe()` function** — takes a `float32` numpy audio array as input. Passes it through `WhisperProcessor` to extract input features. Runs the features through `WhisperForConditionalGeneration` with `torch.no_grad()` to save memory. Decodes the predicted token IDs back into readable text. Returns the text as a plain string. Returns an empty string if anything fails so the rest of the pipeline does not crash
- **Step 3:** Ran `pip install transformers torch` in the terminal. The first run also downloaded the Whisper small model (~460MB) which took 2–3 minutes
- **Step 4:** Tested with TORGO `.wav` files by running `python -m inference.transcribe test_samples\array0001.wav`. Compared the printed transcript against the matching `.txt` file. Accuracy at this stage is ~50–60% because the base Whisper model has not been fine-tuned on dysarthric speech yet — this is expected and will improve significantly on Day 5 when the fine-tuned model is integrated
- **Step 5:** Pushed `inference/transcribe.py` and the updated `.gitignore` with commit message `"Day 4: transcribe.py complete, tested with TORGO samples"`
 
---

## Day 5

## Goal
Pull the fine-tuned model pushed by Developer 1, integrate it into inference/transcribe.py, and run the full live pipeline for the first time — microphone recording to denoising to real Whisper transcription to text output — confirming all three modules work together end to end.
### What changed from Day 4?
On Day 4, transcribe.py was using the base openai/whisper-small model as a placeholder because Developer 1's fine-tuned model was not ready yet. On Day 5, Developer 1 pushed the trained dysvoice_whisper.pt file to Google Drive and shared the download link. The model loading code in transcribe.py was updated to load Developer 1's fine-tuned weights on top of the base Whisper architecture using torch.load() and model.load_state_dict(). This is called transfer learning integration — the model structure stays the same but the learned weights are replaced with the dysarthric-specific ones.
Steps Completed

## Step 1: 
Ran git pull to get the latest code from the repository. Received Developer 1's model/test_model.py and Developer 3's latest main.py updates
## Step 2:
 Downloaded dysvoice_whisper.pt (967MB) from the Google Drive link shared by Developer 1 on WhatsApp. Placed the file in the model/ folder:

  Dysvoice/
  └── model/
      └── dysvoice_whisper.pt

## Step 3: 
Rewrote inference/transcribe.py with Developer 1's exact model loading method:

Load WhisperProcessor from the base openai/whisper-small model on HuggingFace
Load WhisperForConditionalGeneration base architecture from HuggingFace
Load the fine-tuned weights from model/dysvoice_whisper.pt using torch.load() with map_location=config.DEVICE
Apply the weights to the model using model.load_state_dict(state_dict)
Move the model to the correct device using model.to(config.DEVICE)
Set model to evaluation mode using model.eval()
Clear forced decoder settings using model.config.forced_decoder_ids = None and processor.tokenizer.forced_decoder_ids = None


## Step 4: 
Tested with TORGO .wav files — confirmed the terminal printed:

  [transcribe] Loading processor from openai/whisper-small
  [transcribe] Loading base model architecture from openai/whisper-small
  [transcribe] Applying fine-tuned weights from model/dysvoice_whisper.pt
  [transcribe] Fine-tuned model loaded successfully
  Transcript: 'I can play this weekend.'
  Test PASSED — transcribe() returned a string successfully.

## Step 5: 
Ran the full live pipeline for the first time connecting all three modules together:

python  audio = record_audio()      # mic → numpy array
  clean = denoise_audio(audio) # numpy array → cleaned array
  text  = transcribe(clean)    # cleaned array → text string
Spoke into the microphone, confirmed the system printed Listening..., then Recording..., then Done., then printed the final transcript. The full chain worked end to end on a live microphone input

## Step 6: 
Encountered and resolved a ModuleNotFoundError: No module named 'noisereduce' on this machine — fixed by running pip install noisereduce
## Step 7:
 Noticed test_output.wav was accidentally pushed to GitHub. Removed it using git rm --cached test_output.wav, added it to .gitignore, and pushed the fix
## Step 8: 
Developer 1 flagged two lines to add after model.eval() to fix decoder warnings:

python  _model.config.forced_decoder_ids = None
  _processor.tokenizer.forced_decoder_ids = None
```
  Added these lines to `inference/transcribe.py` to suppress the `forced_decoder_ids` conflict warnings that appeared during transcription
  Step 9: Pushed all completed work to GitHub:
```
  git add .
  git commit -m "Day 5: fine-tuned model integrated, full live pipeline tested successfully"
  git push
```
- **Step 10:** Messaged Developer 3 on WhatsApp confirming all three functions are ready with their exact signatures:
```
  record_audio()        → returns numpy array
  denoise_audio(audio)  → returns numpy array
  transcribe(audio)     → returns string
## Result
Full pipeline working end to end. Fine-tuned model loads successfully and returns transcriptions. Live microphone input flows correctly through all three modules — record.py → denoise.py → transcribe.py — and produces a text string output ready for Developer 3's speak() and display_text() functions in main.py.
 
# Developer 3 — Output & Integration
 
## Day 1
 
### Goal
Clone the repository, install all libraries, and write `output/speak.py` entirely on Day 1. This is the simplest module in the project and finishing it completely on the first day means Developer 3 can move straight into `display.py` and `main.py` from Day 2 without any catch-up work.
 
### Steps Completed
 
- **Step 1:** Cloned the GitHub repository once Developer 1 shared the repository link using `git clone`. This downloaded the entire project folder including `config.py`, `requirements.txt`, and all the empty placeholder files Developer 1 had already created — including the `output/` folder where `speak.py` and `display.py` will live
- **Step 2:** Ran `pip install -r requirements.txt` in the terminal. This automatically downloaded and installed every library the project needs — `pyttsx3` for text-to-speech output, `librosa` for loading `.wav` files in demo mode, `argparse` for command-line arguments in `main.py`, and all other dependencies in one command
- **Step 3:** Wrote `output/speak.py` entirely on Day 1. Used `pyttsx3` to convert text strings to spoken audio. Configured the speech rate and volume using the shared values from `config.py` (`TTS_RATE = 150`, `TTS_VOLUME = 1.0`). Wrote two functions inside the file:
  - `speak(text)` — takes a string and plays it aloud through the system speakers. This is the main function that `main.py` will call after every transcription. Returns nothing
  - `save_audio(text, output_path)` — saves the TTS output to a `.wav` file on disk instead of playing it through the speakers. This is the demo backup function — if the live pipeline fails on demo day, pre-generated audio files can be played on demand
- **Step 4:** Tested `speak.py` directly from the terminal. Tested with multiple sentences — *"please bring me water"*, *"turn off the lights"*, *"I need help"* — and confirmed each sentence was spoken clearly through the laptop speakers. Adjusted the speech rate to sound natural and not robotic. Also ran `save_audio()` and confirmed the `.wav` file appeared on disk and played back correctly
- **Step 5:** Ran `git add .`, `git commit -m "Day 1 Speak complete"`, and `git push` to upload the completed `speak.py` to the shared repository so the team can see progress
 
---
 
## Day 2
 
### Goal
Finalise `speak.py` with clean, focused function signatures, write `output/display.py` with a terminal fallback that is already structured to switch to physical OLED hardware with a single flag change when the Raspberry Pi arrives, and push both completed output modules to GitHub.
 
### What is pyttsx3?
`pyttsx3` is a Python text-to-speech library that works completely offline — no internet connection needed. It uses the operating system's built-in speech engine: SAPI5 on Windows, nsss on Mac, and espeak on Linux and the Raspberry Pi. This is important for DysVoice because the final device must work without Wi-Fi. The library takes any text string and converts it into spoken audio through the system's speakers in real time.
 
### What is the OLED display?
The OLED screen is a small physical display that will be attached to the Raspberry Pi via I2C connection. It shows the transcribed text on screen so the user can read what the system heard as well as hear it. It uses the `luma.oled` library with an `ssd1306` driver. Since the hardware has not arrived yet, the OLED import lines were written but left commented out inside `display.py`. When the Pi arrives, these lines get uncommented and the terminal print gets removed — everything else stays the same.
 
### Steps Completed
 
- **Step 1:** Reviewed the `speak.py` written on Day 1 and cleaned it up. Removed unused imports (`numpy`, `wave`, `tempfile`) that were carried over during development but not actually needed in the final version. Tightened the `speak()` function so it does exactly one thing — takes a string, speaks it aloud, returns nothing. This clean, focused signature is important because `main.py` will call `speak(text)` directly and must not receive any unexpected return value that could cause an error
- **Step 2:** Confirmed the exact function signatures that `main.py` depends on before writing anything else. These cannot be renamed or restructured as the team plan document specifies them exactly:
  - `speak(text)` → takes a string, speaks it aloud, returns `None`
  - `save_audio(text, output_path)` → takes a string and a file path, saves the speech as a `.wav` file, returns the file path
- **Step 3:** Wrote the `save_audio()` helper inside `speak.py` using `pyttsx3`'s built-in `save_to_file()` method. Instead of playing audio through the speakers, this saves the TTS output as a `.wav` file to a given path on disk. This is the demo backup strategy — if the live microphone or TTS fails on demo day, pre-generated `.wav` files of common phrases like *"please bring me water"* can be played on demand without running the full pipeline. Tested by calling `save_audio("This is a saved audio clip for the demo.", "test_output.wav")` and confirmed the file appeared on disk and played back correctly
- **Step 4:** Wrote `output/display.py` with a terminal fallback. This module has one job — show the transcribed text on screen after the AI produces it. Since the Raspberry Pi and OLED hardware have not arrived yet, wrote it to print to the terminal as a fallback. The terminal output is formatted inside a clear visual box so it is easy to read during the demo:
  ```
  ┌──────────────────────────────────────────────────┐
  │  Transcript: please bring me water               │
  └──────────────────────────────────────────────────┘
  ```
- **Step 5:** Added an `OLED_ENABLED` flag at the top of `display.py`. When hardware arrives, flipping this flag to `True` and uncommenting the OLED library block will switch the module from terminal output to physical screen output — without changing anything in `main.py`. The function signature `display_text(text)` stays identical either way, so the rest of the pipeline never needs to know whether the output is going to a terminal or a hardware screen
- **Step 6:** Understood what OLED hardware integration will look like when the Pi arrives. The OLED screen connects to the Pi via I2C (SDA → GPIO2, SCL → GPIO3, VCC → 3.3V, GND → GND). The `luma.oled` library with `ssd1306` driver handles all the low-level communication with the screen. The import lines for this are already written and commented out inside `display.py` — ready to activate the moment hardware arrives
- **Step 7:** Tested both files on the laptop. Ran `python output/speak.py` — confirmed all test sentences were spoken clearly through the laptop speakers. Ran `python output/display.py` — confirmed the formatted transcript box printed correctly in the terminal for all four test phrases. Both files ran without errors
- **Step 8:** Ran `git add .`, `git commit -m "Day 2: speak.py finalised, display.py written with terminal fallback"`, and `git push` to upload both completed output modules to the shared repository
 
### Result
Both output modules are complete and pushed. `speak.py` has clean, tested function signatures ready for `main.py` to call. `display.py` works in terminal mode now and is structured to switch to OLED with a single flag change when hardware arrives. Developer 3 can now move on to writing `main.py` on Day 3 with both output modules fully in place.
 
---
