#!/bin/bash
# hardware/setup.sh
# =================
# DysVoice � Raspberry Pi Setup Script
# Run this once on the Pi when hardware arrives.
# Usage: bash hardware/setup.sh

echo "=== DysVoice Hardware Setup ==="

# Step 1: System updates
echo "[1/5] Updating system..."
sudo apt update && sudo apt upgrade -y

# Step 2: System dependencies
echo "[2/5] Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y espeak ffmpeg libsndfile1 i2c-tools python3-rpi.gpio

# Step 3: Python libraries
echo "[3/5] Installing Python libraries..."
pip3 install -r requirements.txt
pip3 install luma.oled RPi.GPIO

# Step 4: Configure audio
echo "[4/5] Configuring audio devices..."
sudo usermod -a -G audio $USER
sudo apt install -y alsa-utils
aplay -l
arecord -l

# Step 5: Test audio
echo "[5/5] Testing audio..."
python3 -c "import pyaudio; p = pyaudio.PyAudio(); print('PyAudio OK'); p.terminate()"
python3 -c "import pyttsx3; e = pyttsx3.init(); print('pyttsx3 OK')"

echo ""
echo "=== Setup Complete ==="
echo "Run the device with: python3 main.py"
