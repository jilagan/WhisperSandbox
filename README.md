# Mic Whisper Setup Guide

This guide will help you set up the prerequisites for running `mic_whisper.py`, a Python script that records audio from your microphone and transcribes it using OpenAI's Whisper model.

## Language Support

Tested and optimized for:
- English
- Tagalog (Filipino)
- Taglish (Code-switching between Tagalog and English)

Language-specific usage:
```bash
# For pure Tagalog content
python mic_whisper.py --language tl

# For English content
python mic_whisper.py --language en

# For Taglish (mixed language content)
python mic_whisper.py  # Let auto-detection handle it
```

Best practices for each language:
- **Tagalog**: Use `--language tl` when the content is primarily Tagalog
- **English**: Use `--language en` for Filipino-accented English
- **Taglish**: Let the auto-detection work by omitting the language flag
- For better accuracy with heavy accents, add the `--model large` flag

## System Requirements

- MacBook with Apple Silicon (M1/M2/M3/M4)
- macOS Ventura or later
- Python 3.9 or later
- Terminal access

## Installation Steps

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python Dependencies

First, install PortAudio (required for microphone access):
```bash
brew install portaudio
brew install ffmpeg
```

### 3. Set Up Python Virtual Environment

Create and activate a new virtual environment:
```bash
# Create a new directory for your project (optional)
mkdir whisper_project
cd whisper_project

# Create virtual environment
python -m venv whisper_env

# Activate virtual environment
source whisper_env/bin/activate
```

### 4. Install Required Python Packages

```bash
# Install PyTorch with M4 optimization
pip install --no-cache-dir torch torchvision torchaudio

# Install Whisper and audio processing libraries
pip install openai-whisper
pip install sounddevice
pip install soundfile
pip install numpy
```

## File Setup

1. Save the `mic_whisper.py` script to your project directory
2. Ensure it has executable permissions:
```bash
chmod +x mic_whisper.py
```

## Usage

Basic command (records for 30 seconds):
```bash
python mic_whisper.py
```

Available options:
```bash
# Record for a specific duration (in seconds)
python mic_whisper.py --duration 60

# Specify language (e.g., Tagalog)
python mic_whisper.py --language tl

# Force CPU usage instead of M4
python mic_whisper.py --cpu

# Use larger model for better accuracy
python mic_whisper.py --model large
```

## Troubleshooting

### Common Issues and Solutions

1. **PortAudio Error**
   ```bash
   # If you see "PortAudio not found" error, try:
   brew reinstall portaudio
   pip uninstall sounddevice
   pip install sounddevice
   ```

2. **Microphone Access**
   - Make sure to grant Terminal/IDE microphone access in System Preferences > Security & Privacy > Privacy > Microphone

3. **PyTorch MPS Issues**
   ```bash
   # If you encounter MPS-related errors, try reinstalling PyTorch:
   pip uninstall torch
   pip install --no-cache-dir torch torchvision torchaudio
   ```

4. **FFmpeg Missing**
   ```bash
   # If Whisper can't find FFmpeg:
   brew install ffmpeg
   ```

### Verifying Installation

Test your setup with:
```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
python -c "import whisper; print('Whisper available')"
python -c "import sounddevice as sd; print(f'Input devices: {sd.query_devices()}')"
```

## Output Files

The script generates two types of files in the same directory:
- Audio recordings: `recording_YYYYMMDD_HHMMSS.wav`
- Transcriptions: `transcription_YYYYMMDD_HHMMSS.txt`

## Memory Management

For long recordings or when processing multiple files, you might want to clear the PyTorch cache:
```python
import torch
torch.mps.empty_cache()  # For M4
```

## Support

If you encounter any issues:
1. Check that all prerequisites are installed
2. Verify microphone permissions
3. Ensure virtual environment is activated
4. Check system audio input settings

## Tips for Best Results

1. **Audio Quality**
   - Use a good quality microphone
   - Record in a quiet environment
   - Speak clearly and at a consistent volume

2. **Language Detection**
   - For mixed language content, let auto-detection work
   - For heavy accents, use the `--model large` flag
   - If accuracy is poor, try explicitly setting the language

3. **Performance**
   - Use M4 acceleration when possible
   - Clear cache between long sessions
   - Monitor system resources during long transcriptions
