import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import torch
import time
import argparse
from datetime import datetime

def get_optimal_device():
    """Determine the best available device for computation"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def record_audio(duration, samplerate=16000, channels=1):
    """
    Record audio from the microphone.
    
    Args:
        duration (int): Recording duration in seconds
        samplerate (int): Sample rate in Hz
        channels (int): Number of audio channels
    """
    print(f"\nRecording for {duration} seconds...")
    print("Speak now!")
    
    # Record audio
    recording = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype=np.float32
    )
    
    # Display a simple progress bar
    for i in range(duration):
        time.sleep(1)
        print(".", end="", flush=True)
    print("\nFinished recording!")
    
    # Wait for recording to complete
    sd.wait()
    
    return recording, samplerate

def save_audio(recording, samplerate):
    """Save the recorded audio to a WAV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    
    sf.write(filename, recording, samplerate)
    print(f"\nAudio saved as: {filename}")
    
    return filename

def transcribe_audio(audio_file, use_m4=True, language=None):
    """Transcribe audio using Whisper"""
    try:
        device = get_optimal_device() if use_m4 else torch.device("cpu")
        print(f"\nUsing device: {device}")
        
        print("Loading Whisper model...")
        model = whisper.load_model("medium")
        
        if device.type == "mps":
            try:
                model = model.to(device)
                print("Using Apple Silicon Neural Engine")
                torch.mps.empty_cache()
            except Exception as e:
                print(f"Warning: Couldn't use Neural Engine: {e}")
                device = torch.device("cpu")
                model = model.to(device)
                print("Falling back to CPU")
        
        # Configure transcription options
        options = {
            "task": "transcribe",
            "fp16": False
        }
        
        if language:
            options["language"] = language
        
        # Transcribe
        print("\nTranscribing...")
        result = model.transcribe(audio_file, **options)
        
        print("\nTranscription:")
        print(result["text"])
        
        # Save transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcription_{timestamp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"\nTranscription saved to: {output_file}")
        
    finally:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Record audio and transcribe with Whisper")
    parser.add_argument(
        "--duration", 
        type=int, 
        default=30,
        help="Recording duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default=None,
        help="Language code (e.g., 'en' for English, 'tl' for Tagalog)"
    )
    parser.add_argument(
        "--cpu", 
        action="store_true",
        help="Force CPU usage instead of M4"
    )
    
    args = parser.parse_args()
    
    # Record audio
    recording, samplerate = record_audio(args.duration)
    
    # Save audio
    audio_file = save_audio(recording, samplerate)
    
    # Transcribe
    transcribe_audio(audio_file, use_m4=not args.cpu, language=args.language)

if __name__ == "__main__":
    main()