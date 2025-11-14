import os
from dotenv import load_dotenv
load_dotenv()
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
import numpy as np
import tempfile

SAMPLE_RATE = 44100     # Standard quality
DURATION = 5            # Record X seconds

openai_api_key = os.getenv("OPENAI_API_KEY")

def record_audio(filename):
    print("🎙️ Recording... Speak now.")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print(f"Saved recording to {filename}")

def transcribe_audio(filename):
    client = OpenAI(api_key=openai_api_key)

    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=f
        )

    return transcript.text

def main():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        filepath = temp.name

    record_audio(filepath)
    print("Transcribing...")
    text = transcribe_audio(filepath)
    print("\nTranscription:")
    print(text)

if __name__ == "__main__":
    main()
