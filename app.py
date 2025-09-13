import streamlit as st
from audio_recorder_streamlit import audio_recorder
from faster_whisper import WhisperModel
from transformers import pipeline, Conversation
import pyttsx3
from pydub import AudioSegment
import io

# Initialize faster-whisper
model = WhisperModel("small", device="cpu")  # CPU

def save_and_prepare_audio(recorded_bytes, target_file="audio.wav"):
    audio = AudioSegment.from_file(io.BytesIO(recorded_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(target_file, format="wav")
    return target_file

def transcribe_audio(audio_path):
    segments, _ = model.transcribe(audio_path)
    return " ".join(segment.text for segment in segments)

generator = pipeline("conversational", model="facebook/blenderbot-400M-distill", device=-1)

def fetch_ai_response(prompt):
    conversation = Conversation(prompt)
    output = generator(conversation)
    return output.generated_responses[-1]

def text_to_speech(text, output_path="response.mp3"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path

# Streamlit UI
st.title("🎤 Chat n’ Talk Buddy")
st.write("Click the microphone and interact!")

recorded = audio_recorder("Click to record")
if recorded:
    audio_file = save_and_prepare_audio(recorded)
    transcribed = transcribe_audio(audio_file)
    st.write("Transcription:", transcribed)
    ai_response = fetch_ai_response(transcribed)
    st.write("AI Response:", ai_response)
    response_audio_file = text_to_speech(ai_response)
    st.audio(response_audio_file)
