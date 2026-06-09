from huggingface_hub import notebook_login
notebook_login()


# Install a transformers version that supports Gemma 3n (>= 4.53)
!pip install "transformers>=4.53.0" "timm>=1.0.16"


! pip install transformers


!pip install whisper
!pip install geopy
!pip install requests
!pip install pyttsx3
!pip install IPython
! pip install gTTS
!pip install google-cloud-texttospeech


pip install --upgrade transformers huggingface_hub 


import kagglehub
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import whisper
import torch
import requests
import pyttsx3  
from IPython.display import Audio, Image, Markdown, display
from pydub import AudioSegment 

engine = pyttsx3.init()

# Download GEMMA model using KaggleHub
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")

# Download the disaster crisis dataset (voice input) using KaggleHub
dataset_path = kagglehub.dataset_download("bhoomikabeniwal/gemma3n-voice-input")
print(f"Disaster Voice Dataset downloaded at: {dataset_path}")

# Load the GEMMA model and processor from the downloaded path
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

whisper_model   = whisper.load_model("base")


import os

def process_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path , format = "opus")
        audio.export("temp.wav" , format = "wav")

        with open("temp.wav" , "rb") as audio_file:
            audio_data = audio_file.read()

        if recognizer.AcceptWaveform(audio_data):
            result = whisper_model.Result()
            text = result.split('"text:"')[:1].split('""')[1]
            print(f"User said: {text}")
            retrun text
        else:
            print("Could not understand the audio.")
            return text
    except Exception as e:
        print(f"error processing audio:{e}")
        retrun None

def generate_response(user_input):
    try:
        input = tokenizer(user_input , return_tensors = "pt")

        outputs = model.generate(input['input_ids'] , max_length = 150 , num_return_sequeneces= 1)
        response = tokenizer.decode(outputs[0] , skip_spoecial_token = True)
        print(f"gemms 3n response:{response}")
        return response
    except exception as e:
        print(f"error generating response: {e}")
        return "sorry , I could not process that."

def speak_text(response_text):
    try:
        engine.say(response_text)
        engine.runAndWait()
    except Exception as e:
        print(f"error with TTS: {e}")

def process_and_assist(audio_file_path):
    user_input = process_audio(audio_file_path)
    if user_input:
        response = generate_response(user_input)
        speak_text(response)

    else:
        print("no valid input to process")

audio_file = os.path.join(audio_folder_path , audio_files[0])
process_and_asssit(audio_file)


import kagglehub
import whisper  # For offline speech recognition (Whisper by OpenAI)
from pydub import AudioSegment  # To handle audio files
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from google.cloud import texttospeech  # Google TTS library

# Initialize Google TTS client
client = texttospeech.TextToSpeechClient()

# Download GEMMA model using KaggleHub
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")

# Download the disaster crisis dataset (voice input) using KaggleHub
dataset_path = kagglehub.dataset_download("bhoomikabeniwal/gemma3n-voice-input")
print(f"Disaster Voice Dataset downloaded at: {dataset_path}")

# Load GEMMA model and processor from the downloaded path
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

# Load Whisper model for speech 
whisper_model = whisper.load_model("base")  # You can choose "small", "medium", or "large" for more accuracy

# Function to process audio file (Opus format)
def process_audio(file_path):
    try:
        # Load the Opus file using pydub and convert to WAV
        audio = AudioSegment.from_file(file_path, format="opus")
        audio.export("temp.wav", format="wav")  # Save as WAV

        # Use Whisper to transcribe the WAV file
        result = whisper_model.transcribe("temp.wav")
        text = result["text"]
        print(f"User said: {text}")
        return text
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# Function to generate response from Gemma 3N model
def generate_response(user_input):
    try:
        # Tokenize the user input
        inputs = processor(user_input, return_tensors="pt")

        # Pass the input through the model to generate a response
        outputs = model.generate(inputs['input_ids'], max_length=150, num_return_sequences=1)

        # Decode the generated output and return the response text
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"Gemma 3N Response: {response}")
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't process that."

# Function to convert text to speech and play it using Google TTS
def speak_text(response_text):
    try:
        # Set up the TTS request
        synthesis_input = texttospeech.SynthesisInput(text=response_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Wavenet-D"
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        # Perform the synthesis request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save the audio to a file and play it
        with open("response.mp3", "wb") as out:
            out.write(response.audio_content)
        
        # Optionally, play the audio using an external player (in a local environment)
        print("Audio has been saved to 'response.mp3'.")
    except Exception as e:
        print(f"Error with TTS: {e}")

# Main function to process audio and provide assistance
def process_and_assist(audio_file_path):
    # Step 1: Process audio to text
    user_input = process_audio(audio_file_path)

    if user_input:
        # Step 2: Generate response from Gemma 3N
        response = generate_response(user_input)

        # Step 3: Convert response to speech and play it
        speak_text(response)
    else:
        print("No valid input to process.")

# Example usage
audio_file = "path_to_your_audio.opus"  # Replace with the actual path to your audio file
process_and_assist(audio_file)


