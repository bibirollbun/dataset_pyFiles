# Install Unsloth
!pip install unsloth 


# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers 
!pip install --no-deps --upgrade timm 


from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # or any other supported model variant
    dtype = None,              
    max_seq_length = 1024,     
    load_in_4bit = True,        
    full_finetuning = False
)



from datasets import load_dataset

# Download the IndicTTS Tamil dataset
dataset = load_dataset("SPRINGLab/IndicTTS_Tamil", split="train")

# Preview a few audio samples and their transcripts
print(dataset[0])  # Each entry usually has 'path', 'audio', 'text' keys



import librosa
# Step 2: Access the first sample's audio array and sampling rate
audio_array = dataset[0]['audio']['array']
original_sr = dataset[0]['audio']['sampling_rate']

# Step 3: Resample audio from original_sr (likely 48000) to 16000 (needed for ASR models like Whisper)
audio_16k = librosa.resample(audio_array, orig_sr=original_sr, target_sr=16000)

print(f"Original Sample Rate: {original_sr}")
print(f"Resampled audio shape: {audio_16k.shape}")



from transformers import pipeline

# Initialize Whisper ASR pipeline (set device as needed: -1 for CPU, 0/1 for GPU)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)

# Transcribe your resampled audio
result = asr(audio_16k)  # You can also pass [audio_16k] as a batch
tamil_transcript = result['text']

print("ASR Transcript:", tamil_transcript)



for i in range(3):  # first 3 samples
    audio_array = dataset[i]['audio']['array']
    original_sr = dataset[i]['audio']['sampling_rate']
    audio_16k = librosa.resample(audio_array, orig_sr=original_sr, target_sr=16000)
    
    result = asr(audio_16k)
    tamil_transcript = result.get('text', '').strip()
    if not tamil_transcript:
        print(f"No transcription for sample {i}")
        continue
    
    prompt = f"Summarize the following Tamil oral literature:\n\n{tamil_transcript}\n\nSummary:"
    inputs = tokenizer(text=prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(**inputs, max_new_tokens=256)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Sample {i} Transcript: {tamil_transcript}")
    print(f"Sample {i} Summary: {summary}")
    print("-" * 60)


