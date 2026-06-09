# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git
# add more to make demo
!pip install gradio


import warnings
warnings.filterwarnings('ignore')


%%time
import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


%%time
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(result)


%%time
def gemma_query(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


%%time
import gradio as gr

# Define the interface
demo = gr.Interface(
    fn=gemma_query,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=gr.Textbox(label="Gemma 3n Response"),
    title="Gemma 3n: [Your Project Name]",
    description="Powered by Gemma 3n â€“ A private, offline-first, multimodal AI model.",
    examples=[["Tell me about climate change."], ["Explain photosynthesis in simple terms."]],
    theme="default"
)

# Launch the interface
demo.launch()


!pip install langdetect -q
!pip install googletrans==4.0.0-rc1 -q
!pip install --upgrade httpx httpcore


%%time
# ğŸ§¹ Clear GPU memory (before loading large models)
import torch
import gc

def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"ğŸ§¹ GPU memory cleared on: {torch.cuda.get_device_name(0)}")
    else:
        print("âš ï¸� No GPU available.")

clean_gpu_memory()


%%time
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


%%time
# ğŸ“š 2. Load GEMMA model + tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_PATH,
    torch_dtype=torch.float16,
    device_map="auto",       # No bitsandbytes / quant
)

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)


%%time
# ğŸ¤– 3. Ask GEMMA a multilingual support question
def ask_gemma(user_input, lang="en"):
    prompt = f"<|start_of_turn|>user (language: {lang}): {user_input}<|end_of_turn|>\n<|start_of_turn|>model:"
    out = pipeline(
        prompt,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        return_full_text=False
    )
    return out[0]["generated_text"].strip()



%%time
from IPython.display import Image

IMAGE_URL = "https://storage.googleapis.com/kaggle-media/competitions/question_goose.png"
Image(url=IMAGE_URL, height=250, width=250)


%%time
import requests

# Download image
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
image_path = "workout_pose.jpg"
with open(image_path, 'wb') as f:
    f.write(requests.get(img_url).content)

# Download audio
audio_url = "https://github.com/mgeier/python-audio/blob/master/examples/audio_samples/440Hz-44100Hz-1sec.wav?raw=true"
audio_path = "breathing.wav"
with open(audio_path, 'wb') as f:
    f.write(requests.get(audio_url).content)


%%time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, pipeline
import torchaudio

def analyze_text(user_input):
    # Text-only analysis
    return f"ğŸ§  Based on your goal '{user_input}', increase protein intake and try HIIT 3x/week."

def analyze_image(image_path):
    # Vision-language analysis
    image = Image.open(image_path)
    prompt = "Describe and assess the workout form in this image."
    #inputs = processor(prompt, image, return_tensors="pt").to("cuda", torch.float16)
    output = model.generate(**inputs, max_new_tokens=100)
    #result = processor.decode(output[0], skip_special_tokens=True)
    return f"ğŸ“¸ Pose analysis: {result}"

def analyze_audio(audio_path):
    # Audio breathing pattern analysis (placeholder)
    waveform, sample_rate = torchaudio.load(audio_path)
    # Just check volume/duration for demo
    duration = waveform.shape[1] / sample_rate
    avg_volume = waveform.abs().mean().item()
    if avg_volume < 0.02:
        return "ğŸ”Š Breathing too shallow. Try deeper inhales."
    elif duration < 5:
        return "â�±ï¸� Breathing too short. Aim for longer rhythmic breathing."
    else:
        return "âœ… Breathing looks steady and controlled."

# --- Demo Input (replace with real inputs in your app)
user_text = "I want to build muscle and lose fat."

# --- Run Analysis
print(analyze_text(user_text))
#print(analyze_image(image_path))
#print(analyze_audio(audio_path))

