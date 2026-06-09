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


!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
!pip install --no-deps unsloth


pip install --upgrade timm


pip install -U bitsandbytes


pip install --upgrade transformers


import torch
import cv2
import gc
from PIL import Image  # Required for image conversion
from transformers import (
    TextStreamer,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# Device configuration
device = torch.device("cuda:1")
torch.cuda.set_device(device)

# Initialize Vision Model (now accepts PIL Images)
vision_model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-large",
    device=device
)

# Initialize Gemma
tokenizer = AutoTokenizer.from_pretrained("jatinsabari/Gemma-3n-deepfake_detector")
model = AutoModelForCausalLM.from_pretrained(
    "jatinsabari/Gemma-3n-deepfake_detector",
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)

# Apply chat template (Gemma-3 format)
tokenizer.chat_template = """<start_of_turn>user
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ message['content'] }}
    {% else %}
        <start_of_turn>model
        {{ message['content'] }}<end_of_turn>
    {% endif %}
{% endfor %}"""

# Video Processing Functions
def extract_key_frames(video_path, target_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // target_frames)
    
    frames = []
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert to RGB and then to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
        if len(frames) >= target_frames:
            break
            
    cap.release()
    return frames

# Process Video
video_path = "/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/DFD_manipulated_sequences/DFD_manipulated_sequences/01_02__exit_phone_room__YVGY8LOK.mp4"
frames = extract_key_frames(video_path)  # Returns list of PIL Images

# Generate descriptions
frame_descriptions = []
for frame in frames:
    # Directly pass PIL Image to the vision model
    result = vision_model(frame)
    frame_descriptions.append(result[0]['generated_text'])

# Construct Chat Input
context = "Video Frame Descriptions:\n" + "\n".join(
    [f"- Frame {i+1}: {desc}" for i, desc in enumerate(frame_descriptions)]
)

messages = [{
    "role": "user", 
    "content": f"{context}\n\n Describe whether the frames of the video is deepfaked or not"
}]

# Generate Response
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

streamer = TextStreamer(tokenizer, skip_prompt=True)
with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer
    )


import torch
import cv2
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Device configuration (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#Initialize Vision Model (BLIP)

vision_model = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-large",
    device=device
)
print("Vision model loaded successfully")



tokenizer = AutoTokenizer.from_pretrained("jatinsabari/Gemma-3n-deepfake_detector")
model = AutoModelForCausalLM.from_pretrained(
    "jatinsabari/Gemma-3n-deepfake_detector",
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)

# Apply chat template (Gemma-3 format)
tokenizer.chat_template = """<start_of_turn>user
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ message['content'] }}
    {% else %}
        <start_of_turn>model
        {{ message['content'] }}<end_of_turn>
    {% endif %}
{% endfor %}"""
print("Gemma model loaded successfully")


def extract_key_frames(video_path, target_frames=8):
    """Extract equally spaced key frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // target_frames)
    
    frames = []
    for i in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
        if len(frames) >= target_frames:
            break
            
    cap.release()
    return frames


video_path = "/kaggle/input/deep-fake-detection-dfd-entire-original-dataset/DFD_manipulated_sequences/DFD_manipulated_sequences/01_02__exit_phone_room__YVGY8LOK.mp4"  # fake video
target_frames = 8  # @param {type:"slider", min:4, max:16, step:2}

# Extract frames and generate descriptions
frames = extract_key_frames(video_path, target_frames)
frame_descriptions = []
for i, frame in enumerate(frames):
    result = vision_model(frame)
    frame_descriptions.append(f"- Frame {i+1}: {result[0]['generated_text']}")

# Build context string
context = "Video Frame Descriptions:\n" + "\n".join(frame_descriptions)
print("Frame descriptions generated")


messages = [{
    "role": "user", 
    "content": f"{context}\n\nAnalyze these video frames and determine if this is a deepfake."
    }]

# Prepare inputs
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
).to(device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

# Extract generated text
input_length = inputs["input_ids"].shape[1]
generated_sequence = outputs[0][input_length:]
english_report = tokenizer.decode(generated_sequence, skip_special_tokens=True)

print("\n===== ENGLISH REPORT =====")
print(english_report)

# Clean up GPU memory
del inputs, outputs
torch.cuda.empty_cache()


target_language = "French"  # @param ["None", "French", "Spanish", "German", "Italian", "Portuguese", "Arabic"]

# Supported languages mapping
language_codes = {
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Arabic": "ar"
}



if target_language != "None":
    # Initialize translation model
    lang_code = language_codes[target_language]
    translation_pipeline = pipeline(
        "translation",
        model=f"Helsinki-NLP/opus-mt-en-{lang_code}",
        device=device if device.type == "cpu" else 0
    )
    
    # Translate report in chunks (transformer models have token limits)
    chunk_size = 400
    translated_chunks = []

    for i in range(0, len(english_report), chunk_size):
        chunk = english_report[i:i + chunk_size]
        result = translation_pipeline(chunk)
        translated_chunks.append(result[0]['translation_text'])
    
    translated_report = " ".join(translated_chunks)
    
    print(f"\n===== {target_language.upper()} TRANSLATION =====")
    print(translated_report)
    
    # Clean up translation model
    del translation_pipeline
    torch.cuda.empty_cache()
else:
    print("Skipping translation as requested")

