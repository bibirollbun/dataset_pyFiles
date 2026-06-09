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



# =================================================================
# CELL 1: SETUP AND IMPORTS (WITH SPEECH-TO-TEXT)
# =================================================================

# Install the powerful whisper library for speech recognition
!pip install -q openai-whisper

import os
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import torch
import ipywidgets as widgets
import io
import whisper # <-- Import whisper
import tempfile # <-- For handling the audio file
from PIL import Image
from IPython.display import display, Markdown
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

print("âœ… Libraries (including Whisper for audio) imported. Dynamo disabled.")


#!ls /kaggle/input/


#!ls /kaggle/input/paligemma-2/
#!ls /kaggle/input/paligemma-2/transformers/
#!ls /kaggle/input/paligemma-2/transformers/paligemma2-3b-mix-448/
#!ls /kaggle/input/paligemma-2/transformers/paligemma2-3b-mix-448/1


# =================================================================
# CELL 2: LOAD THE MODEL (FINAL-FINAL ROBUST VERSION)
# =================================================================

import os

model_path = "/kaggle/input/paligemma-2/transformers/paligemma2-3b-mix-448/1"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_path,
    #torch_dtype=torch.bfloat16,
    #device_map="auto",
    local_files_only=True,
).eval()

processor = PaliGemmaProcessor.from_pretrained(
    model_path,
    local_files_only=True
)

print("âœ… PaliGemma Model and Processor Loaded")


# =================================================================
# CELL 3: THE COMPLETE INTERACTIVE APPLICATION (WITH PRE-LOADED DEMO)
# =================================================================

# --- Dynamo Fix ---
import torch._dynamo
torch._dynamo.disable()
# ------------------

# Load the whisper model once
whisper_model = whisper.load_model("tiny.en")
print("âœ… Whisper Speech-to-Text model loaded.")

# --- UI COMPONENTS ---
# We now add a second button for the demo
prompt_input = widgets.Text(value='Identify the disease on this plant leaf.', description='Prompt:', layout=widgets.Layout(width='80%'))
image_uploader = widgets.FileUpload(accept='image/*', description='Upload Image:')
audio_uploader = widgets.FileUpload(accept='audio/*', description='Upload Audio Note:')
run_custom_button = widgets.Button(description="Run Analysis on Uploaded Files", button_style='success')
run_demo_button = widgets.Button(description="â–º Load & Run Pre-Loaded Demo", button_style='info') # New demo button
output_area = widgets.Output()


# --- ANALYSIS FUNCTION ---
# This function is now more flexible. It takes image and audio data as arguments.
def run_analysis_pipeline(image_data, audio_data, question_text):
    with output_area:
        output_area.clear_output()

        print("ğŸ¤– Transcribing audio notes...")
        # Save audio to a temporary file for whisper to process
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()
            transcribed_text = whisper_model.transcribe(temp_audio.name)['text']
        
        print(f"ğŸ�¤ Farmer's Notes (Transcribed): '{transcribed_text}'")
        print("ğŸ¤– Analyzing with PaliGemma...")

        # Convert image data to a usable Image object
        image = Image.open(io.BytesIO(image_data))
        
        # Create the full prompt
        full_prompt = f"<image> {question_text}\nSpoken notes from the farmer: {transcribed_text}"
        
        # Run the model
        inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        
        # Clean the output
        text_part_of_prompt = f"{question_text}\nSpoken notes from the farmer: {transcribed_text}"
        raw_output_text = processor.decode(outputs[0], skip_special_tokens=True)
        final_answer = raw_output_text[len(text_part_of_prompt):].strip()

        # Display the final, clean result
        output_area.clear_output()
        display(Markdown(f"""
### ğŸŒ¿ FieldScribe AI Analysis:

**Your Question:**
`{question_text}`

**Farmer's Spoken Notes:**
*`{transcribed_text}`*

**AI Answer:**
**{final_answer}**
"""))


# --- BUTTON HANDLERS ---
# Function for the "Run Analysis on Uploaded Files" button
def handle_custom_run(b):
    if not image_uploader.value or not audio_uploader.value:
        with output_area:
            output_area.clear_output()
            print("â�Œ Please upload both an image and an audio file to run a custom analysis.")
        return
    # Get data from the uploader widgets
    image_data = image_uploader.value[0]['content']
    audio_data = audio_uploader.value[0]['content']
    question = prompt_input.value
    run_analysis_pipeline(image_data, audio_data, question)

# Function for the "Load & Run Pre-Loaded Demo" button
def handle_demo_run(b):
    with output_area:
        output_area.clear_output()
        print("ğŸš€ Loading pre-loaded demo assets...")
    
    # Define the paths to your demo files
    demo_image_path = "/kaggle/input/fieldscribe-assets/tomato_leaf.jpg"
    demo_audio_path = "/kaggle/input/fieldscribe-assets/farmer_note.wav"
    
    # Read the file data from the paths
    with open(demo_image_path, "rb") as f:
        image_data = f.read()
    with open(demo_audio_path, "rb") as f:
        audio_data = f.read()
        
    question = prompt_input.value
    # Run the main analysis pipeline with the demo data
    run_analysis_pipeline(image_data, audio_data, question)


# --- FINAL SETUP ---
# Connect the buttons to their respective handler functions
run_custom_button.on_click(handle_custom_run)
run_demo_button.on_click(handle_demo_run)

# Display the UI components in a structured way
display(
    Markdown("### Run a Pre-Loaded Demo"),
    run_demo_button,
    Markdown("---"),
    Markdown("### Or, Run with Your Own Files"),
    prompt_input,
    image_uploader,
    audio_uploader,
    run_custom_button,
    output_area
)

