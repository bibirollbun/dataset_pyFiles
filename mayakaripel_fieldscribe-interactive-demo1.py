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


!pip uninstall -y transformers
!pip install -U transformers



!pip install timm --upgrade
!pip install accelerate


# =================================================================
# FINAL SETUP CELL: Install & Upgrade All Dependencies
# =================================================================

# We use '-U' to force an upgrade to the latest versions.
# We install from the main branch of transformers on GitHub to ensure we have the newest code for Gemma 3n.
#print("Installing and upgrading libraries... This may take a moment.")
#!pip install -q -U git+https://github.com/huggingface/transformers.git


# Now, import everything we'll need for the project
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # We will use AutoModel as it is safer
import ipywidgets as widgets
from IPython.display import display, Markdown
import librosa
import json
from PIL import Image
import tempfile
import numpy as np
import librosa

print("âœ… All libraries installed, upgraded, and imported")


from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.bfloat16,  # Uncomment if you want to use bfloat16 and your hardware supports it
    # device_map="auto",           # Uncomment for automatic device placement
    local_files_only=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

print("âœ… Gemma 3n Model and Tokenizer Loaded")


#import os

#directory = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"
#for root, dirs, files in os.walk(directory):
    #for file in files:
     #   print(os.path.join(root, file))


import ipywidgets as widgets
from IPython.display import display, Markdown

# Create UI components
text_input = widgets.Textarea(
    value='Describe your plant issue here.',
    placeholder='Type your notes or question here',
    description='Notes:',
    layout=widgets.Layout(width='80%', height='100px')
)
run_button = widgets.Button(description="Run Analysis", button_style='success')
output_area = widgets.Output()

print("Please enter your notes or question about the plant.")
display(text_input, run_button, output_area)


def run_analysis(b):
    with output_area:
        output_area.clear_output()
        user_text = text_input.value.strip()
        if not user_text:
            print("â�Œ Error: Please enter some text.")
            return
        print("ğŸ¤– Analyzing your notes with Gemma 3n... Please wait.")

        # Improved prompt
        prompt = f"User question: {user_text}\n\nAI answer:"

        # Prepare inputs and run model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optional: Remove repeated question from result_text if present
        if result_text.startswith(user_text):
            result_text = result_text[len(user_text):].lstrip()

        output_area.clear_output()
        display(Markdown(f"### ğŸŒ¿ FieldScribe AI Analysis:\n\n**Your Input:** `{user_text}`\n\n**Answer:** {result_text}"))

run_button.on_click(run_analysis)

