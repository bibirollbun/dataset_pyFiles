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


# Install required libraries
!pip install peft accelerate bitsandbytes

from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load the configuration for the fine-tuned model
model_id = "/kaggle/input/gemma2-lora-hindichat/transformers/unsloth-4bit/1/lora_model"
config = PeftConfig.from_pretrained(model_id)

# Load the base model and the fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, model_id)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Define the prompt template (same as used during training)
hindi_alpaca_prompt = """नीचे एक निर्देश दिया गया है जो एक कार्य का वर्णन करता है, और इसके साथ एक इनपुट है जो अतिरिक्त संदर्भ प्रदान करता है। एक उत्तर लिखें जो अनुरोध को सही ढंग से पूरा करता हो।

### निर्देश:
{}

### इनपुट:
{}

### उत्तर:
{}"""

# Define new prompts for inference
prompts = [
    hindi_alpaca_prompt.format("भारत के स्वतंत्रता संग्राम में महात्मा गांधी की भूमिका क्या थी?", "", ""),
    hindi_alpaca_prompt.format("सौर मंडल में कौन सा ग्रह सबसे छोटा है?", "", ""),
    hindi_alpaca_prompt.format("पानी का रासायनिक सूत्र क्या है?", "", ""),
    hindi_alpaca_prompt.format("हिमालय पर्वत श्रृंखला की विशेषताएँ बताइए।", "", ""),
    hindi_alpaca_prompt.format("प्रकाश संश्लेषण की प्रक्रिया क्या है?", "", "")
]

# Tokenize the prompts
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

# Generate responses
outputs = model.generate(**inputs, max_new_tokens=512,do_sample=True,
    temperature=0.7,top_k=50, use_cache=True)

# Decode and print the responses
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, response in enumerate(responses):
    print(f"प्रश्न {i+1}: {prompts[i].split('### निर्देश:')[1].split('### इनपुट:')[0].strip()}")
    print(f"उत्तर: {response.split('### उत्तर:')[1].strip()}")
    print("-" * 50)


