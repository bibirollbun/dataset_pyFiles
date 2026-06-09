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


pip install accelerate


pip install requests



pip install git+https://github.com/huggingface/transformers.git --upgrade



!pip install git+https://github.com/huggingface/pytorch-image-models.git



from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import kagglehub

print("\nStep 1: Downloading model using kagglehub...")

# Step 1: Download the model first
path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")

print("Step 2: Path to model files:", path)

print("\nStep 3: Loading tokenizer and model...")

# Step 2: Load tokenizer and model AFTER downloading
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"}
)

print("Tokenizer and model loaded successfully.")




from transformers import GenerationConfig

prompt = "Tell me some interesting facts about the taj mahal."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generation_config = GenerationConfig(
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    max_new_tokens=150
)

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))




pip install git+https://github.com/salesforce/LAVIS.git  # contains BLIP-2



from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import requests
from io import BytesIO

# Load image from the web
url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
 
# Check for valid response and correct content-type
if response.status_code == 200 and "image" in response.headers["Content-Type"]:
    image = Image.open(BytesIO(response.content)).convert("RGB")
else:
    raise ValueError("Image could not be fetched or is not a valid image.")


# Load model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = model.to("cpu")  # Ensure it's on CPU

# Process and caption image
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}

# Generate caption
generated_ids = model.generate(**inputs, max_new_tokens=50)  # limit tokens to speed up on CPU
caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Output the caption
print("ğŸ–¼ï¸� BLIP-2 Caption:", caption)





from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import kagglehub

# Load Gemma model (assuming already downloaded with kagglehub)
path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")

tokenizer = AutoTokenizer.from_pretrained(path)
gemma_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to("cpu")

# Create a prompt using caption
tour_guide_prompt = f"You are a tour guide. Tell a tourist about {caption}. Give interesting facts, history, and cultural significance."

inputs = tokenizer(tour_guide_prompt, return_tensors="pt").to("cpu")
generation_config = GenerationConfig(temperature=0.7, top_p=0.9, do_sample=True, max_new_tokens=200)

# Generate explanation
outputs = gemma_model.generate(**inputs, generation_config=generation_config)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nğŸ—£ï¸� Virtual Tour Guide Says:\n", response)






