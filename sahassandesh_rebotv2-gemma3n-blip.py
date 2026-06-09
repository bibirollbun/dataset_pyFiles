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


!pip install -q transformers requests

import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)



def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption



def classify_caption(caption):
    caption = caption.lower()
    if any(word in caption for word in ['banana', 'apple', 'food', 'peel']):
        return "Organic"
    elif any(word in caption for word in ['plastic', 'bottle', 'wrapper']):
        return "Plastic"
    elif any(word in caption for word in ['can', 'tin', 'metal']):
        return "Metal"
    elif any(word in caption for word in ['paper', 'cardboard', 'newspaper']):
        return "Paper"
    else:
        return "Unknown"



reward_map = {
    "Organic": 0.5,
    "Plastic": 1.0,
    "Metal": 1.5,
    "Paper": 0.8,
    "Unknown": 0.0
}

def get_reward(label):
    return reward_map.get(label, 0.0)



# Paste any image URL here
image_url = "https://as2.ftcdn.net/v2/jpg/15/62/79/39/1000_F_1562793986_fntQKk9hQbBDHQh5lXALneZHcfkdj8tA.jpg"

# Download and open image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# Inference steps
print("ğŸ”„ Generating caption...")
caption = generate_caption(image)
print(f"ğŸ“� Caption: {caption}")

label = classify_caption(caption)
reward = get_reward(label)

print(f"ğŸ�·ï¸� Predicted Class: {label}")
print(f"ğŸ’° Reward Earned: â‚¹{reward:.2f}")


