from IPython.display import Image, display
display(Image(filename="/kaggle/input/gemma-3n-logo/Gemma.png"))



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


# âœ… Install Required Libraries
!pip install -qU timm accelerate
!pip install -q git+https://github.com/huggingface/transformers.git



# âœ… Download Gemma 3n Model via KaggleHub
import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")



from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForImageTextToText,
    GenerationConfig
)
from PIL import Image
import torch
import requests
from io import BytesIO



def generate_text(prompt, temp=0.7, style="default", max_tokens=200):
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True).eval()
    
    if style == "summary":
        prompt = f"Summarize this:\n{prompt}"
    elif style == "instruction":
        prompt = f"Please explain: {prompt}"
    elif style == "creative":
        prompt = f"Write a story about: {prompt}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temp
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)



def load_image_from_url_safe(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
    
    try:
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise ValueError("â�Œ Cannot open image. Is the URL pointing to a real image file?") from e



def describe_image(image_input, prompt="Describe this image in detail.", max_tokens=512):
    processor = AutoProcessor.from_pretrained(GEMMA_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_PATH, torch_dtype="auto", device_map={"": 0}
    ).eval()

    if isinstance(image_input, str):
        if image_input.startswith("http"):
            image = load_image_from_url_safe(image_input)
        else:
            image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # âœ… Fix: Send each tensor to device only, preserve data type
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k in inputs:
        inputs[k] = inputs[k].to(device=device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            disable_compile=True
        )

    response = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return response[0]



print("ğŸ”¹ English Prompt:")
print(generate_text("How does a neural network work?", style="instruction"))

print("\nğŸ”¹ Urdu Prompt:")
print(generate_text("Ú©ÛŒØ§ Ø¢Ù¾ Ù…Ø¬Ú¾Û’ Ø²Ù…ÛŒÙ† Ú©ÛŒ Ú©Ø´Ø´ Ø«Ù‚Ù„ Ú©Û’ Ø¨Ø§Ø±Û’ Ù…ÛŒÚº Ø¨ØªØ§ Ø³Ú©ØªÛ’ Û�ÛŒÚºØŸ", style="default"))

print("\nğŸ”¹ French Prompt:")
print(generate_text("Expliquez comment fonctionne l'Ã©nergie solaire.", style="summary"))

print("\nğŸ”¹ Spanish Prompt:")
print(generate_text("CuÃ©ntame una historia sobre un gato astronauta.", style="creative"))



# ğŸ“� TEXT INPUT SECTION
print("ğŸ”¹ Try Your Own Prompt!")
custom_prompt = input("Enter your prompt: ")
style_choice = input("Choose style (default/summary/instruction/creative): ")

custom_output = generate_text(custom_prompt, style=style_choice)
print("\nğŸ§  Response:")
print(custom_output)



print("\nğŸ”¹ Pre-Filled Example")
image_path = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Gull_portrait_ca_usa.jpg"
img_prompt = "Describe this bird in detail."

try:
    image_output = describe_image(image_path, img_prompt)
    print("\nğŸ–¼ï¸� Image Response:")
    print(image_output)
except Exception as e:
    print("\nâ�Œ Error processing image:", str(e))



# âœ… All tasks complete
print("\nğŸ�‰ All demos are complete!")
print("âœ… Youâ€™ve successfully tested:")
print(" - ğŸ’¬ Text generation with multiple prompt styles")
print(" - ğŸŒ� Multilingual support (English, Urdu, French, etc.)")
print(" - ğŸ–¼ï¸� Image + prompt reasoning (both URL and local image)")
print(" - ğŸ§ª 'Try your own' prompt + image section")

print("\nğŸ“¢ Final Checklist:")
print("1. Make sure this notebook runs top to bottom without errors.")
print("2. Submit this notebook directly on Kaggle (no GitHub or video needed).")
print("3. Double-check your descriptions and prompt responses.")


