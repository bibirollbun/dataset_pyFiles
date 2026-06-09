!pip install timm==1.0.17
!pip install transformers==4.53.2


import os
import torch
import json
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import kagglehub


GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b-it")

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH,
    torch_dtype="auto",
    device_map="auto"
)


PROMPT = """
Extract the following details from the resume text and output in valid JSON format:
- Name
- Email
- Phone
- LinkedIn
- GitHub
- Summary
- Work Experience (list of jobs with title, company, dates, responsibilities)
- Education (list of degrees with institute and duration)
- Certifications
- Projects (name + description)
- Skills (grouped by type: Programming, Libraries, Tools, Cloud)
- Languages
- Achievements
- Hobbies

JSON Output:
"""


from tqdm import tqdm

RESUME_FOLDER = "/kaggle/input/resumes-images-datasets/Resumes Datasets/Bing_images/Accountant resumes"
OUTPUT_FOLDER = "/kaggle/working/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Filter image files only
image_files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

for fname in tqdm(image_files, desc="Extracting JSON from resumes"):
    image = Image.open(f"{RESUME_FOLDER}/{fname}").convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": PROMPT}
        ]
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=True
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]
    outs = model.generate(**inputs, max_new_tokens=1024, disable_compile=True)
    json_str = processor.batch_decode(
        outs[:, input_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = {"error": "parse_failed", "output": json_str}

    with open(f"{OUTPUT_FOLDER}/{fname.rsplit('.',1)[0]}.json", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



import pandas as pd

json_files = os.listdir(OUTPUT_FOLDER)[:5]
records = [json.load(open(f"{OUTPUT_FOLDER}/{f}")) for f in json_files]
df = pd.json_normalize(records)
df.head().T  # transpose to see fields vertically

