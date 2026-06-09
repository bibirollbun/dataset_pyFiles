!pip install -q timm --upgrade
!pip install -q accelerate
!pip install -q --upgrade transformers
!pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q git+https://github.com/huggingface/transformers.git

#!pip uninstall -y transformers
#!pip install -q git+https://github.com/huggingface/transformers.git


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import transformers
import torch
import gc
try:
    del model
    torch.cuda.empty_cache()
    gc.collect()
except:
    pass
# torch.cuda.reset_peak_memory_stats()
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import json
import re
import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["DISABLE_TORCH_COMPILE"] = "1"
import kagglehub
import requests
from io import BytesIO
from IPython.display import display, HTML
import base64

from transformers import pipeline
import torchaudio

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",  # or medium/large
    return_timestamps=False
)

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def get_best_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_gpu = None
    max_free_mem = 0

    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.memory_stats(i)
        free_mem = torch.cuda.get_device_properties(i).total_memory - stats["allocated_bytes.all.current"]
        print(f"GPU {i} free memory: {free_mem / (1024**3):.2f} GB")

        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_gpu = i

    print(f"Best GPU selected: {best_gpu} (Free: {max_free_mem / (1024**3):.2f} GB)")
    return torch.device(f"cuda:{best_gpu}")

device = get_best_gpu()
# device = "cpu" # GPU had some compilation issues, hence CPU for avoiding errors.

model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, 
                                                    device_map=None, 
                                                    torch_dtype="auto").to(device)

processor = AutoProcessor.from_pretrained(GEMMA_PATH)


print(type(model))


# Testing model load
prompt = """It was a dark and stormy night. """
input_ids = processor(text=prompt, return_tensors="pt").to(model.device, dtype=torch.float16)
outputs = model.generate(**input_ids, max_new_tokens=256, disable_compile=True)
text = processor.batch_decode(
    outputs,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False 
)
print(text[0])


def create_unified_prompt(input_content: str, location_hint: str = "Unknown"):
    return f""" 
                You are a multilingual travel assistant running completely offline. Based on the input provided (image, audio, or text), perform the following steps carefully and return a structured JSON.
                
                ---

                TASKS:
                1. Automatically determine the input type: image, speech, or text.
                2. If the input is an **image**:
                   - Extract any visible text (OCR) and translate it into English.
                   - Describe what the image contains (e.g., a road sign, a menu).
                3. If the input is **spoken** or **written**:
                   - Transcribe it (if needed), translate to English, and interpret its meaning.
                4. In all cases:
                   - Identify what kind of object it is (e.g., signboard, menu, spoken phrase).
                   - Suggest a polite or practical reply if appropriate.
                   - Offer at least one cultural or etiquette tip related to the input or region.
                
                Context:
                Region hint: {location_hint}
                
                ---

                Please respond **strictly enclosed in triple backticks and labeled `json`**, like this:

                
                ---
                
                ```json
                {{
                  "input_type": "<image / speech / text>",
                  "description": "<brief description of the content>",
                  "extracted_text": [
                    {{ "original": "<text>", "translation": "<translation>" }}
                  ],
                  "object_type": "<menu, sign, phrase, etc.>",
                  "suggested_reply": "<if relevant>",
                  "cultural_tips": ["<tip1>", "<tip2>"]
                }}
                ```
                Now analyze this input:
                {input_content}
            """

def detect_input_type(input_data):
    if isinstance(input_data, str) and input_data.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    elif isinstance(input_data, str) and input_data.endswith((".wav", ".mp3")):
        return "audio"
    else:
        return "text"

def display_resized_image(image, width_px=300):
    """Display PIL image resized to a fixed width in a notebook cell."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    html = f'<img src="data:image/png;base64,{img_b64}" width="{width_px}px"/>'
    display(HTML(html))


def format_input(input_type, raw_input):
    """
    Formats input data for image, audio (simulated), or text.
    Supports both local paths and URLs for images. Displays image if loaded.
    """
    if input_type == "image":
        try:
            if raw_input.startswith("http://") or raw_input.startswith("https://"):
                response = requests.get(raw_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(raw_input).convert("RGB")

            print("Image loaded successfully. Preview below:")
            display_resized_image(image, width_px=300)

            return image

        except Exception as e:
            raise ValueError(f"Failed to load image from {raw_input}: {e}")

    elif input_type == "text":
        return raw_input

    elif input_type == "audio":
        try:
            if raw_input.startswith("http://") or raw_input.startswith("https://"):
                response = requests.get(raw_input)
                response.raise_for_status()
                with open("/tmp/temp.wav", "wb") as f:
                    f.write(response.content)
                # audio_path = "/kaggle/input/tourist-support/korean_tourist_help.wav"
            else:
                audio_path = raw_input

            print(f"Transcribing audio: {audio_path}")
            transcript = asr_pipeline(audio_path)["text"]
            print(f"Transcription: {transcript}")
            return transcript

        except Exception as e:
            raise ValueError(f"Failed to transcribe audio from {raw_input}: {e}")

    else:
        raise ValueError(f"Unsupported input type: {input_type}")


def run_gemma_inference(input_type, formatted_input, prompt):
    """
    Runs inference on Gemma 3n using image or text input.
    """
    if input_type == "image":
        image_token = processor.tokenizer.image_token
        # print(f"Image token used: {image_token}")
        if image_token not in prompt:
            # print("Image token not found in prompt, adding it")
            prompt = f"{image_token}\n{prompt}"
        inputs = processor(text=prompt, images=formatted_input, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    
    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def parse_structured_json(output_text):
    """
    Extract and parse valid JSON from model output, skipping placeholder templates.
    Supports both fenced (```json ... ```) and raw { ... } blocks.
    """
    json_blocks = re.findall(r"```json(.*?)```", output_text, re.DOTALL)

    # Also consider unfenced JSON blocks if no fenced ones are found
    if not json_blocks:
        json_blocks = re.findall(r"(\{[\s\S]{10,10000}?\})", output_text)

    for block in json_blocks:
        block_cleaned = block.strip()

        # Skip placeholder templates
        if any(ph in block_cleaned for ph in [
            "<text>", "<translation>", "<menu", "<brief",
            "<tip1>", "<image", "<speech", "<if relevant>"
        ]):
            continue

        try:
            return json.loads(block_cleaned)
        except json.JSONDecodeError:
            continue  # Try next

    return {
        "error": "JSON parsing failed.",
        "message": "No valid JSON block found or all blocks had placeholders.",
        "raw_extract": json_blocks[0][:500] if json_blocks else output_text[:500]
    }

def run_travel_translator(input_data, region_hint="Unknown"):
    """
    Runs the complete offline travel translator pipeline:
    - Detects input type (image, audio, text)
    - Formats the input
    - Creates the prompt
    - Sends it to Gemma 3n
    - Parses structured response
    - Displays everything step by step
    
    Parameters:
        input_data: File path or text string
        region_hint: Optional location hint (e.g., "Japan", "Thailand")
    
    Returns:
        dict: Structured JSON response from the assistant
    """
    print("Starting Travel Translator...\n")

    input_type = detect_input_type(input_data)
    print(f"Detected input type: {input_type}")

    formatted_input = format_input(input_type, input_data)

    prompt_input = "<image>" if input_type == "image" else formatted_input
    prompt = create_unified_prompt(prompt_input, location_hint=region_hint)

    if input_type == "image" and "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    print("Running inference with Gemma 3n...\n")
    output_text = run_gemma_inference(input_type, formatted_input, prompt)
    # print("Raw Output from Gemma:\n", output_text)

    structured = parse_structured_json(output_text)
    print("\nStructured JSON Response:")
    print(json.dumps(structured, indent=2, ensure_ascii=False))


# Text input (phrase translation)
run_travel_translator("How do I say thank you in Korean?", region_hint="South Korea")


# Simulated audio (Korean)
run_travel_translator("/kaggle/input/tourist-support/korean_tourist_help.wav", region_hint="Korea")


# Uploaded image (e.g. Japanese road sign)
# link = "https://inhabitat.com/wp-content/blogs.dir/1/files/2017/04/Swindon-Magic-Roundabout2.jpg"
# link = "https://tabimaniajapan.com/wp-content/uploads/2023/12/STOP_SIGN_JAPAN.jpg"
link = "https://d1gymyavdvyjgt.cloudfront.net/drive/images/uploads/headers/ws_cropper/1_0x0_790x520_0x520_1_0x0_790x520_0x520_italian-road-signs-header.jpg"
run_travel_translator(link, region_hint="Italy")


# Text input (phrase translation)
run_travel_translator("How do I approach locals for direction in Japan ?", region_hint="Japan")


link = "https://tabimaniajapan.com/wp-content/uploads/2023/12/STOP_SIGN_JAPAN.jpg"
run_travel_translator(link, region_hint="Japan")

