!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


from time import time
import kagglehub
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoProcessor, AutoModelForImageTextToText


GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")


prompt = """What is the France capital?"""
input_ids = processor(text=prompt, 
                      return_tensors="pt").to(model.device, 
                                              dtype=model.dtype)

outputs = model.generate(**input_ids, 
                         max_new_tokens=32, 
                         disable_compile=True)
text = processor.batch_decode(
    outputs,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)
print(text[0])


# Install required libraries quietly to maintain a clean output
!pip install timm --upgrade --quiet
!pip install accelerate --quiet
!pip install git+https://github.com/huggingface/transformers.git --quiet
!pip install kagglehub --quiet

# Import necessary components
import kagglehub
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import warnings

# Suppress warnings for notebook
warnings.filterwarnings("ignore")

print("Environment configured. All systems are go.")
# --- Model Loading ---
print("Loading Gemma-3N model from Kaggle Hub...")
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

# The processor handles the complex task of converting text and images into the numerical format the model understands.
processor = AutoProcessor.from_pretrained(GEMMA_PATH)

# The model itself. We specify our hardware optimization choices here.
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH, 
    torch_dtype="auto", 
    device_map="auto"
)

print("✅ Gemma-3N model and processor loaded successfully. Ready for analysis.")



# --- Test 1: Simple Q&A ---
prompt = "What is the capital of France, and what is its most famous landmark?"

input_ids = processor(text=prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**input_ids, max_new_tokens=64)
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Prompt: {prompt}")
print(f"Gemma-3N: {text.split(prompt)[-1].strip()}")



# --- Test 2: Python Code Generation ---
prompt = "Write a robust Python function to find the nth Fibonacci number using memoization for efficiency."

input_ids = processor(text=prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**input_ids, max_new_tokens=150)
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Prompt: {prompt}")
print("\nGemma-3N's Code:\n")
# We split to only show the generated code block for clarity
generated_code = text.split('```python')[-1].split('```')[0]
print(generated_code.strip())



# Load a complex scene for deep VQA
image_url = "https://images.stockcake.com/public/6/3/8/638fd805-2862-475f-8c56-4931c4c23c38_large/busy-kitchen-scene-stockcake.jpg" # A busy professional kitchen
image = Image.open(requests.get(image_url, stream=True).raw)

print("Image loaded for Advanced VQA.")
image



# --- Test 3: Basic vs. Specific Prompts
image_token = processor.tokenizer.special_tokens_map.get("image_token", "<|image|>")

prompts = [
    f"{image_token}\nDescribe this image.",
    f"{image_token}\nIdentify three specific actions being performed by the people in this image.",
    f"{image_token}\nWhat type of food is being prepared on the metal counter in the center?"
]

for prompt in prompts:
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"Prompt: {prompt}")
    print(f"Gemma-3N: {text.split(prompt)[-1].strip()}\n" + "-"*50)



# --- Test 4: Inference and Spatial Awareness ---
prompt = f"{image_token}\nBased on the attire of the individuals and the equipment, what is the likely setting? What can you infer about the relationship between the people?"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Prompt: {prompt}")
print(f"Gemma-3N: {text.split(prompt)[-1].strip()}")



from PIL import Image
from IPython.display import display

image_path = "/kaggle/input/gemma-image/Tea_processing_chart.png"
image = Image.open(image_path).convert("RGB")

print("✅ Image loaded from local file.")
display(image)



image_token = processor.tokenizer.special_tokens_map.get("image_token", "<|image|>")

prompts = [
    f"{image_token}\nDescribe this image.",
    f"{image_token}\nIdentify three specific actions being performed by the people in this image.",
    f"{image_token}\nWhat type of food is being prepared on the metal counter in the center?"
]

for prompt in prompts:
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    print(f"Prompt: {prompt}")
    print(f"Gemma-3N: {text.split(prompt)[-1].strip()}\n" + "-"*50)



image_path = "/kaggle/input/gemma-image/Tea_processing_chart.png"
image = Image.open(image_path).convert("RGB")

display(image)

prompt = f"{image_token}\nExtract all visible text from this image, and list it line by line."

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Prompt: {prompt}")
print("Gemma-3N Response:\n", text.split(prompt)[-1].strip())



followup_prompt = f"{image_token}\nThis image shows the tea processing flow. Which of these steps could realistically be automated using machines, sensors, or AI systems? Explain how."

inputs = processor(text=followup_prompt, images=image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300)
text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Output
print(f"Prompt: {followup_prompt}")
print("\nGemma-3N Response:\n", text.split(followup_prompt)[-1].strip())





