!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


import kagglehub

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)
prompt = "Why are there so many Geese on Kaggle?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)


print(result)


from IPython.display import Image
IMAGE_URL="https://storage.googleapis.com/kaggle-media/competitions/question_goose.png"
IMAGE_URL2='https://media.gettyimages.com/id/1638757294/photo/portrait-of-a-businessman-using-laptop-on-a-coffee-shop.jpg?s=612x612&w=gi&k=20&c=8GuzuH7_0o_TYXXEECeSm_122AFqKeWgrGDbX6T1J0E='
IMAGE_URL3='https://img.freepik.com/premium-photo/one-worried-man-front-laptop-with-hand-his-head-looking-display-with-stressed-worried-expression-bad-problems-work-online-security-hack-smart-working-modern-people-indoor_425263-8165.jpg'
Image(url=IMAGE_URL,height=250,width=250),Image(url=IMAGE_URL2,height=250,width=250),Image(url=IMAGE_URL,height=250,width=250)


from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL2},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
text = processor.batch_decode(
    outputs[:, input_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)


print(text[0])


from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

# Define the message: asking model to compare productivity of two individuals in images
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image":  IMAGE_URL2},
            {"type": "image", "image":  IMAGE_URL3},
            {
                "type": "text",
                "text": "Compare the two individuals in these images. Who seems more productive and why?"
            }
        ]
    }
]

# Apply chat template and generate response
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
text = processor.batch_decode(
    outputs[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]

# Print answer
print(text)












#fiend out the images here

