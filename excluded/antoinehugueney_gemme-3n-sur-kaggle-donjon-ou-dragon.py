!pip install timm==1.0.19
!pip install transformers==4.54.1


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
Image(url=IMAGE_URL,height=250,width=250)


from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_URL},
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




