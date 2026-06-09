!pip install unsloth

!pip install --no-deps --upgrade transformers
!pip install --no-deps --upgrade timm


from unsloth import FastModel
import torch
import os

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] 

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None, 
    max_seq_length = 1024, 
    load_in_4bit = True,  
    full_finetuning = False
)


from transformers import TextStreamer
import gc

def do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 128):
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, 
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True
    )

    generated_text = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    )
    
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text


import torch
import gc

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 512


import pandas as pd
df = pd.read_csv('/kaggle/input/crop-recommendation-dataset/Crop_recommendation.csv')
df.head()


import json

formatted_data = []
for _, row in df.iterrows():
    prompt = (
        f"N: {row['N']}, P: {row['P']}, K: {row['K']}, temperature: {row['temperature']:.2f}, "
        f"humidity: {row['humidity']:.2f}, pH: {row['ph']:.2f}, rainfall: {row['rainfall']:.2f}"
    )
    response = f"Recommended crop: {row['label']}"
    formatted_data.append({"prompt": prompt, "response": response})

# Save as JSONL for unsloth
with open("crop_recommendation_train.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item) + "\n")


from datasets import load_dataset

my_text_dataset = load_dataset("json", data_files="/kaggle/working/crop_recommendation_train.jsonl")["train"]



print(my_text_dataset[0])


def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    responses = examples["response"]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix("<bos>")
        for p, r in zip(prompts, responses)
    ]
    return { "text": texts }



dataset = my_text_dataset.map(formatting_prompts_func, batched=True)



model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

trainer.train()


from unsloth.chat_templates import get_chat_template

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant that recommends crops based on soil and climate."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "N: 80, P: 38, K: 67, temperature: 32.01, humidity: 58, pH: 6.90, rainfall: 302.93"}]
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, 
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 512, 
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant that recommends crops based on soil and climate."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "N: 80, P: 38, K: 67, temperature: 32.01, humidity: 58, pH: 6.90, rainfall: 302.93"}]
    }
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, 
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 1024, 
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


from PIL import Image
import matplotlib.pyplot as plt

image = "/kaggle/input/leaf-photo/leaf1.jpg"

img = Image.open(image)

plt.imshow(img)
plt.axis('off')  
plt.show()


messages = [
        {
    "role": "system",
    "content": [{ "type" : "text",
                  "text" : "Answer briefly, be laconic if possible." }]
    },
    {
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : "Who is Lord Krishna?" }]
}]
output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 64)
print(output)


import numpy as np
from time import time
def run_query(user_input, max_new_tokens=128, model=model, tokenizer=tokenizer):
    _start = time()
    messages = [
        {
    "role": "system",
    "content": [{ "type" : "text",
                  "text" : "Answer briefly, be laconic if possible." }]
    },
    {
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : user_input }]
    }]
    output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = max_new_tokens)
    display(output)


run_query("Who is Narendra Modi?", max_new_tokens=16)


from IPython.display import display, Markdown
import numpy as np
from time import time

def colorize_text(text):
    for word, color in zip(["Question", "Answer"], ["red", "green"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

# Run query for image + text input
def run_image_query(image_path, question_text, max_new_tokens=128, model=model, tokenizer=tokenizer):
    start = time()

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Answer briefly, be laconic if possible."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text}
            ]
        }
    ]

    output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens=max_new_tokens)

    end = time()
    elapsed = np.round(end - start, 2)

    formatted_output = f"Answer: {output}."
    display(Markdown(colorize_text(formatted_output)))

# Example usage:
image_path = "/kaggle/input/leaf-photo/leaf1.jpg"


run_image_query(image_path, "What leaf is this? What disease does this leaf have? Please provide a clear diagnosis.", max_new_tokens=128)


run_image_query(image_path, "यह किस पौधे की पत्ती है? इस पत्ती को कौन सी बीमारी है? कृपया एक स्पष्ट निदान दें।", max_new_tokens=128) 


!pip install gTTS


from gtts import gTTS
import IPython.display as ipd
import tempfile
from PIL import Image  
from IPython.display import display
# from langdetect import detect

image = Image.open(image_path)

display(image)

def run_image_query_audio(image_path, question_text, max_new_tokens=256, model=model, tokenizer=tokenizer):
    start = time()

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Answer briefly, be laconic if possible."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question_text}
            ]
        }
    ]

    output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens=max_new_tokens)

    end = time()
    elapsed = np.round(end - start, 2)

    formatted_output = f"Question: {question_text}\nAnswer: {output}"
    display(Markdown(colorize_text(formatted_output)))

    tts = gTTS(output, lang='en')

    with tempfile.NamedTemporaryFile(suffix=".mp3") as fp:
        tts.save(fp.name)
        return ipd.Audio(fp.name, autoplay=True)

audio_player = run_image_query_audio(image_path, "What disease does this leaf have? Please provide a clear diagnosis.")
display(audio_player)


audio_player = run_image_query_audio(image_path, "ఈ ఆకు ఏ రకమైన వ్యాధితో ఉంది? దయచేసి స్పష్టమైన నిర్ధారణను ఇవ్వండి." )
display(audio_player)

