# %%capture
# Install Unsloth package (multimodal model toolkit)
!pip install unsloth

# %%capture
# Install the latest transformers and timm packages needed for Gemma 3N
!pip install --no-deps --upgrade transformers
!pip install --no-deps --upgrade timm



import os
from unsloth import FastModel
import torch

# List of available 4-bit quantized Gemma 3N models (for reference)
fourbit_models = [
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
]  # More models at https://huggingface.co/unsloth

# Load the model and tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",  # You can swap for other models
    dtype=None,            # Auto-detect precision
    max_seq_length=1024,   # Max tokens for long context
    load_in_4bit=True,     # Use 4-bit quantization to save memory
    full_finetuning=False, # Set True if you want to finetune the model
    # token="hf_..."      # Optional, if gated models require a HuggingFace token
)



import torch
import gc

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 512

def do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens=128):
    # Tokenize input with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate output (no streamer as in the initial Notebook)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True
    )

    # Decode just the new generated tokens (excluding prompt)
    generated_text = tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    )

    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()

    return generated_text


messages = [
        {
    "role": "system",
    "content": [{ "type" : "text",
                  "text" : "Answer briefly, be laconic if possible." }]
    },
    {
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : "Who was Rajendra Prasad?" }]
}]
output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 16)
print(output)



from IPython.display import display, Markdown

def colorize_text(text):
    for word, color in zip(["Question", "Answer","Execution time"], ["blue", "red", "green"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text



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
    _end = time()
    formated_output = f"Question: {user_input}\nAnswer: {output}\nExecution time: {np.round(_end-_start, 2)} sec."
    display(Markdown(colorize_text(formated_output)))



run_query("Who was Father of the Nation?", max_new_tokens=16)


run_query("Who was Nethaji?", max_new_tokens=16)


from IPython.display import display, Markdown
import numpy as np
from time import time

# Your colorize function (fixed syntax)
def colorize_text(text):
    for word, color in zip(["Question", "Answer", "Execution time"], ["blue", "red", "green"]):
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

    formatted_output = f"Question: {question_text}\nAnswer: {output}\nExecution time: {elapsed} sec."
    display(Markdown(colorize_text(formatted_output)))

# Example usage:
img_path = "/kaggle/input/plantdisease/PlantVillage/Pepper__bell___Bacterial_spot/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG"



run_image_query(img_path, "What disease does this leaf have? Please provide a clear diagnosis.", max_new_tokens=512)


!pip install gTTS


from gtts import gTTS
import IPython.display as ipd
import tempfile
from PIL import Image  # Add this import
from IPython.display import display

image = Image.open(img_path)

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

    # Display colored text as before
    formatted_output = f"Question: {question_text}\nAnswer: {output}\nExecution time: {elapsed} sec."
    display(Markdown(colorize_text(formatted_output)))

    # Convert text to speech
    tts = gTTS(output, lang='en')

    # Save to a temporary file and play
    with tempfile.NamedTemporaryFile(suffix=".mp3") as fp:
        tts.save(fp.name)
        return ipd.Audio(fp.name, autoplay=True)

# Example usage:
audio_player = run_image_query_audio(img_path, "What disease does this leaf have? Please provide a clear diagnosis.")
display(audio_player)


