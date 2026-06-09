%%capture
# Install Unsloth
!pip install unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


import os
from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # Or "unsloth/gemma-3n-E2B-it"
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
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
                  "text" : "Who was George Washington?" }]
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
    
    


run_query("Who was George Washington?", max_new_tokens=16)


run_query("When was the WWI?", max_new_tokens=16)


run_query("When was The War of 30 years?", 16)


run_query("Who was JFK?", 32)


run_query("Who was Xenophon?", 32)


run_query("Who was Meiji?", 32)


run_query("Name 3 songs by Beach Boys.", 32)


run_query("Who was Maradona?", 32)


run_query("When was released first time the show 'Dallas' in US?", 32)


run_query("When was released the first Star Wars movie?", 32)


run_query("Name 3 most famous albums of The Beatles.", 32)


run_query("What is the real name of Madonna?", 32)


jfk_link = "https://media.cnn.com/api/v1/images/stellar/prod/231120172101-lead-image-john-f-kennedy-life-career.jpg"
user_input = "Describe this picture."

_start = time()
 
messages = [
    {"role": "system",
    "content": [{ "type" : "text",
                  "text" : "Answer briefly, be laconic if possible." }]
    },
    {"role" : "user",
    "content": [
        { "type": "image", "image" : jfk_link },
        { "type": "text",  "text" : user_input}
    ]
}]
       
output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 32)
_end = time()
formated_output = f"Question: {user_input}\nAnswer: {output}\nExecution time: {np.round(_end-_start, 2)} sec."
display(Markdown(colorize_text(formated_output)))
    


from IPython.display import Audio, display
Audio("https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3")


!wget -qqq https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3 -O audio.mp3


audio_file = "audio.mp3"
user_input = "What is this audio about?"
_start = time()
messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "text",  "text" :  user_input}
    ]
}]
output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 32)
_end = time()
formated_output = f"Question: {user_input}\nAnswer: {output}\nExecution time: {np.round(_end-_start, 2)} sec."
display(Markdown(colorize_text(formated_output)))


moon_link = "https://ichef.bbci.co.uk/ace/standard/976/cpsprodpb/F574/production/_107563826_astronautedwine.aldrin-onthe-moon.jpg"


user_input = "What is this audio and images about? How are they related?" 
_start = time()
messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "image", "image" : jfk_link },
        {"type": "image", "image" : moon_link},
        { "type": "text",  "text" : user_input}
    ]
}]
output = do_gemma_3n_inference(model, tokenizer, messages, max_new_tokens = 256)
_end = time()
formated_output = f"Question: {user_input}\nAnswer: {output}\nExecution time: {np.round(_end-_start, 2)} sec."
display(Markdown(colorize_text(formated_output)))

