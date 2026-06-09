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


%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
# 1) Upgrade the HF hub to at least 0.34.0
!pip install --upgrade "huggingface-hub>=0.34.0,<1.0"

# 2) (Re)install the transformers Git main so it sees the new hub version
!pip install --no-deps git+https://github.com/huggingface/transformers.git

# 3) Upgrade timm for Gemma 3N
!pip install --no-deps --upgrade timm

# 4) Now install Unsloth (no deps is fine since Huggingface deps are satisfied)
!pip install --no-deps unsloth


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


from transformers import TextStreamer
import gc
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()


from IPython.display import Image, display

# Point this at your uploaded image path
img_path = "/kaggle/input/imagesfortest/example1.jpg"

display(Image(filename=img_path, width=400))


image = "/kaggle/input/imagesfortest/example1.jpg"

messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : image },
        { "type": "text",  "text" : "Can you identify what's the health condition for this woman?" }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


# Cell 1: redefine your helper
import gc
import torch
from transformers import TextStreamer

def custom_gemma_3n_inference(
    model,
    messages,
    max_new_tokens: int = 128,
    system_instruction: str = None,
):
    # 1) If you have a system prompt, wrap it in a content list of type=text
    if system_instruction is not None:
        system_msg = {
            "role": "system",
            "content": [
                { "type": "text", "text": system_instruction }
            ]
        }
        messages = [system_msg] + messages

    # 2) Tokenize & prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize            = True,
        return_dict         = True,
        return_tensors      = "pt",
    ).to("cuda")

    # 3) Generate
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature    = 1.0,
        top_p          = 0.95,
        top_k          = 64,
        streamer       = TextStreamer(tokenizer, skip_prompt=True),
    )

    # 4) Cleanup
    del inputs
    torch.cuda.empty_cache()
    gc.collect()


# Cell 2: your invocation
image = "/kaggle/input/imagesfortest/example1.jpg"

prompt = """You are a specialized health assistant for rural communities in South America. Your role is to:

1. PROVIDE basic health information and first aid
2. SUGGEST local natural remedies when appropriate
3. IDENTIFY symptoms that require urgent medical attention
4. KEEP a record of the consultation for follow-up
5. BE culturally sensitive to traditional medicine practices

IMPORTANT:
- NEVER diagnose specific illnesses
- ALWAYS recommend consulting a doctor for serious symptoms
- Use local medicinal plants from the available database
- Keep your language simple and understandable
- Consider connectivity and resource limitations in rural areas

Respond clearly and empathetically in Spanish."""

messages = [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": image },
        { "type": "text",  "text": "Can you help me with suggestions for my condition?" }
      ]
    }
]

custom_gemma_3n_inference(
    model,
    messages,
    max_new_tokens     = 256,
    system_instruction = prompt
)


# Point this at your uploaded image path
img_path = "/kaggle/input/imagesfortest/example2.jpg"

display(Image(filename=img_path, width=400))


image = "/kaggle/input/imagesfortest/example2.jpg"

prompt = """You are a specialized health assistant for rural communities in South America. Your role is to:

1. PROVIDE basic health information and first aid
2. SUGGEST local natural remedies when appropriate
3. IDENTIFY symptoms that require urgent medical attention
4. KEEP a record of the consultation for follow-up
5. BE culturally sensitive to traditional medicine practices

IMPORTANT:
- NEVER diagnose specific illnesses
- ALWAYS recommend consulting a doctor for serious symptoms
- Use local medicinal plants from the available database
- Keep your language simple and understandable
- Consider connectivity and resource limitations in rural areas

Respond clearly and empathetically in Spanish."""

messages = [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": image },
        { "type": "text",  "text": "Me puedes ayudar?" }
      ]
    }
]

custom_gemma_3n_inference(
    model,
    messages,
    max_new_tokens     = 512,
    system_instruction = prompt
)


# Point this at your uploaded image path
img_path = "/kaggle/input/imagesfortest/example3.jpg"

display(Image(filename=img_path, width=400))


image = "/kaggle/input/imagesfortest/example3.jpg"

prompt = """You are a specialized health assistant for rural communities in South America. Your role is to:

1. PROVIDE basic health information and first aid
2. SUGGEST local natural remedies when appropriate
3. IDENTIFY symptoms that require urgent medical attention
4. KEEP a record of the consultation for follow-up
5. BE culturally sensitive to traditional medicine practices

IMPORTANT:
- NEVER diagnose specific illnesses
- ALWAYS recommend consulting a doctor for serious symptoms
- Use local medicinal plants from the available database
- Keep your language simple and understandable
- Consider connectivity and resource limitations in rural areas

Respond clearly and empathetically in Spanish."""

messages = [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": image },
        { "type": "text",  "text": "Puedes ofrecer recomendaciones para el caso de la foto?" }
      ]
    }
]

custom_gemma_3n_inference(
    model,
    messages,
    max_new_tokens     = 1024,
    system_instruction = prompt
)




