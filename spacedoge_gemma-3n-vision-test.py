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

!pip install --no-deps git+https://github.com/huggingface/transformers.git@6b09c8eab05820d480f4da97d23456428d410082
!pip install --no-deps --upgrade timm


!pip show transformers


%%capture

from unsloth import FastModel
import torch


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    dtype = None, # fp16
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)


from transformers import TextStreamer
# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )


from PIL import Image
import numpy as np

image_test1 = Image.open("/kaggle/input/gemma3n-vision-tests/ocr_test1.jpg")
image_test1.resize((384,384))


messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : "/kaggle/input/gemma3n-vision-tests/ocr_test1.jpg" },
        { "type": "text",  "text" : "Read text from this image." }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(messages, max_new_tokens = 256)


image_test2 = Image.open("/kaggle/input/gemma3n-vision-tests/ocr_test2.jpg")
image_test2.resize((384,384))


messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : "/kaggle/input/gemma3n-vision-tests/ocr_test2.jpg" },
        { "type": "text",  "text" : "Read text from this image." }
    ]
}]
do_gemma_3n_inference(messages, max_new_tokens = 256)


import os
import psutil
import gc
import ctypes


def force_memory_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)


del model
del tokenizer
force_memory_cleanup()


from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer
import torch


GEMMA_PATH = '/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1'

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH, torch_dtype=torch.float32, device_map="cpu"
)


def do_gemma_3n_transformers_inference(
        messages, transpose_image=True, max_new_tokens=128
    ):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)
    
    if transpose_image:
        inputs["pixel_values"] = inputs["pixel_values"].permute(0,1,3,2)
    input_len = inputs["input_ids"].shape[-1]

    _ = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(processor.tokenizer, skip_prompt = True)
    )
    


messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : "/kaggle/input/gemma3n-vision-tests/ocr_test1.jpg"},
        { "type": "text",  "text" : "Read text from this image."}
    ]
}]
do_gemma_3n_transformers_inference(messages, max_new_tokens = 64)


messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : "/kaggle/input/gemma3n-vision-tests/ocr_test1.jpg"},
        { "type": "text",  "text" : "Read text from this image."}
    ]
}]
do_gemma_3n_transformers_inference(messages, transpose_image=False, max_new_tokens = 64)

