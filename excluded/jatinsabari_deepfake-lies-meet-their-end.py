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
!pip install "transformers>=0.34.0"
 # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


!pip install "transformers>=0.34.0"



import unsloth
from unsloth import FastModel
import torch

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



audio_file = "/kaggle/input/deep-voice-deepfake-voice-recognition/KAGGLE/AUDIO/REAL/biden-original.wav" #real audio

messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "text",  "text" : "check whether the audio file is real or not? use reality check, artifacts, GUN's" }
    ]
}]
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


image_fake = "/kaggle/input/deepfake-and-real-images/Dataset/Train/Fake/fake_0.jpg" #fake image

messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : image_fake },
        { "type": "text",  "text" : "use lighting physics, artifacts to find the given image is manipulated: give reasons?" }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_3n_inference(model, messages, max_new_tokens = 256)


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


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from datasets import load_dataset
dataset = load_dataset("/kaggle/input/full-finetune-gemma-deepfake", split = "train[:3000]")


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


dataset[10]


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


dataset[12]["text"]


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
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


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[19]["labels"]]).replace(tokenizer.pad_token, " ")


# @title Show current memory stats
import gc
gc.collect()
torch.cuda.empty_cache()
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
gc.collect()
torch.cuda.empty_cache()
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


import gc
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)
image = "/kaggle/input/deepfake-and-real-images/Dataset/Train/Real/real_0.jpg"
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},  # Image part
        {"type": "text", "text": "<image_soft_token> whether the image is manipulated or not? give reasons."}  # Text part
    ]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
model.generate(
    **inputs,
    max_new_tokens = 200, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
gc.collect()



import gc
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)
audio = "/kaggle/input/deep-voice-deepfake-voice-recognition/KAGGLE/AUDIO/FAKE/trump-to-taylor.wav"
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},  # Image part
        {"type": "text", "text": "<audio_soft_token> why the audio is manipulated? give reasons based on reality check"}  # Text part
    ]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
model.generate(
    **inputs,
    max_new_tokens = 200, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
gc.collect()



# Save the final model
trainer.save_model("gemma-deepfake-detector")

# Save tokenizer
tokenizer.save_pretrained("gemma-deepfake-detector")

import os

save_path = "gemma-deepfake-detector"
print("Model saved to:", os.path.abspath(save_path))
print("Saved files:")
for f in os.listdir(save_path):
    print("-", f)



from transformers import AutoConfig

config = AutoConfig.from_pretrained("gemma-deepfake-detector")
config.save_pretrained("gemma-deepfake-detector")  # optional, it may already exist



from huggingface_hub import HfApi, create_repo
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your details
HF_TOKEN = "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"  # Get from https://huggingface.co/settings/tokens
MODEL_PATH = "/kaggle/working/gemma-deepfake-detector"  # Unzipped folder
REPO_NAME = "jatinsabari/Gemma-3n-deepfake_detector"



# Upload files
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path=MODEL_PATH,
    repo_id=REPO_NAME,
    repo_type="model"
)

print(f"Model uploaded to https://huggingface.co/{REPO_NAME}")

