import pkg_resources

# List of package names
package_list = ["accelerate", "datasets", "huggingface_hub", "transformers",
                "torch", "wandb", "triton", "shtab", "hf_transfer",
                "xformers", "tyro", "cut_cross_entropy", "bitsandbytes",
                "peft", "trl", "unsloth_zoo", "unsloth"]

# Print the versions of the listed packages
# pkg_resources is deprecated as an API
# but we can still use it while we can
# for a quick preview.
for package in package_list:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package} version: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package} is not installed")
    except:
        print(f"checking {package} was unsuccessful")


# Unsloth is quick and it attempts to patch the system including transformers.
# It is essential we use the version that is working without whilesale
# replacement (upgrade) of the Kaggle's environment includig torch
# and the cuda support. Also uses T-4 (CUDA: 7.5) instead of the older P100,
# which is not well supported by recent unsloth.
# 
# Specify the version of torch to keep to prevent upgrading into a package
# combination that's not well tested or broken. If need we can specify 
# all the dependency: they are "bitsandbytes-0.45.0 cut_cross_entropy-25.1.1 
# hf_transfer-0.1.9 huggingface_hub-0.27.1 peft-0.14.0 shtab-1.7.1 
# tokenizers-0.20.3 transformers-4.46.3 triton-3.1.0 trl-0.13.0 
# tyro-0.9.8 unsloth-2024.12.8 unsloth_zoo-2024.12.3 xformers-0.0.28.post1"
# Among them, these were upgraded: 
# huggingface-hub-0.24.7 tokenizers-0.19.1 transformers 4.44.2
# 
!pip install unsloth==2024.12.8 unsloth_zoo==2024.12.3 torch==2.4.1+cu121


import requests
import os

def download_google_drive_file(file_id, folder_path, file_name=None):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get the file name from the URL
    if not file_name:
        file_name = url.split('/')[-1]
    file_path = os.path.join(folder_path, file_name)

    # Download the file
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                file.write(chunk)
    print(f"File downloaded and saved to {file_path}")


# We can download from Google Drive before the dataset is visible in Kaggle.
# https://drive.google.com/file/d/1xLq1AQcYwixGKpo3FB7Uhdj18qAPUtY-/view?usp=sharing

download_google_drive_file('1xLq1AQcYwixGKpo3FB7Uhdj18qAPUtY-', 'poems', file_name='LvShi_7.csv')


# %%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# !pip install torch==2.5.1+cu121 "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install torch==2.5.1+cu121 "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/gemma-2-2b-bnb-4bit",
    "unsloth/gemma-2-2b-it-bnb-4bit",
] # More models at https://huggingface.co/unsloth

# We use the base model instead of instruct model (chat) for better text completion
# The smallest variant works since we don't have a lot of data anyway.
# It is also greener and better for the environment.
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "google/gemma-2-2b",
    model_name = "unsloth/gemma-2-2b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# I uploaded the dataset to Kaggle but for unknown reasons it is not available.
# So I'm downloading from my shared Google Drive and load it manually.
# This example code is an attempt to demystify the process of creating
# and using dataset with the transformers+unsloth framework.

# The CSV format allows easy quality control and simple statistics gathering.

import csv, os, gc

# Function to read CSV and create dictionaries
def read_csv_as_dicts(filename):
    rows = []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            rows.append(row)
    return rows

# Example usage
filename = os.path.join('poems', 'LvShi_7.csv')
poems = read_csv_as_dicts(filename)

# Print dictionaries
for poem in poems:
    print(poem)
    break


# Instead of just feeding the language model pure Chinese glyphs. We hypothesized
# that the phonetics (pin yin with tone in the end and a symbol for ping ze)
# can help the language model to connect the sound, which is extremely important
# for Chinese poems which were originally written for singing with predefined
# accompanion music. Our dataset allows us to experiment with different approaches.
# This is the idea of hybrid tokenization for multiple stream of data, 
# without changing tokenizer or deep into the model.

def make_hybrid_training_text(poem):
    paragraphs=poem['paragraphs']
    pinyin_pick=poem['pinyin_pick']
    style=poem['style']

    # The dataset is created and checked with exactly fixed lengths, -- reason
    # for a csv file in final form.
    # we still use some length checks though to guard just in case of file corruption.

    # paragraphs looks like a list but in reality it is a string because of the serialization
    # "['é�”ä¾†ä½•è™•æ›´è¿½å°‹ï¼Œæ”¾æ› èª°è«–å�¤èˆ‡ä»Šã€‚', 'é¢¨å¸¦æ³‰è�²æµ�è°·å�£ï¼Œé›²å’Œå±±å½±è�½æ½­å¿ƒã€‚', 'è³‡èº«è‡ªæœ‰è¡£ä¸­å¯¶ï¼Œæ¿Ÿä¸–èª°è—�å®¤å…§é‡‘ã€‚', 'ç­–æ�–å�¶ä¾†æ�—ä¸‹å��ï¼Œé³¥è�²ç›¸å’Œå”±åœ“éŸ³ã€‚']"
    training_text = []
    verse_positions=(2, 10, 22, 30, 42, 50, 62, 70)
    verses=[]
    if len(paragraphs)==80:
        for i in verse_positions:
            verses.append(paragraphs[i:i+7])
    verses = "".join(''.join(verses))
    pinyin = [p for p in pinyin_pick.split(" ") if p]
    style = style.replace(" ", "")

    # print(f'len(verses)={len(verses)}, len(pinyin)={len(pinyin)}, len(style)={len(style)}')
    if len(verses)==56 and len(pinyin)==56 and len(style)==56:
        for i in range(0, 8):
            for j in range(0, 7):
                x = i * 7 + j
                training_text.append(verses[x])
                training_text.append(pinyin[x])
                training_text.append(style[x])
                training_text.append(' ')
            training_text.append(paragraphs[verse_positions[i]+7])
            training_text.append(' ')
        training_text=training_text[:-1]
    training_text=''.join(training_text)
    return(training_text)

# remove the very few exact duplicates
text_list = list(set([make_hybrid_training_text(poem) for poem in poems]))
del poems
text_list[-1]


from datasets import Dataset, load_dataset
# dataset = load_dataset("roneneldan/TinyStories", split = "train[:2500]")

dataset = Dataset.from_dict({"text": text_list})
# dataset = dataset.train_test_split(test_size=0.2)
# train_dataset = dataset['train']
# val_dataset = dataset['test']

del text_list
gc.collect()

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
dataset = dataset.map(formatting_prompts_func, batched = True,)


for row in dataset[:5]["text"]:
    print("=========================")
    print(row)


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 10,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 500,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs-64",
        report_to = "none", # Use this for WandB etc
    ),
)


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


FastLanguageModel.for_inference(model)


# generate using the rarely seen poem start by æ��ç™½.
# max_new_tokens is set to 256 since we have th complex glyph, pin yin, and flat-oblique assignment based on style rules
#
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    "å�¾wu2- å…„xiong1- è©©shi1^ é…’jiu3^ ç¹¼ji4^ é™¶tao2- å�›jun1-ï¼Œ"
]*1, return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
tokenizer.batch_decode(outputs)


# TextStreamer using the rarely seen poem start by æ��ç™½
# max_new_tokens is set to 256 since we have th complex glyph, pin yin, and flat-oblique assignment based on style rules

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    "å�¾wu2- å…„xiong1- è©©shi1^ é…’jiu3^ ç¹¼ji4^ é™¶tao2- å�›jun1-ï¼Œ"
]*1, return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 256)


model.save_pretrained("lora_model") # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference


inputs = tokenizer(
[
    "å�¾wu2- å…„xiong1- è©©shi1^ é…’jiu3^ ç¹¼ji4^ é™¶tao2- å�›jun1-ï¼Œ"
]*1, return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


if False:
    # I highly do NOT suggest - use Unsloth if possible
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")


# Merge to 16bit
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit_forced",)

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")


import os
import zipfile

def zip_directory(folder_path, dest_file_path):
    with zipfile.ZipFile(dest_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                print(f'storing {file_path} as {arcname}')
                zip_file.write(file_path, arcname)
    print(f'finished creating {dest_file_path}')


# zip up for download just in case
#
# zip_directory('/kaggle/working/lora_model', 'VersedGemma-0.3-lvshi-unsloth-10epochs-lora_model.zip')
# zip_directory('/kaggle/working/model', 'VersedGemma-0.3-lvshi-unsloth-10epochs-model.zip')
zip_directory('/kaggle/working/outputs-64', 'VersedGemma-0.3-lvshi-unsloth-10epochs-ckpts.zip')


# create a link in the netbook for downloading if the 'MORE OPTIONS' doesn't work
# from IPython.display import FileLink
# FileLink(r'VersedGemma-0.3-lvshi-unsloth-10epochs-ckpts.zip')


# clean up (room)

# os.remove('VersedGemma-0.2-lvshi-unsloth-10epochs-lora_model.zip')
# os.remove('VersedGemma-0.2-lvshi-unsloth-10epochs-model.zip')
# os.remove('VersedGemma-0.3-lvshi-unsloth-10epochs-ckpts.zip')


import kagglehub

# Option 1: prompt for credentials if not existing  ~/.kaggle/kaggle.json
# kagglehub.login()

# Option 2: Environment variables (export your Kaggle username and token)
# export KAGGLE_USERNAME=your_username
# export KAGGLE_KEY=your_key

kh_owner = 'll2000'
kh_model = 'VersedGemma-v0.3-Unsloth-10ep'
kh_framework='transformers'
kh_model_version='v1'

handle = f'{kh_owner}/{kh_model}/{kh_framework}/{kh_model_version}'

# Uploading model weight and tokenizer with their configurations. kagglehub.login() will be called automatically if need
# kagglehub.model_upload(handle, model_save_name_or_path)


from kagglehub.handle import parse_model_handle
from kagglehub.clients import KaggleApiV1Client

def model_exists(owner, model):
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{owner}/{model}/get")
        return True
    except Exception as e:
        # "403 Client Error" doesn't prevent upload - they should use 404 Not Found instead.
        print(f"Error checking model existence: {e}")
        return False

# check and upload following example of https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/models_helpers.py
def upload_model(handle, local_model_dir, overwrite=False):
    try:
        if not overwrite:
            h = parse_model_handle(handle)
            if model_exists(h.owner, h.model):
                print(f"Model with handle {h.owner}/{h.model} already exists. Skipping upload.")
                return
        kagglehub.model_upload(handle, local_model_dir)
        print(f'Model uploaded to Kaggle with handle: {handle}')
    except Exception as e:
        print(f"Error uploading model: {e}")


if False:
    
    kagglehub.login()

    upload_model(handle, '/kaggle/working/model')


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

