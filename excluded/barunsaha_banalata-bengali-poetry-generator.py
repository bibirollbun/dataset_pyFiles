import os

from kaggle_secrets import UserSecretsClient


# Get/create access token with write permissions on Hugging Face
HF_ACCESS_TOKEN = UserSecretsClient().get_secret('HF_ACCESS_TOKEN')

os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')
os.environ['WANDB_PROJECT'] = 'banalata'


# !pip install -q pip3-autoremove
# !pip-autoremove torch torchvision torchaudio -y
# !pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.47.1  # https://github.com/unslothai/unsloth/issues/1527
# !pip install -q -U --no-cache-dir unsloth

# January 18, 2025
!pip install unsloth
!pip install -q kagglehub
!pip install -q keras-hub
!pip install wandb


from datetime import datetime


BASE_MODEL_NAME = 'unsloth/gemma-2-9b'
TARGET_MODEL_NAME = 'banalata'

# Most of the poems in our datasets have up to 2048 tokens
# We will visualize the tokens distribution later
MAX_SEQ_LENGTH = 2048  # Choose any! Unsloth supports RoPE Scaling internally!
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.


def get_today() -> str:
    return datetime.today().strftime('%Y-%m-%d')


def get_run_name() -> str:
    return f'{TARGET_MODEL_NAME}-{get_today()}-kaggle'


import torch

from huggingface_hub import login as hf_login
from kaggle_secrets import UserSecretsClient
from unsloth import FastLanguageModel


# Get/create access token with write permissions on Hugging Face
# Add it to the Kaggle notebook via the Add-ons > Secrets menu in the toolbar at the top
HF_ACCESS_TOKEN = UserSecretsClient().get_secret('HF_ACCESS_TOKEN')

# # For Colab notebooks
# from google.colab import userdata
# HF_ACCESS_TOKEN = userdata.get('HF_ACCESS_TOKEN')
# #

hf_login(token=HF_ACCESS_TOKEN)  # Comment out this step if you do not wish to login

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)


model = FastLanguageModel.get_peft_model(
    model,
    r=4,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        "embed_tokens", "lm_head",
    ],
    lora_alpha=8,  # 16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=12345,
    use_rslora=True,   # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


LLM_PROMPT_TRAINING = '''নীচে একটি নির্দেশ দেওয়া হয়েছে যা একটি কাজের বর্ণনা করে। একটি প্রতিক্রিয়া লিখুন যা নির্দেশ অনুসরণ করে যথাযথভাবে কাজটি সম্পন্ন করে।

### নির্দেশ:
{}

### প্রতিক্রিয়া:
{}'''

LLM_PROMPT_INFERENCE = '''নীচে একটি নির্দেশ দেওয়া হয়েছে যা একটি কাজের বর্ণনা করে। একটি প্রতিক্রিয়া লিখুন যা নির্দেশ অনুসরণ করে যথাযথভাবে কাজটি সম্পন্ন করে।

### নির্দেশ:
{}

### প্রতিক্রিয়া:
'''

# Prompts to test text generation
PT_TEST_PROMPTS = [
    'রাশিরাশি মণি-মুক্ত-মাণিক্য ভেসে আসে',
    'সবুজঘেরা গ্রামের বুকে',
    'মহাভারতের গল্পে প্রাচীন',
    'আজ পৃথিবী নীল ও আকাশ সবুজ',
    'সূর্যোদয় থেকে সূর্যাস্ত পর্যন্ত পিঠভাঙ্গা খাটুনি',
    'স্বপ্নের ধূসর নদীর',

    # The following prompts contain a phrase and a poet's name
    '''সবুজঘেরা গ্রামের বুকে

    রবীন্দ্রনাথ ঠাকুর''',
    '''স্বপ্নের ধূসর নদীর

    জীবনানন্দ দাস''',
    '''তন্ত্র, মন্ত্র, সাধক ও সাধনা

    রামপ্রসাদ সেন''',
    '''মনের মানুষ

    লালন ফকির''',
]

# Prompts to test text generation based on instructions
FT_TEST_PROMPTS = [
    'শূন্যস্থান পূরণ করুন: পশ্চিমবঙ্গের রাজধানী হল _____।',
    'সৌরজগত সম্পর্কে সংক্ষেপে বলুন।',
    'জীবনানন্দ দাশের স্টাইল-এ একটি কবিতা লিখুন।',
    'একটা সুকান্ত ভট্টাচার্যের স্টাইলে একটা কবিতা লিখুন।',
    'একটা কৌতুক কবিতা লেখ।',
    'বনলতা সেন কবিতার মূলভাব উল্লেখ কর।',
]


import os

import kagglehub

# Download latest version
# kagglehub.login()
bengali_poems_path = kagglehub.dataset_download('barunsaha/bengali-poems')

print('Path to Bengali poems dataset files:', bengali_poems_path)
os.listdir(bengali_poems_path)


import pathlib

import datasets
import matplotlib.pyplot as plt
import pandas as pd

from datasets import concatenate_datasets, load_dataset


# ROOT_DIR = '/kaggle/input/bengali-poems/bengali_poems/'
# ROOT_DIR = '/root/.cache/kagglehub/datasets/barunsaha/bengali-poems/versions/4/'
EOS_TOKEN = tokenizer.eos_token


def clean_text(text: str) -> str:
    text = text.strip()
    # Replace two hyphens with an em dash
    text = text.replace('--', '—')
    # Replace two end of sentence markers with an end of stanza (or poem) marker
    text = text.replace('।।', '॥')
    # En dash with em dash
    text = text.replace('–', '—')
    # NBSP with space
    text = text.replace(' ', ' ')

    text = text.replace(' ;', ';')
    text = text.replace(' ।', '।')
    text = text.replace(' ॥', '॥')

    return text


def formatting_pre_training(examples) -> dict:
    texts  = examples['text']
    outputs = []

    for text in texts:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = clean_text(text) + EOS_TOKEN
        outputs.append(text)

        outputs.extend(chunks)

    return {'text' : outputs}


def formatting_instructions_tuning(examples) -> dict:
    question = examples['question']
    answer = examples['answer']
    outputs = []

    for q, a in zip(question, answer):
        text = clean_text(LLM_PROMPT_TRAINING.format(q, a)) + EOS_TOKEN
        outputs.append(text)

    return {'text' : outputs}


data = [
    {'text': file.read_text(encoding='utf-8').strip()}
    for file in pathlib.Path(f'{bengali_poems_path}').rglob('*.txt')
]
print(f'#files: {len(data)}')

df = pd.DataFrame(data)
print(df)

ds_text = datasets.Dataset.from_pandas(df)
print(ds_text)

ds_text = ds_text.map(
    formatting_pre_training,
    batched=True,
)

# A tiny instruction-following dataset
ds_instructions = load_dataset('barunsaha/bangla_nirdeshabali', split='train')
ds_instructions = ds_instructions.map(
    formatting_instructions_tuning,
    batched=True,
    remove_columns=['id', 'question', 'answer']
)
print(ds_instructions)

ds_text = concatenate_datasets([ds_text, ds_instructions]).shuffle()


total_tokens = 0
tokens_count = []

for row in ds_text:
    n_tokens = len(tokenizer.encode(row['text']))
    total_tokens += n_tokens
    tokens_count.append(n_tokens)

print(f'Total tokens available: {total_tokens} (or {total_tokens / 1e6} million)')

# Plot the cumulative distribution
plt.hist(tokens_count, bins=MAX_SEQ_LENGTH, cumulative=True, density=True)
plt.xlabel(f'#tokens ({MAX_SEQ_LENGTH=})')
plt.ylabel('Cumulative density')
plt.axvline(x=MAX_SEQ_LENGTH, color='r')
plt.show()


new_dataset = ds_text.train_test_split(test_size=0.1)
print(new_dataset)


from transformers import EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported


training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    warmup_ratio=0.1,
    num_train_epochs=5,

    learning_rate=7e-4,

    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),

    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=12345,
    output_dir="outputs",
    report_to="wandb", # Use this for WandB etc
    run_name=get_run_name(),

    save_strategy="epoch",
    save_total_limit=2,

    # Eval
    fp16_full_eval=True,
    per_device_eval_batch_size=4,
    eval_accumulation_steps=2,
    eval_strategy="epoch",
    # eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=8,
    train_dataset=new_dataset["train"],
    eval_dataset=new_dataset["test"],
    # callbacks=[
    #     EarlyStoppingCallback(early_stopping_patience=4),
    # ],
    args=training_args,
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


# from unsloth import unsloth_train


# # https://unsloth.ai/blog/gradient
# # unsloth_train fixes gradient_accumulation_steps
# # trainer_stats = trainer.train()
# trainer_stats = unsloth_train(trainer)
# print(trainer_stats)


import pandas as pd

df_train_history = pd.DataFrame(trainer.state.log_history)
print(df_train_history.tail())
df_train_history.plot(x='epoch', y=['loss', 'eval_loss'])


from transformers import TextStreamer


MAX_OUTPUT_TOKENS = 256


text_streamer = TextStreamer(tokenizer)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

for idx, query in enumerate(PT_TEST_PROMPTS, start=1):
    print(f'#{idx} {query}\nResponse:')
    prompt = query
    inputs = tokenizer([prompt], return_tensors='pt').to('cuda')
    # print(inputs)

    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MAX_OUTPUT_TOKENS)
    print('=' * 60, end='\n\n')

for idx, query in enumerate(FT_TEST_PROMPTS, start=1):
    print(f'#{idx} {query}\nResponse:')
    prompt = LLM_PROMPT_INFERENCE.format(query)
    inputs = tokenizer([prompt], return_tensors='pt').to('cuda')
    # print(inputs)

    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MAX_OUTPUT_TOKENS)
    print('=' * 60, end='\n\n')


#
# UNCOMMENT and run this block if you are running the code outside Kaggle
#

# import kagglehub

# # The following will show a UI -- read the instructions on how to get an access token
# kagglehub.login()


# Commented out on January 18, 2025, in order not to update the model on Kaggle Hub

# TARGET_MODEL_NAME = 'banalata'
# # For Hugging Face upload
# HF_USER_NAME = 'barunsaha'  # REPLACE this with your HF user name

# model.save_pretrained(TARGET_MODEL_NAME)  # Local saving
# tokenizer.save_pretrained(TARGET_MODEL_NAME)
# # trainer.save_model(TARGET_MODEL_NAME)

# model.push_to_hub(f'{HF_USER_NAME}/{TARGET_MODEL_NAME}')  # Online saving
# tokenizer.push_to_hub(f'{HF_USER_NAME}/{TARGET_MODEL_NAME}')  # Online saving


# import kagglehub
# import keras_hub


# kaggle_username = kagglehub.whoami()['username']

# kagglehub.model_upload(
#   handle=f'{kaggle_username}/{TARGET_MODEL_NAME}/transformers/9b-lora',
#   local_model_dir=TARGET_MODEL_NAME,
#   version_notes=f'Update {get_today()}',
# )


# # Save to q4_k_m GGUF
# quantization = 'q4_k_m'
# gguf_dir = f'{TARGET_MODEL_NAME}_{quantization}'
# model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method='q4_k_m')
# # model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m")

# kagglehub.model_upload(
#   handle=f'{kaggle_username}/{TARGET_MODEL_NAME}/gguf/{quantization}',
#   local_model_dir=gguf_dir,
#   version_notes=f'Update {get_today()}',
# )


# import os

# import kagglehub
# from transformers import TextStreamer
# from unsloth import FastLanguageModel


# # kagglehub.login()

# # Download latest version
# lora_path = kagglehub.model_download('barunsaha/banalata/transformers/9b-lora')

# print('Path to model files:', lora_path)
# print('\n'.join(os.listdir(lora_path)))


# MAX_OUTPUT_TOKENS = 256

# # Load the LoRA adapter and prepare for inference.
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=lora_path,  # The LoRA adapter that we have downloaded
#     max_seq_length=MAX_SEQ_LENGTH,
#     dtype=DTYPE,
#     load_in_4bit=LOAD_IN_4BIT,
#     device_map={'': 1},  # 'auto',  # https://stackoverflow.com/a/76469875/147021
# )
# FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# text_streamer = TextStreamer(tokenizer)

# for idx, query in enumerate(PT_TEST_PROMPTS, start=1):
#     print(f'#{idx} {query}')
#     prompt = query
#     inputs = tokenizer([prompt], return_tensors='pt').to('cuda')
#     # print(inputs)

#     _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=MAX_OUTPUT_TOKENS)
#     print('=' * 60, end='\n\n')


# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit")

# Just LoRA adapters -- we have already done this earlier, in a different way
if False: model.save_pretrained_merged("model", tokenizer, save_method="lora")
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method="lora")

