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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("WANDB")
os.environ["WANDB_API_KEY"] = secret_value_0


!pip install -qqqU datasets huggingface_hub peft accelerate bitsandbytes peft markdown2 evaluate
!pip install trl transformers
!pip install -qqqU trl==0.12.0
!pip install -qqqU transformers==4.46.0


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig,AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import markdown2
from IPython.display import display, HTML, clear_output
import time
from huggingface_hub import login
import os
from kaggle_secrets import UserSecretsClient
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import Trainer
from datasets import load_dataset,concatenate_datasets,DatasetDict
from peft import AutoPeftModelForCausalLM
###################################################


def format_dataset(sample):
    messages_formatted = [
        {"content":"Translate following information to Hindi: "+sample["translation"]["en"], "role":"user"},
        {"content":sample["translation"]["hi"], "role":"assistant"}
    ]
    sample["messages"] = messages_formatted

    return sample
dataset = load_dataset(path="cfilt/iitb-english-hindi")
dataset = dataset.map(format_dataset)


dataset['train'][0]['messages']


dataset_ins = load_dataset(path="smangrul/hindi_instruct_v1")


dataset_ins['train'][0]['messages']


def format_instruction_dataset(sample):
    msg=[]
    for i in sample['messages']:
        if i['role']=='system':
            continue
        else:
            msg.append(i)
    sample["messages"] = msg
    return sample
dataset_ins = dataset_ins.map(format_instruction_dataset)


dataset_ins['train'][0]['messages']


def normalize_dataset(dataset, text_column):
    return dataset.map(lambda x: {"messages": x[text_column]}, remove_columns=dataset['train'].column_names)

dataset = normalize_dataset(dataset, "messages")
dataset_ins = normalize_dataset(dataset_ins, "messages")


# Combine 'train' splits
train_combined = concatenate_datasets([dataset["train"], dataset_ins["train"]])

# Combine 'test' splits
test_combined = concatenate_datasets([dataset["test"], dataset_ins["test"]])


# Create a new combined DatasetDict
dataset_mod = DatasetDict({
    "train": train_combined,
    "test": test_combined
})


model_name = r"/kaggle/input/gemma/transformers/2b-it/2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")


################################################################
param_rank=8
param_lora_alpha=32
# hyperparameter tuning required
lora_config = LoraConfig(
    r=param_rank,
    lora_alpha=param_lora_alpha,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],## which modules in the transformer layers are modified using LoRA 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)



tokenizer.padding_side = "right"
model.config.eos_token_id = tokenizer.eos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id


finetune_name = "Gemma2-2B-HindiTranslation"
output_dir_name = finetune_name + "_" + str(param_rank) + "_" + str(param_lora_alpha)
output_dir_path = '/kaggle/working/'+output_dir_name
####################################################


max_seq_length = 1024 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_mod["train"],
    eval_dataset=dataset_mod["test"],
    dataset_text_field="messages",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        gradient_checkpointing=True,
        max_steps=1000,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),  # Usar BF16 si está disponible
        logging_steps=10,
        output_dir=output_dir_name,
        optim="adafactor",  # Optimizador compatible con QLoRA
        learning_rate=5e-4,  # Tasa de aprendizaje típica para LoRA
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb",
        eval_strategy="no",
        eval_steps=0.1,
    ),
)



trainer.train()


trainer.save_model(output_dir_name)


# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=output_dir_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(
    output_dir_name, safe_serialization=True, max_shard_size="2GB"
)


prompt = f"""Translate following information to Hindi: you may be having ."""

chat = [{"role": "user", "content": f"{prompt}"}]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat, return_tensors="pt").to('cuda')

_ = merged_model.to('cuda').generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.3, top_p=0.9)


print('\n\n', tokenizer.decode(_[0], skip_special_tokens=True))




