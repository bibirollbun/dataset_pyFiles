!pip uninstall transformers --yes
!pip install unsloth
!pip install bitsandbytes


import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


from datasets import load_dataset
from pathlib import Path

files = list(map(str, Path("/kaggle/input/wiki-20220301-en-sci").glob("*.parquet")))
dataset = load_dataset("parquet", data_files=files, split="train")


dataset[:3]


from unsloth import FastModel


FastModel.from_pretrained?


model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-1B",
        max_seq_length = 512, 
        dtype = torch.bfloat16,
        full_finetuning = True,
        load_in_4bit = False,
        load_in_8bit = False,
        device_map = 'auto'
)


from trl import SFTTrainer,SFTConfig

training_args = SFTConfig(
    warmup_ratio=0.1, 
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    # per_device_eval_batch_size=2,
    num_train_epochs=1,
    report_to='none',
    output_dir = f'./checkpoints',
    # fp16=True,
    bf16=True,
    optim='paged_adamw_32bit', #'paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch'
    torch_compile=True, #model = torch.compile(model)
    gradient_accumulation_steps=1,
    logging_steps=100,
    save_strategy="steps",
    save_only_model=True,
    save_total_limit=1,
    save_steps=100,
    lr_scheduler_type='cosine',
    weight_decay=0.01,
)


torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset_text_field='text',
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
trainer.save_model(f'./Llama3.2-1b-wiki')







