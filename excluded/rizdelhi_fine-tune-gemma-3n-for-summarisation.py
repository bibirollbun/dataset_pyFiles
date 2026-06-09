import numpy as np 
import pandas as pd
import os


if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo -q
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer -q
    !pip install --no-deps unsloth -q


!pip install --no-deps --upgrade timm -q


!pip install --no-deps git+https://github.com/huggingface/transformers.git  -q # Only for Gemma 3N


###### load the compressed model 
# Model Quantization


from unsloth import FastModel
import torch
max_seq_length = 2048 # Choose any! max - 32K
load_in_4bit = True # Use 4bit quantization to reduce memory usage.
model_checkpoint = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit" # prefer to load from unsloth, as it is optimized for bugs in kaggle

model, tokenizer = FastModel.from_pretrained(
    model_checkpoint,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    full_finetuning=False,
)
model.config.use_cache=False
model.config.pretraining_tp=1


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!
    r = 32,           # Larger = higher accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0,
    use_gradient_checkpointing=True,
    use_rslora=False,
    bias = "none",
    random_state = 3468
)


from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from datasets import load_dataset

dataset = load_dataset("abisee/cnn_dailymail", '3.0.0', split = "train[:120000]")
#print(dataset[11])


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


def format_summarization_for_gemma(batch):
    return {
        "text": [
            "<start_of_turn>user\n"
            f"Summarize the following text:\n{article}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{summary}\n"
            "<end_of_turn>"
            for article, summary in zip(batch["article"], batch["highlights"])
        ]
    }


dataset = dataset.map(format_summarization_for_gemma, batched = True)
#dataset[100]["text"]


eval_dataset = load_dataset("abisee/cnn_dailymail", '3.0.0', split="train[120000:150000]")
eval_dataset = eval_dataset.map(format_summarization_for_gemma, batched = True)
#eval_dataset[100]["text"]


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
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


#tokenizer.decode(trainer.train_dataset[100]["input_ids"])


#tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# @title Show current memory stats
import gc
gc.collect()
torch.cuda.empty_cache()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model
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


#trainer.model.save_pretrained("gemma-3n-E4B-it-unsloth-bnb-4bit-finetune")


#tokenizer.save_pretrained("gemma-3n-E4B-it-unsloth-bnb-4bit-finetune")


gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


!rm -rf /kaggle/working/temp


#!mkdir -p ~/.huggingface
#!echo -n $huggingface_token > ~/.huggingface/token


# Save to q4_k_m 
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
huggingface_token = user_secrets.get_secret("hugging face token")
login(huggingface_token)


# Save merged model (after LoRA/adapters applied)
model.save_pretrained_merged("gemma-3n-E4B-it-unsloth-bnb-4bit-finetune", tokenizer=tokenizer, save_method="merged_16bit")


for f in os.listdir('/kaggle/working'):
    path = '/kaggle/working/' + f
    if os.path.isfile(path):
        print(f"{f}: {os.path.getsize(path)/1e6:.2f} MB")


gc.collect()


if False:
    model.save_pretrained_gguf(
        "gemma-3n-E4B-it-unsloth-bnb-4bit-finetune",
        quantization_type="Q8_0",
    )


if True:
    model.push_to_hub_gguf(
    "gemma-3n-E4B-it-unsloth-bnb-4bit-finetune",
    quantization_type="Q8_0",
    repo_id="rizwan/gemma-3n-summarization-qlora-gguf",
    token=huggingface_token
)

