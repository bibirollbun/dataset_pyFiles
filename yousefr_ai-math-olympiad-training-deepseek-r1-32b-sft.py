#!pip install accelerate bitsandbytes trl -q
from IPython.display import clear_output

!python -m pip install --no-index -v --find-links=/kaggle/input/aimo-packages/offline_packages trl --pre
!python -m pip install --no-index -v --find-links=/kaggle/input/aimo-packages/offline_packages bitsandbytes --pre 
#!python -m pip install --no-index -v --find-links=/kaggle/input/aimo-packages/offline_packages vllm --pre
#!python -m pip install --no-index -v --find-links=/kaggle/input/aimo-packages/offline_packages timm --pre
#!python -m pip install --no-index -v --find-links=/kaggle/input/aimo-packages/offline_packages triton --pre

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

clear_output()


# Restart the notebook so autoawq gets recognized
#!pip install autoawq -q


from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch

token = ""
quantized = True
dora = True

# using dora will increase the model reasoning and accuracy
peft_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM, inference_mode = False, r = 128, lora_alpha = 128, lora_dropout=0.1, use_dora = dora
)

if not quantized:

    bits_config = BitsAndBytesConfig(
        load_in_4bit = True, load_4bit_use_double_quant = True, bnb_4bit_quant_type = 'nf4', bnb_4bit_compute_dtype = torch.float16
    )

max_memory = {
    0: "22.5GB",
    1: "22.5GB",
    2: "22.5GB",
    3: "22.5GB"
}
# unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit
tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
                                             device_map="auto",
                                             max_memory = max_memory,
                                             torch_dtype = torch.float16)   

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# we can modify this to make the model more performant
# we could experiment with decreasing it to 256
context_length = model.config.max_position_embeddings = 512

torch.cuda.empty_cache()

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = True)
    
model = get_peft_model(model, peft_config)

for name, param in model.named_parameters():
    if "lora" in name:  # Ensuring LoRA layers are trainable
        param.requires_grad = True
    
tokenizer.pad_token = tokenizer.eos_token


import pandas as pd
from datasets import Dataset

# Only 532 problems exceed the 1024 length
dataset_dir = "/kaggle/input/math-problems-imo/math_problems.parquet"

dataset = pd.read_parquet(dataset_dir)

def filter_long_strings(row):
    return all(len(str(value)) <= context_length for value in row)

# Apply the filter to remove rows with long strings
dataset = dataset[dataset.apply(filter_long_strings, axis=1)]

# normalize the dataset
dataset['problem'] = dataset['problem'].str.lower()
dataset['solution'] = dataset['solution'].str.lower()

# Dataset sample
dataset = dataset[:100]

dataset = Dataset.from_pandas(dataset)

def tokenize_text(example):
    # Tokenize the problem (the input)
    input_encodings = tokenizer(
        example['problem'],
        padding='max_length',
        truncation=True,
        max_length=context_length,
        padding_side='right'
    )
    
    # Tokenize the solution (the target)
    target_encodings = tokenizer(
        example['solution'],
        padding='max_length',
        truncation=True,
        max_length=context_length,
        padding_side='right'
    )
    
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

dataset = dataset.map(tokenize_text)

dataset = dataset.remove_columns(["problem", "solution"])

dataset


from transformers import DataCollatorForLanguageModeling
import gc

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)

bs = 8

dataset = dataset.shuffle()

dataset = dataset.train_test_split(test_size = 0.1)
train_dataset, eval_dataset = dataset["train"], dataset["test"]

gc.collect()

print(f'Number of elements in the Train Dataset: {len(train_dataset)}')
print(f'Number of elements in the Validation Dataset: {len(eval_dataset)}')


from transformers import get_scheduler
from accelerate import Accelerator
import torch.nn as nn
import sklearn

lr = 5e-5
epochs = 2
weight_decay = 0.0001

# Separate parameters into two groups: with and without weight decay
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if "bias" in name or "LayerNorm.weight" in name:  # Exclude biases & LayerNorm
        no_decay_params.append(param)
    else:
        decay_params.append(param)

# Create optimizer with separate parameter groups
optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0},  # No decay for these
], lr = lr)


accelerator = Accelerator()

model, optimizer, train_dataset, eval_dataset = accelerator.prepare(
    model, optimizer, train_dataset, eval_dataset
)

# we could experiment with T_max
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100)

def compute_accuracy(eval_preds):
    
    predictions, labels = eval_preds

    predictions = torch.tensor(predictions).argmax(dim=-1).view(-1).cpu().numpy()
    labels = torch.tensor(labels).view(-1).cpu().numpy()
    
    # Compute accuracy
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)

    return { 'accuracy': accuracy }


from trl import SFTConfig, SFTTrainer

logging_steps = 1

model.train()
training_args = SFTConfig(
    max_seq_length = context_length,
    output_dir = "/kaggle/working/",
    packing = True, # experiment with packing 
    fp16 = False, # gives error when turned on
    num_train_epochs = epochs,
    report_to = 'none',
    per_device_train_batch_size = bs,
    per_device_eval_batch_size = bs,
    logging_steps = logging_steps,
    load_best_model_at_end = True,
    gradient_checkpointing = True, # turn on if memory issues appear
    eval_steps = logging_steps,
    #gradient_accumulation_steps = 2,
    save_strategy = "epoch",
    eval_strategy = "epoch",
)

trainer = SFTTrainer(
    model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = data_collator,
    args = training_args,
    optimizers = (optimizer, lr_scheduler),
    compute_metrics = compute_accuracy,
)

trainer.train()


import traceback
import torch
import sys
import gc

def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()


model_dir = "/kaggle/working/deepseek_model"

model.save_pretrained(model_dir)

# disable dora before inference
if dora:
    model = model.merge_and_unload()

clean_mem()


import pandas as pd

test_dataset_dir = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"

test_dataset = pd.read_csv(test_dataset_dir)

test_dataset


def format_prompt(question, prompt_type):
    """Formats the prompt using different engineered templates."""
    
    messages = None  # Default to None to ensure all cases are covered
    
    match prompt_type:
        case "basic":
            messages = [
                {"role": "system", "content": "You are a math expert. Solve the problem and provide a final numerical answer. Don't overthink. Give answer within 100 words"},
                {"role": "user", "content": question + "\nReturn final answer within \\boxed{}, after taking modulo 1000."}
            ]
        case "step_by_step":
            messages = [
                {"role": "system", "content": "You are an expert problem solver. Solve the problem step by step and provide a final numerical answer. Don't overthink. Give answer within 100 words"},
                {"role": "user", "content": question + "\nReturn final answer within \\boxed{}, after taking modulo 1000."}
            ]
        case "concise":
            messages = [
                {"role": "system", "content": "Solve the problem and return the final numerical answer. Don't overthink. Give answer within 100 words"},
                {"role": "user", "content": question + "\nReturn final answer within \\boxed{}, after taking modulo 1000."}
            ]
        case _:
            raise ValueError("Unknown prompt type")
    
    return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)


# BASELINE OF PREVIOUS ACCURACY WAS 0
import csv 
import re

def extract_boxed_value(text):
    matches = re.findall(r'\\boxed{([^}]*)}', text)  # Find all occurrences
    if matches[-1] == '': return "None"
    return matches[-1] if matches else "None"  # Return last match, or None if no match

outputs, ids = [], []

# to avoid "Setting `pad_token_id` to `eos_token_id`:100001 for open-end generation."
model.generation_config.pad_token_id = tokenizer.pad_token_id

correct, total, nones = 0, 0, 0

for id_, prompt in test_dataset.iterrows():
    
    prompt["problem"] = format_prompt(prompt["problem"], "concise")
    
    inputs = tokenizer(prompt["problem"], return_tensors="pt", padding = True, truncation = True)

    # Move input to the same device as the model
    inputs = {key: val.to(model.device) for key, val in inputs.items()}

    answer = prompt["answer"]

    # Generate output
    with torch.no_grad():
        #output_tokens = model.generate(**inputs, **sampling_params)
        output_tokens = model.generate(**inputs, max_length = 1024, temperature = 0.7, top_p = 0.9, top_k = 30, do_sample = True)

    # Decode the generated output
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens = True)
    
    answer_text = extract_boxed_value(output_text)

    if answer_text == answer: correct += 1
    if answer_text == "None": nones += 1
    total += 1
    print(answer_text)
    outputs.append(output_text)
    ids.append(id_)

accuracy = correct / total

print("Accuracy: ", accuracy)
print("Amount of Nones in answers: ", nones)
# Save the outputs with the ids into the csv file
csv_file_name = "submission.csv"
with open(csv_file_name, "w", encoding = "utf-8", newline = "") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "answer"])

    for id_, output in zip(ids, outputs):
        writer.writerow([id_, output])


answers = pd.read_csv(csv_file_name)
print(answers['answer'][2])




