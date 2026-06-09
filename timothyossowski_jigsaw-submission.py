!uv pip install -U --system --no-index --find-links='/kaggle/input/jigsaw-dependencies/whls/' 'bitsandbytes==0.47.0' 'accelerate==1.10.1' 'peft==0.17.1' 'trl==0.15.2' 'triton==3.4.0' 'cut_cross_entropy==25.1.1' 'unsloth_zoo' 'sentencepiece==0.2.1' 'protobuf==6.32.0' 'unsloth' 'pillow==11.3.0' 'transformers' 'numpy<2'


%%writefile train_model.py
lock_path = "/tmp/fastlm_lock.lock"
from filelock import FileLock
with FileLock(lock_path):
    from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader

import random
import pandas as pd
import numpy as np
import torch
import os
import time
import sys

os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'

user_prompt = """<subreddit>r/{subreddit}</subreddit>
<rule>{rule}</rule>
<comment>{body}</comment>
Does the comment violate the rule? Answer with just yes/no
"""

TEMPLATES = {
    "/kaggle/input/unsloth-mistral-7b-instruct-v0.3/pytorch/1/1": {
        "system": "",
        "instruction_part": "<s>[INST]",
        "response_part": "[/INST]"
    },
    "/kaggle/input/gemma-2-9b-it-bnb-4bit/pytorch/1/1" : {
        "system": "",
        "instruction_part": "<bos><start_of_turn>user",
        "response_part": "<start_of_turn>model"
    },
    "/kaggle/input/unsloth-qwen3-8b/pytorch/1/1": {
        "system": "",
        "instruction_part": "<|im_start|>user",
        "response_part": "<|im_start|>assistant"
    },
    "/kaggle/input/unsloth-llama-3.1-8b-instruct/pytorch/1/1": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|>",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>"
    }
}

partition_num = sys.argv[1] if len(sys.argv) > 1 else ''

if partition_num == '0':
    LR = 1.5e-4
    RULE_LIMITS = {
        0: 3000,
        1: 3000,
        2: 3000,
        3: 3000,
        4: 3000,
        5: 3000
    }
elif partition_num == '1':
    LR = 1.5e-4
    RULE_LIMITS = {
        0: 2000,
        1: 4000,
        2: 4000,
        3: 4000,
        4: 2000,
        5: 2000
    }
elif partition_num == '2':
    LR = 1.5e-4
    RULE_LIMITS = {
        0: 4000,
        1: 2000,
        2: 2000,
        3: 2000,
        4: 4000,
        5: 4000
    }
elif partition_num == '3':
    LR = 1.5e-4
    RULE_LIMITS = {
        0: 3000,
        1: 3000,
        2: 3000,
        3: 3000,
        4: 3000,
        5: 3000
    }


model_name_or_path = '/kaggle/input/gemma-2-9b-it-bnb-4bit/pytorch/1/1'
seed = int(partition_num)
random.seed(seed)
print(f"Processing partition {partition_num} ...")

max_seq_length = 400
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name_or_path,
    max_seq_length = max_seq_length,
    dtype = None,  # Let unsloth handle dtype automatically
    load_in_4bit = True,
    device_map = None
)

yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_token_id  = tokenizer.encode("no", add_special_tokens=False)[0]

# Ensure a pad token exists for proper padding during collation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    
    test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv', index_col='row_id')
else:
    LIMIT_PER_RULE = 100
    test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv', index_col='row_id')

sys_prompt = ''
training_data = []
test_data = []
unique_questions = set()
unique_rule_examples = {}
rules = sorted(test['rule'].unique())

for i, row in test.sample(frac=1, random_state=seed).iterrows():
    for key in ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2', 'body']:
        if key == 'body' and 'rule_violation' not in row:
            continue

        prompt = user_prompt.format(
            subreddit=row['subreddit'],
            rule=row['rule'],
            body=row[key]
        )

        if prompt in unique_questions:
            continue
        unique_questions.add(prompt)
        
        entry = {}
        entry['instruction'] = prompt

        if key == 'body':
            entry['output'] = "yes" if row['rule_violation'] == 1 else "no"
        else:
            entry['output'] = "yes" if 'positive' in key else "no"

        if row['rule'] not in unique_rule_examples:
            unique_rule_examples[row['rule']] = set()

        rule_idx = rules.index(row['rule'])
        unique_rule_examples[row['rule']].add(prompt)
        if len(unique_rule_examples[row['rule']]) > RULE_LIMITS[rule_idx]:
            continue
        
        training_data.append(entry)

for i, row in test.iterrows():
    test_prompt = user_prompt.format(
        subreddit=row['subreddit'],
        rule=row['rule'],
        body=row['body']
    )

    entry = {'instruction': test_prompt}
    test_data.append(entry)

dataset = [[{"role": "user", "content": entry['instruction'] },
            {"role": "assistant", "content": entry['output']}] for entry in training_data]


def formatting(dataset):
    texts = []
    for i in range(len(dataset)):
        prompt = tokenizer.apply_chat_template(dataset[i], tokenize=False, add_generation_prompt=False, enable_thinking=False).replace(sys_prompt, "")
        texts.append(prompt)
    return Dataset.from_dict({'text': texts})

random.shuffle(dataset)
filename = f'submission_{partition_num}.csv'

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,    
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = seed,
    use_rslora = False,  
    loftq_config = None, 
)

dataset = formatting(dataset)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate = LR,
        warmup_steps = 0,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        num_train_epochs=1,
        seed = seed,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir=None,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=True
    ),
)

# Now this works
trainer = train_on_responses_only(
    trainer,
    instruction_part = TEMPLATES[model_name_or_path]['instruction_part'],
    response_part    = TEMPLATES[model_name_or_path]['response_part']
)

print("Starting training...")
trainer_stats = trainer.train()

model.save_pretrained(f'lora_model_{partition_num}')
tokenizer.save_pretrained(f'lora_model_{partition_num}')



%%writefile infer.py
lock_path = "/tmp/fastlm_lock.lock"
from filelock import FileLock
with FileLock(lock_path):
    from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import torch
import os
import time
import sys

os.environ['UNSLOTH_DISABLE_STATISTICS'] = '1'

user_prompt = """<subreddit>r/{subreddit}</subreddit>
<rule>{rule}</rule>
<comment>{body}</comment>
Does the comment violate the rule? Answer with just yes/no
"""

partition_num = sys.argv[1]
model_name_or_path = f'lora_model_{partition_num}'
#model_name_or_path = "/kaggle/input/unsloth-llama-3.1-8b-instruct/pytorch/1/1"
seed = int(partition_num)
random.seed(seed)
print(f"Processing partition {partition_num} ...")

max_seq_length = 512
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f'lora_model_{partition_num}',
    max_seq_length = max_seq_length,
    dtype = None,  # Let unsloth handle dtype automatically
    load_in_4bit = True,
    device_map = None
)

yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_token_id  = tokenizer.encode("no", add_special_tokens=False)[0]

# Ensure a pad token exists for proper padding during collation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv', index_col='row_id')


sys_prompt = ""


test_data = []

for i, row in test.iterrows():
    test_prompt = user_prompt.format(
        subreddit=row['subreddit'],
        rule=row['rule'],
        body=row['body']
    )
    
    entry = {'instruction': test_prompt}
    test_data.append(entry)


filename = f'submission_{partition_num}.csv'
dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

# Move model to correct dtype and enable fast inference
model = model.to(dtype=dtype)
model = FastLanguageModel.for_inference(model)
model.config.use_cache = True

# Pre-tokenize prompts WITHOUT padding
batch_texts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": entry['instruction']}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ).replace(sys_prompt, "")
    for entry in test_data
]

# Tokenize WITHOUT padding
input_ids_list = [tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0) for text in batch_texts]
attention_masks_list = [torch.ones_like(x, dtype=torch.long) for x in input_ids_list]
lengths = [len(x) for x in input_ids_list]

# Sort by sequence length (descending)
sorted_indices = np.argsort([-l for l in lengths])
sorted_inputs = [input_ids_list[i] for i in sorted_indices]
sorted_masks = [attention_masks_list[i] for i in sorted_indices]

# Keep a reverse map to restore original order later
unsort_indices = np.argsort(sorted_indices)

# Dynamic padding per batch
def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    max_len = max(x.size(0) for x in input_ids)  # pad to max length in batch
    input_ids = torch.stack([torch.cat([x, x.new_full((max_len - x.size(0),), tokenizer.pad_token_id)]) for x in input_ids])
    attention_masks = torch.stack([torch.cat([m, m.new_zeros((max_len - m.size(0),))]) for m in attention_masks])
    return input_ids, attention_masks

dataset = list(zip(sorted_inputs, sorted_masks))
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=2, pin_memory=True)

# Run inference
all_probs_sorted = []

start_time = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    for input_ids, attention_mask in tqdm(loader, desc="Generating"):
        input_ids = input_ids.to("cuda", non_blocking=True)
        attention_mask = attention_mask.to("cuda", non_blocking=True)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        logits = outputs.scores[0]  # (batch_size, vocab_size)
        logits = logits[:, [no_token_id, yes_token_id]]
        probs = torch.softmax(logits, dim=-1)
        all_probs_sorted.append(probs)

all_probs_sorted = torch.cat(all_probs_sorted, dim=0)

# Reorder to original order
all_probs = all_probs_sorted[unsort_indices].cpu().numpy()

end_time = time.time()
print(f"Inference took {end_time - start_time:.2f}s for {len(test_data)} samples.")


probs = [x[1].item() for x in all_probs]
sub['rule_violation'] = probs

sub.to_csv(filename, index=True)


import os, time, subprocess, pandas as pd

def wait_then_read(outfile, proc, settle_time=1.0, read_if_failed=True):
    """
    Wait for the process to finish.
    - If it failed (non-zero exit), optionally read outfile only if it already exists and is stable.
    - If it succeeded (zero exit), give a brief settle_time then read, but do NOT wait forever.
    Returns a DataFrame or None.
    """
    rc = proc.wait()

    if os.path.exists(outfile):
        time.sleep(5)
        if outfile.endswith(".csv"):
            return pd.read_csv(outfile)

# Training Models 1 and 2
# Start both jobs (capture logs so you can inspect failures)
log0 = open("job0.log", "w")
log1 = open("job1.log", "w")

proc0 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python train_model.py 0"],
     stdout=log0, stderr=subprocess.STDOUT
)
proc1 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=1 /usr/local/bin/python train_model.py 1"],
     stdout=log1, stderr=subprocess.STDOUT
)

wait_then_read("lora_model_0", proc0)
wait_then_read("lora_model_1", proc1)

log0.close(); log1.close()

print("Finished Training, check logs for debugging")

# Start both jobs (capture logs so you can inspect failures)
log2 = open("job2.log", "w")
log3 = open("job3.log", "w")

proc2 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python infer.py 0"],
     stdout=log2, stderr=subprocess.STDOUT
)
proc3 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=1 /usr/local/bin/python infer.py 1"],
     stdout=log3, stderr=subprocess.STDOUT
)

df0 = wait_then_read("submission_0.csv", proc2)
df1 = wait_then_read("submission_1.csv", proc3)

log2.close(); log3.close()

if df0 is not None and df1 is not None:
    print("Successfully did inference")
else:
    print("One or both inferences failed...")



log4 = open("job4.log", "w")
log5 = open("job5.log", "w")

proc4 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python train_model.py 2"],
     stdout=log4, stderr=subprocess.STDOUT
)
proc5 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=1 /usr/local/bin/python train_model.py 3"],
     stdout=log5, stderr=subprocess.STDOUT
)

wait_then_read("lora_model_2", proc4)
wait_then_read("lora_model_3", proc5)

log4.close(); log5.close()

print("Finished Training, check logs for debugging")

# Start both jobs (capture logs so you can inspect failures)
log6 = open("job6.log", "w")
log7 = open("job7.log", "w")

proc6 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=0 /usr/local/bin/python infer.py 2"],
     stdout=log6, stderr=subprocess.STDOUT
)
proc7 = subprocess.Popen(
    ["bash", "-lc", "CUDA_VISIBLE_DEVICES=1 /usr/local/bin/python infer.py 3"],
     stdout=log7, stderr=subprocess.STDOUT
)

df2 = wait_then_read("submission_2.csv", proc6)
df3 = wait_then_read("submission_3.csv", proc7)

log6.close(); log7.close()

if df2 is not None and df3 is not None:
    print("Successfully did inference")
else:
    print("One or both inferences failed...")



!cat job0.log


import pandas as pd
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv', index_col='row_id')
sub = pd.read_csv('submission_0.csv', index_col='row_id')
df_1 = pd.read_csv('submission_1.csv', index_col='row_id')
df_2 = pd.read_csv('submission_2.csv', index_col='row_id')
df_3 = pd.read_csv('submission_3.csv', index_col='row_id')

sub["rule_violation"] = 0.25 * sub["rule_violation"] + 0.25 * df_1["rule_violation"] + 0.25 * df_2["rule_violation"] + 0.25 * df_3["rule_violation"]

combined = pd.concat([train_df, test], ignore_index=True)
examples = {'positive': set(), 'negative': set()}
rules = sorted(test['rule'].unique())

for _, row in combined.iterrows():
    # Add positive examples
    examples['positive'].update([row['positive_example_1'], row['positive_example_2']])

    # Add negative examples
    examples['negative'].update([row['negative_example_1'], row['negative_example_2']])

    # Add body based on rule_violation
    if 'rule_violation' in row:
        if row['rule_violation'] == 0:
            examples['negative'].add(row['body'])
        elif row['rule_violation'] == 1:
            examples['positive'].add(row['body'])

# --- Remove conflicts: anything in both sets ---
conflicts = examples['positive'] & examples['negative']
examples['positive'] -= conflicts
examples['negative'] -= conflicts

print(f"Removed {len(conflicts)} conflicting examples.")

# Create the new column for pseudolabels
def get_pseudolabels(row):
    body = row['body']
    
    if body in examples['positive']:
        return 1
    elif body in examples['negative']:
        return -1
    
    return 0

test['pseudo_label'] = test.apply(get_pseudolabels, axis=1)

# Check the results
print("Value counts for column pseudo_label:")
print(test['pseudo_label'].value_counts())

assert (test['pseudo_label'].value_counts() != 0).any(), "All counts are zero!"

# Combine with offline predictions and exact matching
legal_predictions = pd.read_csv('/kaggle/input/million_data_with_predictions/pytorch/1/20/merged_legal_predictions.csv')
print(legal_predictions.columns)
exclude_cols = ['subreddit', 'body']

# Select only the columns to average
cols_to_average = legal_predictions.drop(columns=exclude_cols)

# Compute the row-wise average
averages = cols_to_average.mean(axis=1)

# Create keys as (subreddit, body) pairs
keys = list(zip(legal_predictions['subreddit'], legal_predictions['body']))

# Create the dictionary
body_to_violation = dict(zip(keys, averages))

# Make a key column for easy mapping
test["key"] = list(zip(test["subreddit"], test["body"]))

# Map rule_violation values from your body_to_violation dictionary
mapped = test["key"].map(body_to_violation)

# Condition: same rule
same_rule_mask = test["rule"] == "No legal advice: Do not offer or request legal advice."

# Apply averaging only where rule matches and mapping exists
sub.loc[same_rule_mask & mapped.notna(), "rule_violation"] = (
    mapped[same_rule_mask & mapped.notna()] + sub.loc[same_rule_mask & mapped.notna(), "rule_violation"]
) / 2

# Apply pseudo_label adjustments in a vectorized way
sub.loc[test["pseudo_label"] == -1, "rule_violation"] = 0
sub.loc[test["pseudo_label"] == 1, "rule_violation"] = 1

sub.to_csv('submission.csv')

