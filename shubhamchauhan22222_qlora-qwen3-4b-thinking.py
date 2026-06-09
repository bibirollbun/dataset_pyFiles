# Install necessary libraries
!pip install -q -U bitsandbytes
!pip install -q -U transformers
!pip install -q -U peft
!pip install -q -U accelerate
!pip install -q -U datasets


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
model_name = "Qwen/Qwen3-4B-Thinking-2507"
EPOCHS = 4

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-math/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


# Define the model name
model_name = "Qwen/Qwen3-4B-Thinking-2507"


import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256


def format_input(row):
    # x = "This answer is correct."
    # if not row['is_correct']:
    #     x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        # f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0])


lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )


# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


# Define the QLoRA quantization configuration
# The corrected configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)


import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Qwen models usually don't have a pad token, so we set it to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Load the 4-bit quantized model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    quantization_config=quantization_config,
    device_map="auto", # Automatically maps model layers to available devices (GPU/CPU)
    trust_remote_code=True,
)

# The model's config might not have pad_token_id, set it from the tokenizer
model.config.pad_token_id = tokenizer.pad_token_id


from peft import LoraConfig, get_peft_model, TaskType

# Define the LoRA configuration
lora_config = LoraConfig(
    r=16,  # The rank of the LoRA matrices. A higher rank means more trainable parameters.
    lora_alpha=32, # A scaling factor for the LoRA matrices.
    lora_dropout=0.1, # Dropout probability for LoRA layers.
    bias="none",
    task_type=TaskType.SEQ_CLS, # Specify the task type for sequence classification
    # Target modules for Qwen3 models
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# Wrap the base model with the PEFT model
model = get_peft_model(model, lora_config)

for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()
        
# Print the number of trainable parameters
model.print_trainable_parameters()


training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS, # Keep your epochs, 3 is a good start
    per_device_train_batch_size=4,  # CRITICAL: Use a small batch size (1 or 2)
    gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
    per_device_eval_batch_size=4,
    learning_rate=2e-3, # A common learning rate for LoRA
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True, # MUST be True for T4/P100 GPUs
    gradient_checkpointing=True, # Saves more memory
    optim="adamw_8bit",# A compatible, memory-efficient optimizer
# ðŸ”¥ Learning rate scheduler
    lr_scheduler_type="linear",   # linear decay
    warmup_steps=300              # optional warmup
)


# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

trainer.train()


trainer.save_model(f"ver_{VER}")      
tokenizer.save_pretrained(f"ver_{VER}")










