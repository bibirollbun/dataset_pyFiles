from huggingface_hub import notebook_login
notebook_login()


!pip install transformers


!pip install bitsandbytes datasets transformers accelerate peft


!pip install datasets


!pip install --upgrade torch transformers



!pip uninstall torch torchvision -y
!pip cache purge
!pip install torch torchvision --no-cache-dir


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from bitsandbytes.optim import Adam8bit
import pandas as pd
from datasets import Dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset from CSV
file_path = "/kaggle/input/mathsprobs/math_problems.csv"
dataset = pd.read_csv(file_path)

# Preprocess dataset
def preprocess_data(row):
    input_text = row['problem']
    response_text = row['answer']
    return pd.Series({
        "input_text": input_text.strip(),
        "response_text": response_text.strip()
    })

# Apply preprocessing
processed_dataset = dataset.apply(preprocess_data, axis=1)

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(processed_dataset)


# Load tokenizer
model_checkpoint = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    full_text = [
        inp + tokenizer.eos_token + resp
        for inp, resp in zip(examples["input_text"], examples["response_text"])
    ]
    tokenized = tokenizer(
        full_text,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    # Add labels, ignoring the padding token by setting it to -100
    tokenized["labels"] = [
        [label if label != tokenizer.pad_token_id else -100 for label in input_ids]
        for input_ids in tokenized["input_ids"]
    ]
    return tokenized

# Tokenize the dataset
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=hf_dataset.column_names)

split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])




import torch
torch.cuda.empty_cache()
!export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers.optimization import AdamW

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 for better precision
    bnb_4bit_compute_dtype=torch.float16  # Correct PyTorch data type for mixed precision
)

# Set device for training, ensuring it's the same device as where the model is loaded
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU availability

# Load model with 4-bit quantization and ensure it's loaded on the correct device
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    quantization_config=bnb_config,  # Apply 4-bit quantization settings
    device_map={"": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"},  # Explicitly map to the device
)

# Tokenizer (Assuming you have a tokenizer variable defined, otherwise load it)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token if it's missing

# Add LoRA configuration
lora_config = LoraConfig(
    r=8,                      # Rank of low-rank updates
    lora_alpha=32,            # LoRA scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.1,         # Dropout for LoRA layers
    bias="none",              # No additional bias
    task_type="CAUSAL_LM",    # Task type for fine-tuning
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Larger batch size due to memory efficiency
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    warmup_steps=200,
    weight_decay=0.01,
    fp16=True,                     # Mixed precision for speed (using float16)
    save_total_limit=3,
    report_to="none"
)

# Define optimizer (make sure you're using the correct optimizer, here I assume AdamW)
optimizer = AdamW(model.parameters(), lr=2e-4)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Make sure you have the correct training dataset
    eval_dataset=val_dataset,     # Make sure you have the correct evaluation dataset
    tokenizer=tokenizer,
    optimizers=(optimizer, None),  # Use optimizer, second is for scheduler
)

# Train the model
trainer.train()








