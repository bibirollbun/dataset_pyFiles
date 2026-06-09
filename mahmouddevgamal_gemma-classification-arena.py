# !pip install transformers peft accelerate bitsandbytes \
#     -U --no-index --find-links /kaggle/input/lmsys-wheel-files
# !pip install bitsandbytes -U --no-index --find-links /kaggle/input/bitsandbytes-0-45-0


import time
import warnings
warnings.filterwarnings("ignore")

import logging

# Suppress warnings from the transformers library
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


%%time
from transformers import (pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
                          AutoConfig, BitsAndBytesConfig, TextDataset, DataCollatorWithPadding)
from datasets import Dataset
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import PeftModel, get_peft_model, LoraConfig, TaskType


# Ensure GPU utilization
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


%%time
train_data = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet")
test_data = pd.read_parquet("/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet")

# Encode labels
train_data['label'] = train_data['winner'].map({'model_a': 0, 'model_b': 1})

# Preprocess function
def preprocess_function(row):
    return {
        "input_text": f"Prompt: {row['prompt']} | Response A: {row['response_a']} | Response B: {row['response_b']}",
        "labels": 0 if 'model_a' == row['winner'] else 1
    }

processed_data = train_data.apply(preprocess_function, axis=1)
df = pd.DataFrame(processed_data.tolist())

# Split into train and evaluation datasets
train_data, eval_data = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)


%%time

model_path = "/kaggle/input/gemma/transformers/2b-it/3"

# Optional: Configure quantization (uncomment if needed and supported)
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Load base model configuration
config = AutoConfig.from_pretrained(model_path)
config.hidden_activation = "gelu"
config.use_cache = False
config.num_labels = 2
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    config=config,
    # quantization_config=quantization_config,
    device_map="auto",
    ignore_mismatched_sizes=True
)

# Verify if meta tensors exist and initialize them
if any(param.device.type == "meta" for param in model.parameters()):
    print("Meta tensors found. Initializing...")
    model.tie_weights()  # Tie weights, ensures shared weights are properly linked after initialization.
    model = model.to_empty(device=device)  #replaces meta tensors with empty tensors, clearing the "meta" state and making the model ready for proper initialization or loading onto a device.
    model = model.to(device)  # Move to CUDA device

# Print model to get hidden layer names, to use it later in LoRA configuration
print(model)

# Adjust tokenizer and model
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))


# Apply LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)


# subset data to debug
subset_size = 5000  # Number of examples to use for training
train_dataset = train_dataset.shuffle(seed=42).select(range(subset_size))
subset_size = int(subset_size*2/8)  # Number of examples to use for test, keep its ratio 8 to 2
eval_dataset = eval_dataset.shuffle(seed=42).select(range(subset_size))


%%time
train_dataset = train_dataset.map(lambda x: tokenizer(x['input_text'], padding=True, truncation=True, max_length=512), batched=True)
eval_dataset = eval_dataset.map(lambda x: tokenizer(x['input_text'], padding=True, truncation=True, max_length=512), batched=True)


train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    # fp16_opt_level="O2",
    report_to="none",
    warmup_steps=500,
    save_steps=500,
    gradient_checkpointing=True
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


torch.cuda.empty_cache()


%%time
trainer.train()


torch.cuda.empty_cache()


# model.save_pretrained("./trained_gemma_predictor_model")
# tokenizer.save_pretrained("./trained_gemma_predictor_model")


def predict(prompt, response_a, response_b):
    input_text = f"Prompt: {prompt} | Response A: {response_a} | Response B: {response_b}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "model_a" if probabilities[0][0] > probabilities[0][1] else "model_b"


# Section 9: Evaluate Responses
def evaluate_responses(test_data):
    results = []
    for _, row in test_data.iterrows():
        results.append({
            'id': row["id"],
            'winner': predict(row["prompt"], row["response_a"], row["response_b"])
        })
    return pd.DataFrame(results)


results_df = evaluate_responses(test_data)
results_df.to_csv("submission.csv", index=False)
print(results_df.head())

