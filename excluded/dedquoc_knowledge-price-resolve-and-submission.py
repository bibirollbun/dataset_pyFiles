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


# install
!pip install accelerate
!pip install -U bitsandbytes


#!pip uninstall -y wandb


import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import os
import kaggle_evaluation.konwinski_prize_inference_server




class Config:
    base_pth = Path('/kaggle/input/konwinski-prize')
    datazip_pth = Path('//kaggle/input/konwinski-prize/data')
    working_pth = Path('/kaggle/working')
    repo_pth = Path('/kaggle/working')
    data_pth = Path('/kaggle/working/data')
    repo_pth = Path('/kaggle/working/data/repos')


!unzip -n ../input/konwinski-prize/data.a_zip


import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

# Load the dataset
issues_df = pd.read_parquet(Config.working_pth / "data/data.parquet")
issues_df = issues_df.head(10)
# Basic exploration
print(issues_df.head())
print(issues_df.info())
print(issues_df.describe())

# Visualize the target variable
sns.countplot(x='instance_id', data=issues_df)
plt.show()



df_filtered = issues_df[['instance_id', 'repo', 'problem_statement', 'patch', 'test_patch', 'pull_number', 'base_commit', 'issue_numbers']]

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df_filtered)

# Advanced dataset visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Problem statement length distribution
sns.histplot(df_filtered['problem_statement'].apply(len), bins=50, kde=True, log_scale=True, ax=axes[0, 0])
axes[0, 0].set_xlabel("Problem Statement Length")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Distribution of Problem Statement Lengths")

# Patch vs. Test Patch length comparison
patch_lengths = df_filtered[['patch', 'test_patch']].dropna()
patch_lengths['patch_length'] = patch_lengths['patch'].apply(lambda x: len(str(x)))
patch_lengths['test_patch_length'] = patch_lengths['test_patch'].apply(lambda x: len(str(x)))
sns.scatterplot(x=patch_lengths['patch_length'], y=patch_lengths['test_patch_length'], alpha=0.5, ax=axes[0, 1])
axes[0, 1].set_xlabel("Patch Length")
axes[0, 1].set_ylabel("Test Patch Length")
axes[0, 1].set_title("Patch vs. Test Patch Length Comparison")

# Repository distribution
sns.countplot(y=df_filtered['repo'], order=df_filtered['repo'].value_counts().index[:10], ax=axes[1, 0])
axes[1, 0].set_xlabel("Count")
axes[1, 0].set_ylabel("Repository")
axes[1, 0].set_title("Top 10 Repositories by Issue Count")

# Pull request number distribution
sns.histplot(df_filtered['pull_number'].dropna(), bins=50, kde=True, ax=axes[1, 1])
axes[1, 1].set_xlabel("Pull Request Number")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Distribution of Pull Request Numbers")

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of issues per repository
plt.figure(figsize=(12, 5))
sns.countplot(y=issues_df["repo"], order=issues_df["repo"].value_counts().index, palette="viridis")
plt.title("Distribution of GitHub Issues by Repository")
plt.xlabel("Count")
plt.ylabel("Repository")
plt.show()


issues_df["patch_length"] = issues_df["patch"].apply(lambda x: len(str(x)))
plt.figure(figsize=(10, 5))
sns.histplot(issues_df["patch_length"], bins=30, kde=True)
plt.title("Distribution of Patch Lengths")
plt.xlabel("Patch Length (# characters)")
plt.show()


from wordcloud import WordCloud

text = " ".join(issues_df["problem_statement"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in GitHub Issues")
plt.show()


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
# from kaggle_evaluation.konwinski_prize_inference_server import KonwinskiPrizeInference
import kaggle_evaluation.konwinski_prize_inference_server as kp


# Ensure GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Select AI model (GPT-2 or GPT-J-6B)
# Select AI model with 4-bit quantization
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
print(f"Loading tokenizer for {MODEL_NAME}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Tokenization function
def tokenize_function(examples):
    return tokenizer.batch_encode_plus(
        examples["problem_statement"], padding="longest", truncation=True, max_length=128, return_tensors="pt"
    )

dataset = dataset.map(tokenize_function, batched=True, batch_size=8)

# Load Model with 4-bit Quantization
#quant_config = BitsAndBytesConfig(load_in_4bit=True)  # Use 4-bit to reduce memory
#print(f"Loading model {MODEL_NAME}...")

# Load Model with Optimized 4-bit Quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for reduced memory
    bnb_4bit_quant_type="nf4",  # Use Normalized Float 4-bit for better accuracy
    llm_int8_enable_fp32_cpu_offload=True  # Offload computation to CPU if needed
)

print(f"Loading model {MODEL_NAME} with optimized quantization...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config, 
    device_map="auto"
)

# Attach LoRA adapters for fine-tuning
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1, 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Ensure correct tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["problem_statement"], padding="max_length", truncation=True, max_length=64
    )

dataset = dataset.map(tokenize_function, batched=True, batch_size=8)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])  # Ensure format for Trainer


# Training setup
training_args = TrainingArguments(
    output_dir="models/optimized-model",
    per_device_train_batch_size=1,  # Keep batch size small
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch
    num_train_epochs=2,
    logging_steps=50,
    fp16=True,  # Enable mixed precision
    save_total_limit=1
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=dataset
)
trainer.train()

# Save model
model.save_pretrained("fine_tuned_optimized_model")
tokenizer.save_pretrained("fine_tuned_optimized_model")


df_filtered.to_parquet("submission.parquet", index=False)

# Kaggle Evaluation API
inference_api = kp()
def generate_patch(instance):
    input_text = instance["problem_statement"]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=80).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for instance in inference_api:
    suggested_patch = generate_patch(instance)
    inference_api.submit(instance["instance_id"], suggested_patch)

inference_api.complete()


