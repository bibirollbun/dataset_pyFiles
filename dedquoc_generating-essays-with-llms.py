import pandas as pd
import numpy as np
import random
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


# Clear GPU memory
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


clear_gpu_memory()


df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns)
df.head()


# Ensure 'topic' column exists and rename it to 'prompt' for consistency
if 'topic' in df.columns:
    df.rename(columns={'topic': 'prompt'}, inplace=True)
elif 'prompt' not in df.columns:
    raise KeyError("Neither 'prompt' nor 'topic' found in dataset. Check the CSV file format.")

# recheck datasets columes
print("Dataset Columns:", df.columns)
df.head()


# Clear GPU memory and delete existing objects if they exist
if torch.cuda.is_available():
    torch.cuda.empty_cache()
for obj in ['model', 'pipe', 'tokenizer']:
    if obj in globals():
        del globals()[obj]

# Load LLM model and tokenizer
model_name = "/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1"  # Change to "google/gemma-2" if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    # model = torch.nn.DataParallel(model).to(device)
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).to(device)

clear_gpu_memory()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure model parameters are on the same device
for param in model.parameters():
    param.requires_grad = False  # Ensure model is in inference mode
    param.data = param.data.to(device)


def generate_essay(prompt):
    """
    Generate an essay using an LLM model to maximize disagreement among judges.
    """
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids
    input_ids = input_ids.to(device)  # Move input tensors to the same device as the model
    
    with torch.no_grad():
        if isinstance(model, torch.nn.DataParallel):
            model_device = next(model.module.parameters()).device
            input_ids = input_ids.to(model_device)
            model.module.to(model_device)  # Ensure model is on the same device
            output = model.module.generate(input_ids, max_length=200)
        else:
            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            model.to(model_device)  # Ensure model is on the same device
            output = model.generate(input_ids, max_length=200)
    
    essay = tokenizer.decode(output[0], skip_special_tokens=True)
    return essay

# Generate sample essays
df['essay'] = df['prompt'].apply(generate_essay)

# Save submission file
df[['id', 'essay']].to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")

clear_gpu_memory()

