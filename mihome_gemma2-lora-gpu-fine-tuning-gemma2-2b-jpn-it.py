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



!pip uninstall preprocessing -y
!pip install nltk==3.2.4

!pip install huggingface-hub==0.23.5
!pip install peft==0.12.0
!pip install llama-index-llms-huggingface==0.4.0



import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoConfig
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.box import ROUNDED

os.environ["WANDB_DISABLED"] = "true"  # Disable WANDB

# Set the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer (specifying local path)
def load_model_and_tokenizer(model_dir, fine_tuned=False):
    """
    Load model and tokenizer, transfer them to the device.
    If fine_tuned is True, load the fine-tuned model.
    """
    if fine_tuned:
        config = AutoConfig.from_pretrained(model_dir)  # Load the config
        model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, attn_implementation='eager').to(device)  # Specify eager attention
    else:
        config = AutoConfig.from_pretrained(model_dir)  # Load the config
        model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, attn_implementation='eager').to(device)  # Specify eager attention
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Display model information
    print(f"Loaded model from {model_dir}")
    print(f"Model type: {type(model)}")

    return model, tokenizer

# LoRA configuration (parameters can be adjusted externally)
def apply_lora(model, rank=8, alpha=32, dropout=0.1, target_modules=["q_proj", "v_proj"]):
    """
    Apply LoRA to the model. Parameters can be adjusted externally.
    LoRA Rank: 16 to 32 (use 4 to 8 for simpler tasks)
    LoRA Alpha: 8 to 32
    LoRA Dropout: 0.1 to 0.3
    LoRA Target Modules: Based on the model structure (e.g., ["attention", "feedforward"])
    """
    lora_config = LoraConfig(
        r=rank,  # LoRA Rank
        lora_alpha=alpha,  # LoRA Alpha
        target_modules=target_modules,  # Modules to apply LoRA to
        lora_dropout=dropout  # Dropout rate for LoRA layers
    )
    model = get_peft_model(model, lora_config)
    return model

# Load and process data from the JSON file and tokenize it
def load_and_process_data(file_path, tokenizer):
    """
    Load and tokenize data from the provided JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # Structure the data
    data = [
        f"Instruction: {item['instruction']}\nResponse: {item['output']}"
        for item in dataset
    ]

    # Tokenize the data
    encodings = tokenizer(data, truncation=True, padding=True, max_length=512)
    
    # Convert to PyTorch dataset format
    dataset = Dataset.from_dict(encodings)
    
    # Add labels
    dataset = dataset.map(lambda x: {'labels': x['input_ids']}, batched=True)
    
    return dataset

# Function to generate and display text based on a prompt
def generate_and_display(prompt, model, tokenizer, max_length=512):
    """
    Generate text based on the given prompt and display the result using rich panels.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(inputs['input_ids'], max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Display prompt
    prompt_panel = Panel(
        Text(prompt, style="bold magenta"),
        title="[blue]Input Prompt[/blue]",
        border_style="blue",
        box=ROUNDED,
    )
    
    # Display generated text
    generated_md = Markdown(generated_text)
    token_count = len(generated_text.split())
    token_info = Text(f"\n\nGenerated {token_count} tokens.", style="italic cyan")
    
    # Display output panel
    output_panel = Panel(
        generated_md,
        title="[green]Generated Response[/green]",
        border_style="green",
        box=ROUNDED,
    )
    
    # Console output
    console = Console()
    console.print(prompt_panel)
    console.print()
    console.print(output_panel)
    console.print(token_info)

# Function for testing text generation with given prompts
def test_generation(model, tokenizer):
    """
    Test text generation based on given prompts and display the results.
    """
    print("\n--- Generation Results ---")
    
    """
    prompts = [
        "When did Virgin Australia Airlines start operating?",
        "Why can camels live for long periods without water?",
        "Alice's parents have three daughters: Amy, Jessie, and what is the name of the third daughter?"
    ]
    """
    
    prompts = [
    "ヴァージン・オーストラリア航空はいつから運航を開始したのですか？",
    "ラクダはなぜ水なしで長く生きられるのか？",
    "アリスの両親には3人の娘がいる：エイミー、ジェシー、そして三女の名前は？"
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(inputs['input_ids'], max_length=512, num_return_sequences=1)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}\n")

# Fine-tune the model
def fine_tune(model, tokenizer, train_data, eval_data, output_dir="./output_model", rank=8, alpha=32, dropout=0.1, target_modules=["q_proj", "v_proj"]):
    """
    Fine-tune the model.
    """
    # Model information before fine-tuning
    print("--- Before Fine-tuning ---")
    print(f"Model type before fine-tuning: {type(model)}")

    # Apply LoRA
    model = apply_lora(model, rank, alpha, dropout, target_modules)

    # Trainer configuration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Number of epochs
        per_device_train_batch_size=1,  # Batch size
        logging_dir="./logs",  # Log directory
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        warmup_steps=100,  # Number of warmup steps
        weight_decay=0.01,  # Weight decay
        lr_scheduler_type="linear",  # Learning rate scheduling
        eval_strategy="steps",  # Evaluation strategy
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="steps",  # Save the model periodically
        run_name="fine_tuning_run",  # Run name
        report_to="none",  # Disable WANDB reporting

        learning_rate=0.0005  # Learning rate (default is 0.0001)
    )

    # Data collator configuration
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Fine-tuning process
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")

    # Model information after fine-tuning
    print("\n--- After Fine-tuning ---")
    print(f"Model type after fine-tuning: {type(model)}")

# Interface for choosing the model
def choose_model():
    """
    Interface to choose the model.
    """
    model_choice = input("Choose the model to use (0: Original, 1: Fine-tuned Model): ")
    if model_choice == "1":
        fine_tuned_model_dir = './fine_tuned_model'  # Directory where the fine-tuned model is saved
        model, tokenizer = load_model_and_tokenizer(fine_tuned_model_dir, fine_tuned=True)
    else:
        original_model_dir = '/kaggle/input/gemma2-2b-jpn-it-m/pytorch/default/1/gemma2-2b-JPN-IT/'  # Original model directory
        model, tokenizer = load_model_and_tokenizer(original_model_dir, fine_tuned=False)
    
    return model, tokenizer

def main():
    model, tokenizer = choose_model()  # Select model

    # Test generation with the selected model
    test_generation(model, tokenizer)

    # Load and process the databricks-dolly-15k-ja-json data
    file_path = '/kaggle/input/databricks-dolly-15k-ja3-json/databricks-dolly-15k-ja3.json'  # Path to training data file
    train_data = load_and_process_data(file_path, tokenizer)

    # Create the evaluation dataset
    eval_data = train_data.train_test_split(test_size=0.1)["test"]  # 90% for training, 10% for evaluation

    # LoRA settings (default values)
    rank = 16  # LoRA rank
    alpha = 64  # LoRA alpha
    dropout = 0.2  # Dropout rate
    target_modules = ["q_proj", "v_proj"]  # Modules to apply LoRA to

    # Fine-tune the model
    fine_tune(model, tokenizer, train_data, eval_data, output_dir="./fine_tuned_model", rank=rank, alpha=alpha, dropout=dropout, target_modules=target_modules)

    # Test generation again after fine-tuning
    test_generation(model, tokenizer)

if __name__ == "__main__":
    main()


