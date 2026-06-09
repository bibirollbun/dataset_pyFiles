%%capture
%pip install -U transformers datasets accelerate peft trl bitsandbytes


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

import os
import torch
import bitsandbytes as bnb

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, setup_chat_format


base_model = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2/"
dataset_name = "maharnab/hindi_instruct"
new_model = "Gemma-2-2b-it-hindi"


# Check CUDA device capability and set appropriate configurations
# Flash Attention v2 requires CUDA device capability >= 8.0
if torch.cuda.get_device_capability()[0] >= 8:
    # Install Flash Attention if capability allows
    !pip install -qqq flash-attn
    torch_dtype = torch.bfloat16               # Use bfloat16 precision for better performance on supported hardware
    attn_implementation = "flash_attention_2"  # Use Flash Attention v2
else:
    torch_dtype = torch.float16    # Use float16 for older hardware
    attn_implementation = "eager"  # Default attention implementation


# Configuration for Quantized LoRA (QLoRA)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # Enable 4-bit quantization for efficient model loading
    bnb_4bit_quant_type="nf4",           # Use NormalFloat4 (NF4) quantization
    bnb_4bit_compute_dtype=torch_dtype,  # Set computation precision based on hardware support
    bnb_4bit_use_double_quant=True       # Use double quantization for improved accuracy
)


# Load the pretrained causal language model with quantization configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model,                              # The base model identifier or path
    quantization_config=quantization_config,          # Apply QLoRA configuration
    device_map="auto",                       # Automatically map model to available devices
    attn_implementation=attn_implementation  # Set attention implementation
)

# Load the tokenizer corresponding to the pretrained model
tokenizer = AutoTokenizer.from_pretrained(
    base_model,             # The base model identifier or path
    trust_remote_code=True  # Trust custom tokenizer code if provided by the model
)


def find_all_linear_names(model):
    """
    This function searches for all linear layers of the 4-bit format 
    in a given model and returns their names, excluding the 'lm_head' 
    module if present.

    Args:
    - model: The model to search for linear layers in.

    Returns:
    - List of module names associated with linear layers.
    """
    # The target class for linear layers (4-bit format)
    cls = bnb.nn.Linear4bit
    lora_module_names = set()  # Set to hold the unique names of the target linear modules

    # Iterate over all named modules in the model
    for name, module in model.named_modules():
        # Check if the module is of the target class
        if isinstance(module, cls):
            names = name.split('.')  # Split the module name by dots to isolate components
            # Add the first or last part of the name (depending on the structure) to the set
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove 'lm_head' if present in the set (needed for 16-bit models)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    # Return the list of linear module names
    return list(lora_module_names)

# Get the list of linear module names in the model
modules = find_all_linear_names(model)


# LoRA configuration setup
peft_config = LoraConfig(
    r=16,                    # Rank for LoRA
    lora_alpha=32,           # Scaling factor for LoRA
    lora_dropout=0.05,       # Dropout rate for LoRA
    bias="none",             # No bias in LoRA layers
    task_type="CAUSAL_LM",   # Task type for causal language modeling
    target_modules=modules   # The list of target modules (linear layers)
)


# Set the padding side for the tokenizer (important for certain models)
tokenizer.padding_side = 'right'

# Reset chat template to ensure no leftover settings
tokenizer.chat_template = None

# Setup chat format with the model and tokenizer
model, tokenizer = setup_chat_format(model, tokenizer)

# Apply LoRA configurations to the model
model = get_peft_model(model, peft_config)


import json
from datasets import load_dataset


# Load datasets
instruct_hindi = load_dataset("OdiaGenAI/instruction_set_hindi_1035", split='all')
conv_hindi = load_dataset("SherryT997/HelpSteer-hindi", split='all')
history_hindi = load_dataset("kaifahmad/indian-history-hindi-QA-3.4k", split='all')
health_hindi = load_dataset("OdiaGenAI/health_hindi_200", split='all')


# Function to reformat datasets
def reformat_data(dataset, instruction_key, output_key):
    reformatted = []
    for entry in dataset:
        reformatted.append({
            "instruction": "You are a helpful assistant.",
            "input": entry[instruction_key],
            "output": entry[output_key]
        })
    return reformatted

# Function to reformat conversation datasets
def reformat_conversation_data(dataset):
    reformatted = []
    for conversation in dataset['conversations']:
        for i in range(len(conversation) - 1):  # Iterate over all conversation turns
            if conversation[i]['from'] == 'human' and conversation[i + 1]['from'] == 'gpt':
                reformatted.append({
                    "instruction": "You are a helpful assistant.",
                    "input": conversation[i]['value'],      # Human input
                    "output": conversation[i + 1]['value']  # GPT's response
                })
    return reformatted

# Reformat each dataset
instruct_data = reformat_data(instruct_hindi, "Instruction", "Output")
health_data = reformat_data(health_hindi, "Instruction", "Output")
history_data = reformat_data(history_hindi, "Question", "Answer")
conversation_data = reformat_conversation_data(conv_hindi)


# Combine all datasets
combined_data = instruct_data + health_data + history_data + conversation_data

# Write to JSONL file
output_file = "hindi_dataset.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for entry in combined_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Combined dataset saved to {output_file}")


# Importing the dataset and shuffling it for randomness
dataset = load_dataset(dataset_name, split="all")  # Load the entire dataset
dataset = dataset.shuffle(seed=3117)               # Shuffle the dataset with a fixed seed for reproducibility


def format_chat_template(row):
    """
    This function formats each row of the dataset into a chat-like structure 
    and applies the chat template for tokenization.

    Args:
    - row: The current dataset row, containing 'instruction', 'input', and 'output'.

    Returns:
    - The updated row with a 'text' field containing the formatted chat template.
    """
    # Construct a JSON-like structure for the chat conversation (system, user, assistant)
    row_json = [
        {"role": "system", "content": row["instruction"]},  # System message: the instruction
        {"role": "user", "content": row["input"]},          # User message: the input question
        {"role": "assistant", "content": row["output"]}     # Assistant message: the model's response
    ]
    # Apply the tokenizer to format the row using the chat template without tokenizing
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Apply the chat formatting to the entire dataset using multiple processes (num_proc=4 for parallelism)
dataset = dataset.map(format_chat_template, num_proc=4)


# Split the dataset into training and test sets (90% train, 10% test)
dataset = dataset.train_test_split(test_size=0.1)


# Setting Hyperparameters for Training
trainer = SFTTrainer(
    model=model,                     # The model to be trained
    processing_class=tokenizer,      # The tokenizer used for data processing
    train_dataset=dataset["train"],  # Training dataset
    eval_dataset=dataset["test"],    # Evaluation dataset
    peft_config=peft_config,         # LoRA configuration for model adaptation
    args=SFTConfig(
        output_dir=new_model,                                   # Directory where the trained model will be saved
        per_device_train_batch_size=1,                          # Batch size for training
        per_device_eval_batch_size=1,                           # Batch size for evaluation
        gradient_accumulation_steps=2,                          # Number of steps for gradient accumulation
        optim="paged_adamw_32bit",                              # Optimizer type for training
        num_train_epochs=1,                                     # Number of training epochs
        eval_strategy="steps",                                  # Evaluation strategy during training
        eval_steps=int(len(dataset["train"]) // (1 * 2) // 5),  # Frequency of evaluation in steps
        logging_steps=10,                                       # Frequency of logging during training
        warmup_steps=30,                                        # Number of steps for learning rate warmup
        logging_strategy="steps",                               # Logging strategy to use (log every 'steps' steps)
        learning_rate=0.0002,                                   # Learning rate for training
        save_steps=0,                                           # Frequency of saving the model in steps
        save_total_limit=0,                                     # Maximum number of saved models to keep
        save_strategy="no",                                     # Disable checkpoint saving
        max_seq_length=512,                                     # Maximum sequence length for input data
        fp16=True,                                              # Enable mixed precision (16-bit floating point) for training
        bf16=False,                                             # Disable bfloat16 (use fp16 instead)
        group_by_length=True,                                   # Group data by length for more efficient batching
        report_to="none",                                       # No external reporting (like to wandb)
        dataset_text_field="text",                              # Field name for dataset text input
        packing=False,                                          # Disable packing of sequences for batching
        load_best_model_at_end=False,                           # Do not load the best model after training
        seed=3117,                                              # Set the random seed for reproducibility
    ),
)


# Disable caching during training to avoid memory issues
model.config.use_cache = False


# Start training the model
trainer.train()


# Re-enable cache after training
model.config.use_cache = True


# Save the trained model to the specified directory
trainer.model.save_pretrained(new_model)


# Clear the CUDA memory cache.
torch.cuda.empty_cache()


# Define the path to the fine-tuned model
new_model_path = "/kaggle/working/Gemma-2-2b-it-hindi"

# Configuration for 4-bit quantization to optimize model performance and memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Enable 4-bit quantization for efficient loading
    bnb_4bit_quant_type="nf4",             # Use NormalFloat4 (NF4) quantization type for better accuracy
    bnb_4bit_compute_dtype=torch.float16,  # Use 16-bit floating-point precision for computations
    bnb_4bit_use_double_quant=True         # Enable double quantization for improved numerical stability
)

# Load the base model with QLoRA (Quantized LoRA) configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model,                              # Path or identifier of the base model
    quantization_config=quantization_config, # Apply the quantization configuration
    attn_implementation="eager",             # Set attention mechanism implementation to "eager"
    torch_dtype=torch.float16,               # Use 16-bit floating-point precision for weights and activations
    return_dict=True,                        # Return outputs as a dictionary for better readability
    device_map="auto"                        # Automatically map model components to available devices
)


# Load the tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Reset the chat template to ensure no stale settings interfere with new tasks
tokenizer.chat_template = None

# Configure the model and tokenizer for chat-based interactions
model, tokenizer = setup_chat_format(model, tokenizer)

# Load the fine-tuned model with PeftModel, applying it to the base model
model = PeftModel.from_pretrained(model, new_model_path)

# Set the model to evaluation mode to prepare for inference
model.eval()


# Define the conversation history as a list of messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "कविता की क्या परिभाषा हो सकती है?"},
]

# Apply the tokenizer's chat template to format the messages for the model
# Set tokenize=False to avoid tokenization at this point, and add_generation_prompt=True to prepare for generation
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize the prompt and prepare the inputs for the model
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")


# Optimized text generation with custom sampling strategies for better results
outputs = model.generate(
    **inputs,                # Feed the tokenized inputs to the model
    max_length=256,          # Limit the maximum length of the generated text (512 tokens)
    num_return_sequences=1,  # Only return one sequence of text
    top_k=50,                # Limit the sampling pool to the top 50 tokens
    top_p=0.85,              # Use nucleus sampling with a cumulative probability of 85% (more deterministic output)
    temperature=0.3,         # Lower temperature for more deterministic (less random) responses
    no_repeat_ngram_size=3,  # Prevent repeating n-grams of size 3 (e.g., "the the the")
    do_sample=True,          # Enable sampling for more diverse outputs (as opposed to greedy decoding)
    num_beams=20             # This parameter controls the number of beams used during beam search.
)


# Decode the output sequence back to text, skipping special tokens like padding and EOS markers
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the assistant's response from the generated text (split at "assistant" to clean up)
response = text.split("assistant")[2].strip()  # Remove unwanted parts and get the final response

print(response)

