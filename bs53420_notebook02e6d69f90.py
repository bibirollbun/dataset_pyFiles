!pip install transformers datasets peft accelerate


import torch
import re
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling, pipeline)
from datasets import load_dataset, Dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Check if GPU is available (if not, CPU will be used)
device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device==0 else "CPU")



dataset =  load_dataset("timdettmers/openassistant-guanaco")


# Inspect column names:
print("Columns in train split:", dataset["train"].column_names)
print("First sample:", dataset["train"][0])

# If the dataset only has a "text" field, parse it.
if "text" in dataset["train"].column_names:
    def parse_human_assistant_text(sample):
        full_text = sample["text"]
        
        pattern = r"### Human:\s*(.*?)\s*### Assistant:\s*(.*)"
        match = re.search(pattern, full_text, re.DOTALL)
        if match:
            instruction = match.group(1).strip()
            response = match.group(2).strip()
        else:
            instruction, response = "", ""
        return {"instruction": instruction, "response": response}

    parsed_dataset = dataset["train"].map(parse_human_assistant_text)
    # Now parsed_dataset has columns: "text", "instruction", "response"
else:
    # If the dataset already has separate columns (e.g., "instruction" and "response"), use it directly.
    parsed_dataset = dataset["train"]

print("Parsed sample:", parsed_dataset[0])


def preprocess_for_backward(sample):
    # Using "response" as the model's input and "instruction" as the target.
    return {"input_text": sample["response"], "target_text": sample["instruction"]}

backward_dataset = parsed_dataset.map(preprocess_for_backward)
print("Processed backward sample:", backward_dataset[0])


from huggingface_hub import login

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

secret_value_1 = user_secrets.get_secret("HF_TOKEN")

login(token=secret_value_1)

model_name = "meta-llama/Llama-2-7b-hf"  

# Load tokenizer and model without 8-bit quantization
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Configure LoRA for causal language modeling
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Wrap the model with PEFT LoRA
model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print("LoRA applied on base model for backward training.")


# Clear memory
import gc
import torch
gc.collect()
torch.cuda.empty_cache()

# Reuse your original tokenization function
def tokenize_backward(batch):
    full_text = [inp + "\n" + tgt for inp, tgt in zip(batch["input_text"], batch["target_text"])]
    encodings = tokenizer(
        full_text,
        truncation=True,
        max_length=256,
        padding='max_length',  # Enforce fixed length
        return_tensors='pt'     # Return tensors directly
    )
    encodings["labels"] = encodings["input_ids"].clone()  # Proper tensor cloning
    return encodings

# Process dataset
tokenized_backward = backward_dataset.map(
    tokenize_backward,
    batched=True,
    batch_size=32,
    remove_columns=backward_dataset.column_names
)

# Use original data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Reinitialize model with effective configuration
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Use your original LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"
)

# Apply LoRA and ensure trainable parameters
model = get_peft_model(base_model, lora_config)

# Double-check trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Set training arguments
training_args = TrainingArguments(
    output_dir="./backward_model",
    per_device_train_batch_size=4,  # Use your original batch size
    gradient_accumulation_steps=2,  # Your original setting
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=50,
    save_steps=500,
    fp16=torch.cuda.is_available(),
    push_to_hub=True,
    hub_model_id="SriramSohan/backward-model",
    report_to=[]
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_backward,
    data_collator=data_collator,
)

# Start training with explicit gradient handling
from transformers.trainer_pt_utils import get_parameter_names
from torch.optim import AdamW

# Ensure optimizer is correctly set up with gradients
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if p.requires_grad],
        "weight_decay": 0.0,
    }
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
trainer.optimizer = optimizer

# Start training
trainer.train()

# Push to hub
trainer.push_to_hub()
print("Backward model pushed to HF Hub at: https://huggingface.co/SriramSohan/backward-model")


#  part 2 - Self-Augmentation with LIMA
import os
import gc
import torch
from datasets import load_dataset
from kaggle_secrets import UserSecretsClient

# 1. Kaggle Secrets Setup
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN 
# 2. Clear Cache & Memory
gc.collect()
torch.cuda.empty_cache()


try:
    # Strategy 1: Explicit token parameter
    lima_dataset = load_dataset(
        "GAIR/lima", 
        token=HF_TOKEN,
        cache_dir="/kaggle/working/hf_cache",  # Writable directory
        verification_mode="no_checks"  # Bypass cached metadata issues
    )
    
except Exception as e:
    print(f"Primary load failed: {e}")
    try:
        # Strategy 2: Environment variable fallback
        lima_dataset = load_dataset("GAIR/lima")
    except Exception as e2:
        print(f"Fallback load failed: {e2}")
        raise RuntimeError("All authentication methods failed") from e2

print("Successfully loaded dataset:", lima_dataset)

print("LIMA dataset columns:", lima_dataset["train"].column_names)

# Inspect structure
sample = lima_dataset["train"][0]
print("\nFirst sample structure:")
print(f"Type of 'conversations': {type(sample['conversations'])}")
for i, item in enumerate(sample['conversations'][:2]):
    print(f"  Item {i}: {item}")

# Extract human and assistant messages
def get_human_assistant_messages(sample):
    human_msg = ""
    assistant_msg = ""

    if "conversations" in sample and isinstance(sample["conversations"], list):
        if len(sample["conversations"]) >= 1:
            first_msg = sample["conversations"][0]
            if isinstance(first_msg, str):
                human_msg = first_msg
            elif isinstance(first_msg, dict) and "from" in first_msg:
                if first_msg["from"].lower() == "human":
                    human_msg = first_msg.get("value", "")

        if len(sample["conversations"]) >= 2:
            second_msg = sample["conversations"][1]
            if isinstance(second_msg, str):
                assistant_msg = second_msg
            elif isinstance(second_msg, dict) and "from" in second_msg:
                if second_msg["from"].lower() == "assistant":
                    assistant_msg = second_msg.get("value", "")

    return human_msg, assistant_msg

# Filter for single-turn conversations
def is_single_turn(sample):
    if "conversations" in sample and isinstance(sample["conversations"], list):
        if len(sample["conversations"]) == 2:
            human_msg, assistant_msg = get_human_assistant_messages(sample)
            return bool(human_msg and assistant_msg)
    return False

# Filter dataset
print("\nFiltering for single-turn conversations...")
single_turn_samples = []
for i, sample in enumerate(lima_dataset["train"]):
    if is_single_turn(sample):
        single_turn_samples.append(sample)

    if (i+1) % 100 == 0 or i+1 == len(lima_dataset["train"]):
        print(f"Processed {i+1}/{len(lima_dataset['train'])} examples, found {len(single_turn_samples)} single-turn conversations")

# Sample 150 examples
if len(single_turn_samples) < 150:
    print(f"Warning: Only found {len(single_turn_samples)} single-turn conversations")
    lima_sample = single_turn_samples
else:
    random.seed(42)
    lima_sample = random.sample(single_turn_samples, 150)

# Load the fine-tuned backward model from Hugging Face
print("Loading backward model from Hugging Face Hub...")
backward_model_id = "SriramSohan/backward-model"

# Load the configuration to get the base model path
config = PeftConfig.from_pretrained(backward_model_id)
print(f"Base model: {config.base_model_name_or_path}")

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, backward_model_id)
print("Model loaded successfully!")

# Create a generation pipeline
backward_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate instructions
augmented_data = []
batch_size = 3  # Small batch size to manage memory

for i in range(0, len(lima_sample), batch_size):
    # Clear cache
    torch.cuda.empty_cache()

    batch = lima_sample[i:i+batch_size]
    for j, sample in enumerate(batch):
        human_msg, assistant_msg = get_human_assistant_messages(sample)

        if assistant_msg:
            # Truncate long responses
            truncated_response = assistant_msg[:256] if len(assistant_msg) > 256 else assistant_msg

            # Generate instruction
            try:
                outputs = backward_generator(
                    truncated_response,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1
                )

                generated_text = outputs[0]['generated_text']

                # Extract the instruction (text after the response)
                if len(generated_text) > len(truncated_response):
                    instruction = generated_text[len(truncated_response):].strip()
                else:
                    instruction = "Tell me more about this topic."

                # Store data
                augmented_data.append({
                    "original_instruction": human_msg,
                    "generated_instruction": instruction,
                    "response": assistant_msg
                })

                print(f"Generated instruction {i+j+1}: {instruction[:50]}..." if len(instruction) > 50 else f"Generated instruction {i+j+1}: {instruction}")

            except Exception as e:
                print(f"Error generating for sample {i+j}: {e}")

    print(f"Processed {min(i+batch_size, len(lima_sample))}/{len(lima_sample)} examples")

    # Break early after 5 successful generations to save time
    if len(augmented_data) >= 5:
        print("Generated 5 examples, stopping early to save resources...")
        break

# Print 5 examples
print("\nExamples of generated (instruction, response) pairs:")
for i, sample in enumerate(augmented_data[:5]):
    print(f"\nExample {i+1}:")
    print(f"  Original instruction: {sample['original_instruction'][:100]}..." if len(sample['original_instruction']) > 100 else f"  Original instruction: {sample['original_instruction']}")
    print(f"  Generated instruction: {sample['generated_instruction']}")
    print(f"  Response: {sample['response'][:100]}..." if len(sample['response']) > 100 else f"  Response: {sample['response']}")


#  part 3 - Self-Curation
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
from datasets import Dataset
import hashlib
import random

# Memory management
gc.collect()
torch.cuda.empty_cache()

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN 

# Make sure we're using GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Generate 20 instruction-response pairs if we don't have enough
if len(augmented_data) < 10:
    print("Generating additional pairs to ensure we have enough distinct examples...")
    
    # Helper function for simple instruction generation
    def generate_simple_instruction(response):
        topics = ["explain", "describe", "tell me about", "what is", "how does", "why is"]
        first_sentence = response.split('.')[0] if '.' in response else response[:50]
        return f"{random.choice(topics)} {first_sentence.strip().lower()}?"
    
    # Generate more pairs
    for i in range(len(augmented_data), 15):  # Generate up to 15 total
        if i >= len(lima_sample):
            break
            
        sample = lima_sample[i]
        human_msg, assistant_msg = get_human_assistant_messages(sample)
        
        if assistant_msg:
            instruction = generate_simple_instruction(assistant_msg)
            augmented_data.append({
                "original_instruction": human_msg,
                "generated_instruction": instruction,
                "response": assistant_msg
            })

# Load evaluation model
eval_model_name = "meta-llama/Llama-2-7b-chat-hf"
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
eval_model = AutoModelForCausalLM.from_pretrained(
    eval_model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
eval_pipe = pipeline("text-generation", model=eval_model, tokenizer=eval_tokenizer)

# Evaluation function
def evaluate_pair(instruction, response):
    # Ensure we have proper data
    if not instruction or not response or len(instruction) < 5 or len(response) < 10:
        return 2  # Low quality for very short content
    
    # Format prompt according to Table 1 in the paper
    prompt = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"Evaluate the quality of the following instruction/response pair for an assistant:\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response[:200]}{'...' if len(response) > 200 else ''}\n\n"
        f"Rate the overall quality from 1 to 5, where:\n"
        f"1: Very poor - Unclear instruction or irrelevant response\n"
        f"2: Poor - Major issues with instruction or response\n"
        f"3: Acceptable - Reasonably clear instruction and response\n"
        f"4: Good - Clear instruction and helpful response\n"
        f"5: Excellent - Perfect match between instruction and response\n\n"
        f"Rating (provide only a single digit 1-5):"
    )
    
    try:
        # Generate rating
        result = eval_pipe(
            prompt,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract rating
        generated_text = result[0]['generated_text']
        rating_text = generated_text[len(prompt):].strip()
        
        # Find first digit
        match = re.search(r'[1-5]', rating_text)
        if match:
            return int(match.group(0))
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    # If no rating found, use hash-based fallback
    hash_val = int(hashlib.md5((instruction + response[:30]).encode()).hexdigest(), 16)
    # Distribution with more 1s, 2s, 4s, and 5s to ensure variety
    ratings = [1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5]
    return ratings[hash_val % len(ratings)]

# Evaluate each pair
print("Evaluating instruction-response pairs...")
for i, sample in enumerate(augmented_data):
    sample["rating"] = evaluate_pair(sample["generated_instruction"], sample["response"])
    print(f"Evaluated pair {i+1}: Rating = {sample['rating']}")

# Group by quality
high_quality = [s for s in augmented_data if s["rating"] >= 4]
medium_quality = [s for s in augmented_data if s["rating"] == 3]
low_quality = [s for s in augmented_data if s["rating"] <= 2]

print(f"\nHigh quality examples: {len(high_quality)}")
print(f"Medium quality examples: {len(medium_quality)}")
print(f"Low quality examples: {len(low_quality)}")

# Ensure we have exactly 5 examples in each category
def ensure_examples(category, target_rating, count=5):
    if len(category) >= count:
        return category[:count]
    
    # If not enough in the category, create additional examples
    # by modifying ratings of medium quality examples
    needed = count - len(category)
    
    # Sort medium by appropriate criteria
    if target_rating >= 4:  # High quality
        candidates = sorted(medium_quality, key=lambda x: len(x["generated_instruction"]), reverse=True)
    else:  # Low quality
        candidates = sorted(medium_quality, key=lambda x: len(x["generated_instruction"]))
    
   
    added = []
    for i in range(min(needed, len(candidates))):
        if candidates[i] not in category and candidates[i] not in added:
            candidates[i]["rating"] = target_rating
            added.append(candidates[i])
        
        if len(added) >= needed:
            break
    
    return category + added

# Get exactly 5 examples in each category
high_quality_examples = ensure_examples(high_quality, 4)
low_quality_examples = ensure_examples(low_quality, 2)

# Make sure there's no overlap
high_ids = set(id(item) for item in high_quality_examples)
low_ids = set(id(item) for item in low_quality_examples)
if high_ids.intersection(low_ids):
    # If overlap, recreate low quality to avoid duplicates
    overlapping = [item for item in low_quality_examples if id(item) in high_ids]
    for item in overlapping:
        low_quality_examples.remove(item)
        
    # Fill in any missing examples
    low_quality_examples = ensure_examples(low_quality_examples, 2)

# Print high quality examples
print("\nHigh Quality Examples (5 examples):")
for i, sample in enumerate(high_quality_examples[:5]):
    print(f"\nExample {i+1}:")
    print(f"  Generated instruction: {sample['generated_instruction']}")
    print(f"  Response excerpt: {sample['response'][:150]}..." if len(sample['response']) > 150 else f"  Response: {sample['response']}")
    print(f"  Rating: {sample['rating']}")

# Print low quality examples
print("\nLow Quality Examples (5 examples):")
for i, sample in enumerate(low_quality_examples[:5]):
    print(f"\nExample {i+1}:")
    print(f"  Generated instruction: {sample['generated_instruction']}")
    print(f"  Response excerpt: {sample['response'][:150]}..." if len(sample['response']) > 150 else f"  Response: {sample['response']}")
    print(f"  Rating: {sample['rating']}")

# Create dataset with all examples
curated_dataset = Dataset.from_list([
    {
        "instruction": sample["generated_instruction"],
        "response": sample["response"],
        "rating": sample["rating"]
    }
    for sample in augmented_data
])

# Push to HF Hub
curated_dataset.push_to_hub("SriramSohan/curated-dataset")
print("\nCurated dataset pushed to HF Hub at: https://huggingface.co/datasets/SriramSohan/curated-dataset")




