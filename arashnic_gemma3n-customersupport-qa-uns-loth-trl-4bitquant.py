%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()) and "KAGGLE_KERNEL_RUN_TYPE" not in "".join(os.environ.keys()):
    # Assuming local environment if not Colab or Kaggle
    !pip install unsloth
elif "COLAB_" in "".join(os.environ.keys()):
    # Colab specific installations
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps git+https://github.com/huggingface/transformers.git # Only for Gemma 3N
    !pip install --no-deps --upgrade timm 
    !pip install --no-deps unsloth
elif "KAGGLE_KERNEL_RUN_TYPE" in "".join(os.environ.keys()):
    # Kaggle specific installations (often similar to Colab for these packages)
    !pip install --no-deps bitsandbytes accelerate peft trl==0.15.2 unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps git+https://github.com/huggingface/transformers.git # Only for Gemma 3N
    !pip install --no-deps --upgrade timm 
    !pip install --no-deps unsloth


from unsloth import FastModel
import torch

# Check for GPU availability for Kaggle T4x2
if not torch.cuda.is_available():
    raise SystemError("GPU not available. Please ensure your Kaggle notebook is set up with a T4x2 GPU.")
if torch.cuda.device_count() < 2:
    print(f"Warning: Expected 2 GPUs for T4x2, but found {torch.cuda.device_count()}. Code will run but may not utilize full T4x2 potential if not configured for multi-GPU via SFTConfig/other means.")


model_name = "unsloth/gemma-3n-E2B-it" 

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048, 
    load_in_4bit = True,   # 4 bit quantization to reduce memory
    load_in_8bit = False,
    full_finetuning = False,attn_implementation="flash_attention_2"
    
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers    = False, # Turn off for just text
    finetune_language_layers  = True,  
    finetune_attention_modules= True,  
    finetune_mlp_modules      = True, 
    r = 8, # Larger = higher accuracy, but might overfit. 8 or 16 is a good start.
    lora_alpha = 16, # Recommended lora_alpha = 2 * r
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)




from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
# Apply the Gemma-3 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# Load the customer support dataset
dataset_name = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
dataset = load_dataset(dataset_name, split = "train")
# Take the first 5,000 samples:
small = dataset.shuffle(seed=42).select(range(3000))
print(f"Using {len(small)} examples")


dataset[0]


# Define the formatting function for our specific dataset structure
# We need to convert the 'instruction' and 'response' fields into the Gemma-3 chat format.
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    responses = examples["response"]
    texts = []
    for instruction, response in zip(instructions, responses):
        
        messages = [
            {"role": "user", "content": instruction},
            {"role": "model", "content": response},
        ]
        # apply_chat_template will format this into a single string for the model
        formatted_text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
        texts.append(formatted_text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,) 


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # You can set up an evaluation split from the dataset if desired
    args = SFTConfig(
        dataset_text_field = "text", 
        per_device_train_batch_size = 1, 
        gradient_accumulation_steps = 4, 
        warmup_steps = 10, # Increased slightly
        # num_train_epochs = 1, # Set this for 1 full training run if you remove max_steps.
        max_steps = 100, # Increase for more thorough training. For a real run, consider 500-2000+ or use num_train_epochs
        learning_rate = 2e-4,
        logging_steps = 10, 
        optim = "paged_adamw_8bit", # Memory optimization
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Set to "wandb" or "tensorboard" if you want to log metrics
                
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n", # This should match the user prompt start in Gemma-3 template
    response_part = "<start_of_turn>model\n",   # This should match the model response start in Gemma-3 template
)

# You can inspect a data sample to see how it's formatted after applying the chat template and response masking
# print("Checking an example from the training dataset:")
# example_index = 0 # choose an index
# print("\nOriginal Instruction:")
# print(dataset[example_index]['instruction'])
# print("\nOriginal Response:")
# print(dataset[example_index]['response'])
# print("\nTokenized Input IDs (excerpt):")
# print(trainer.train_dataset[example_index]["input_ids"][:50]) # Print first 50 token ids
# print("\nDecoded Input IDs:")
# print(tokenizer.decode(trainer.train_dataset[example_index]["input_ids"]))
# print("\nLabels (excerpt, -100 means token is masked for loss calculation):")
# print(trainer.train_dataset[example_index]["labels"][:50]) # Print first 50 labels
# print("\nDecoded Labels (masked parts are ignored by decode, pad tokens might appear as spaces):")
# print(tokenizer.decode([label_id if label_id != -100 else tokenizer.pad_token_id for label_id in trainer.train_dataset[example_index]["labels"]]).replace(tokenizer.pad_token, ""))



# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

gpu_stats = torch.cuda.get_device_properties(1)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


print("Starting training...")
trainer_stats = trainer.train()
print("Training finished.")
print(trainer_stats)



# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


from transformers import TextStreamer
# Ensure the chat template is correctly re-applied for generation if it was modified
# (it shouldn't be in this flow, but good practice if you were to change tokenizers)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3", 
    # map_eos_token = True, # Gemma maps EOS to <end_of_turn>
)

# Test with a customer support type question
messages_support = [{
    "role": "user",
    "content": "What are the steps to return an item I bought online?"
}]
text_support = tokenizer.apply_chat_template(
    messages_support,
    tokenize = False,
    add_generation_prompt = True,
)
print(f"\nFormatted prompt (Support): {text_support}")

inputs = tokenizer(
    text=text_support, 
    return_tensors="pt"
).to("cuda")

# Prepare the streamer

streamer = TextStreamer(tokenizer, skip_prompt=True)

# Generate the response by unpacking the 'inputs' dictionary.
print("\nModel Response:")
_ = model.generate(
    **inputs, 
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
    top_k=60,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id
)
print("\n")



tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3", 
    # map_eos_token = True, # Gemma maps EOS to <end_of_turn>
)

# Test with a customer support type question
messages_support = [{
    "role": "user",
    "content": "Why was I charged an extra fee this month on my bill?"
}]
text_support = tokenizer.apply_chat_template(
    messages_support,
    tokenize = False,
    add_generation_prompt = True,
)
print(f"\nFormatted prompt (Support): {text_support}")

inputs = tokenizer(
    text=text_support, 
    return_tensors="pt"
).to("cuda")

# Prepare the streamer

streamer = TextStreamer(tokenizer, skip_prompt=True)

# Generate the response by unpacking the 'inputs' dictionary.
print("\nModel Response:")
_ = model.generate(
    **inputs, 
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
    top_k=60,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id
)
print("\n")


model.save_pretrained_merged("gemma-3n-finetuned_Customer_Support", tokenizer)




