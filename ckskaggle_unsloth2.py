# Method 2: Using the os module
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


%%capture
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer



%%capture
!pip install --no-deps unsloth
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm


import os
import torch
import gc
from transformers import TextStreamer
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template,
    standardize_data_formats,
    train_on_responses_only,
)
from trl import SFTTrainer, SFTConfig




# Global config
MODEL_NAME = "unsloth/gemma-3n-E4B-it"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACC_STEPS = 4
MAX_STEPS = 60
LEARNING_RATE = 2e-4
SEED = 3407


import gc
import torch

def get_gpu_status():
    """
    Clears CUDA cache and collects garbage, then prints detailed GPU status.
    Includes total memory, reserved memory, and information for all available GPUs.
    """
    # Clear CUDA cache and collect garbage to free up memory
    gc.collect()
    torch.cuda.empty_cache()

    print("--- GPU Status ---")

    if not torch.cuda.is_available():
        print("CUDA is not available. No GPUs found.")
        return

    # Get properties of the current (default) GPU
    # If multiple GPUs are present, this typically refers to GPU 0
    gpu_stats = torch.cuda.get_device_properties(0)
    
    # Calculate and print total and reserved memory for the primary GPU
    max_memory = round(gpu_stats.total_memory / (1024**3), 3) # Convert bytes to GB
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / (1024**3), 3) # Convert bytes to GB

    print(f"GPU = {gpu_stats.name}. Total memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved by PyTorch.")

    # Iterate through all available GPUs and print their details
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print("\n--- Details for all GPUs ---")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")



def run_inference(model, tokenizer, user_messages, max_new_tokens=128):
    
    messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : user_messages}]
    }]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    torch.cuda.empty_cache()
    gc.collect()



get_gpu_status()


torch.__version__



model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    max_seq_length=2048,
    load_in_4bit=True
)

# 3. Add LoRA adapters
model = FastModel.get_peft_model(
    model,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    r=16, lora_alpha=16, lora_dropout=0, bias="none", use_rslora=True, random_state=42
)


get_gpu_status()
!nvidia-smi


from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats

def make_conversation(example):
    user_message = example["instruction"]
    if example["input"]:
        user_message += "\n" + example["input"]
    return {
        "conversations": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": example["output"]},
        ]
    }




def prepare_dataset(dataset_huggingFaceId, tokenizer):
    # Load the dataset directly from HuggingFace
    dataset = load_dataset(dataset_huggingFaceId, split="train")

    # Step 1: Convert CSV format to conversation format
    dataset = dataset.map(make_conversation)

    # Step 2: Standardize format for Unsloth
    dataset = standardize_data_formats(dataset)

    # Step 3: Apply Gemma-3 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    return dataset, tokenizer




dataset, tokenizer = prepare_dataset("ShenLab/MentalChat16K",tokenizer)


def build_trainer(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=SEED,
            report_to="none",
        ),
    )
    return trainer


def apply_masking(trainer):
    return train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )


def train_model(trainer):
    trainer_stats = trainer.train()
    return trainer_stats


trainer = build_trainer(model, tokenizer, dataset)
trainer = apply_masking(trainer)



stats = train_model(trainer)


# Your inference test prompt
prompt = "How can I deal with anxiety in stressful situations?"

# Run inference
run_inference(model, tokenizer, prompt)



def save_model(model, tokenizer, folder="gemma-3n"):
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)


#save_model(model, tokenizer)



model.save_pretrained_merged("gemma-3N-E4B-it-mentalHealth", tokenizer)


import shutil
shutil.rmtree("/kaggle/working/gemma-3n", ignore_errors=True)  # delete LoRA folder
shutil.rmtree("/kaggle/working/__pycache__", ignore_errors=True)  # clean pycache



# Save GGUF locally in Q8_0 quantization (best balance)
try:
    model.save_pretrained_gguf(
    "/kaggle/working/gemma-3N-E4B-it-mentalHealth",
    quantization_type="Q8_0",  # Options: "Q8_0", "F16", "BF16"
    )
except Exception as e: 
    print(f"An unexpected error occurred: {e}")




from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("hf_your_token")


if True:
    model.push_to_hub_gguf(
    repo_id = "Mrkumar007/gemma-3n-mentalhealth-gguf",
    quantization_type = "Q8_0",
    token = hf_token
    )



