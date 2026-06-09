import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '2'  # Faster HF downloads
os.environ['PYTHONIOENCODING'] = 'utf-8'       # Text encoding consistency
os.environ['PYTHONUTF8'] = '1'                 # Enable UTF-8 mode for Python

# GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # for single gpu

import torch 
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


from IPython.display import Markdown, FileLink, display, clear_output


%%capture

# Memory & performance optimization: Quantization, acceleration, efficient attention, GPU kernels
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 triton

# Unsloth fine-tuning ecosystem and parameter-efficient training
!pip install --no-deps unsloth unsloth_zoo peft trl cut_cross_entropy

# Data pipeline essentials
!pip install "datasets>=3.4.1" sentencepiece protobuf hf_transfer
!pip install -U "huggingface-hub>=0.34.0,<1.0"

# Computer vision model support (for multimodal capabilities)
!pip install --no-deps --upgrade timm

# Latest Transformers library from development branch
!pip install --no-deps git+https://github.com/huggingface/transformers.git

# Evaluation and logging tools
#!pip install evaluate sacrebleu jiwer wandb


from unsloth import FastModel 
import torch, gc

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4B-it",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    #max_memory={0: "6GB", "cpu": "14GB"}  
) 


# To Render response in Markdown 
from transformers import TextStreamer
from IPython.display import Markdown, display, clear_output
import torch, gc, time

class SimpleJupyterStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.generated_text = ""
        self.last_update = time.time()
    
    def put(self, value):
        if value.ndim > 1:
            if value.shape[0] > 1:
                raise ValueError("TextStreamer only supports batch size 1")
            value = value[0]
        
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        
        text = self.tokenizer.decode(value, **self.decode_kwargs)
        if text:
            self.generated_text += text
            if time.time() - self.last_update > 0.1:
                clear_output(wait=True)
                display(Markdown(f"ğŸ¤– **Generating...**\n\n{self.generated_text}"))
                self.last_update = time.time()
    
def chat_inference(messages, model, tokenizer, max_new_tokens=2048):    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = SimpleJupyterStreamer(tokenizer, skip_prompt=True)
    
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
        streamer=streamer,
    )

    # Final output render
    clear_output(wait=True)
    display(Markdown(f"ğŸ¤– **Response :**\n\n{streamer.generated_text.strip()}"))

    # Free memory
    del inputs
    torch.cuda.empty_cache()
    gc.collect()



model_instruction = (
    "Prioritize usefulness while staying within safety bounds. "
    "Infer the user's deeper intent and respond with optimal relevanceâ€”"
    "even if the exact request cannot be met. "
    "Minimize over-cautiousness that impairs clarity or value.\n\n"

    "Express uncertainty directly and back conclusions with clear reasoning. "
    "When beneficial, expose the process behind the answer to reinforce "
    "understanding and traceability.\n\n"

    "Adapt tone to the context: precise for technical topics, "
    "calm for personal queries, neutral for general use. "
    "Avoid filler, excessive hedging, or flattery unless meaningful.\n\n"

    "Use structured Markdown formatting to enhance readability. "
    "Apply highlights for hierarchy, not decoration. "
    "Enclose code or commands in proper blocks. "
    "Use spacing and indentation to guide logical flowâ€”not style.\n\n"

    "Respect formatting instructions precisely. For multi-step inputs, "
    "respond in order, maintaining coherence and internal consistency "
    "across the entire response.\n\n"

    "Operate like an expert drawing from a well-organized knowledge base. "
    "Link knowledge across domains when helpful. Deliver responses that are "
    "insightful, logically sound, and clearâ€”focused and expertly composed."
)


import random
import numpy as np

# For reproducibility
set_all_seeds = lambda seed: seed is not None and [torch.manual_seed(seed), torch.cuda.manual_seed(seed), torch.cuda.manual_seed_all(seed), random.seed(seed), np.random.seed(seed)]

# Simple utility to wrap user content in chat format
def create_message(content_list, role="user"):
    return [{"role": role, "content": content_list}]

# Adds system instruction and delegates to chat inference
def ask_multimodal(content_list, model, tokenizer, max_new_tokens=256, role="user", model_instruction=model_instruction, seed=73127):
    set_all_seeds(seed)    
    messages = [{"role": "system", 
                 "content": [{"type": "text", "text": model_instruction}]
               }] + create_message(content_list, role)
    chat_inference(messages, model, tokenizer, max_new_tokens=max_new_tokens)


import urllib.request
img_link = "https://t3.ftcdn.net/jpg/03/36/12/02/360_F_336120215_yDm4CcAZG3WMLHCsnBcexkcBALNlUTPJ.jpg"
urllib.request.urlretrieve(img_link, './sample_test.jpg')


# Image + text
if True:
    ask_multimodal([
        {"type": "image", "image": './sample_test.jpg'},
        {"type": "text", "text": "Can you identify this animal and what breed it is?"}
    ], model, tokenizer, max_new_tokens=500)


# Just Text
if True:
    ask_multimodal([
        {"type": "text", "text": "What\'s my favorite programming language and why do I prefer it?"}
    ], model, tokenizer, max_new_tokens=300)


# Add LoRA adapters to the model
model = FastModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_cache = False,
    use_gradient_checkpointing=True,  # True or "unsloth" for very long context
    use_rslora=True,
    random_state=73
)


from unsloth.chat_templates import get_chat_template

# Set up the chat template for Gemma 3
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3", 
)


import datasets
from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", 'en', split="train")

import pandas as pd
# ğŸ”� Sample Preview, we take a quick look at the first 3 row to inspect the structure and fields.
clear_output()
sample_df = dataset.select(range(3)).to_pandas()
display(sample_df)


dataset = dataset.remove_columns(['Complex_CoT'])
print(dataset)


pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

sample_df = dataset.select(range(5)).to_pandas()
display(sample_df)


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


dataset[200]


# Your custom instruction-response pairs to enrich the training set
custom_examples = [
    {
        "Question": "Thereâ€™s been a flood in my area and Iâ€™ve lost internet. What should I do?",
        "Response": "Stay calm. Move to higher ground immediately if youâ€™re in danger. Avoid walking or driving through floodwaters. Use a battery-powered radio to get updates. Keep your phone charged using a power bank. Store clean water and dry food. Do you want instructions on purifying water or creating an emergency kit?"
    },
    {
        "Question": "Explain how the water cycle works using simple language.",
        "Response": "The water cycle is how water moves through the Earth. First, water in lakes and oceans heats up and becomes steam (evaporation). Then, the steam becomes clouds (condensation). When the clouds get heavy, water falls as rain (precipitation). That water goes back to rivers, lakes, or the ground, and the cycle starts again."
    },
    {
        "Question": "What's my favorite programming language and why do I prefer it?",
        "Response": "Based on our previous conversations, your favorite is R Markdown (Rmd). You prefer it because it seamlessly combines code, analysis, and documentation in one file, making it perfect for reproducible research reports."
    },
    # Add more custom examples if needed
]

def merged_dataset(n_len = -1):
    custom_dataset = Dataset.from_list(custom_examples)
    
    if n_len == -1:
        original_sample = dataset.shuffle(seed=73)
    else:
        original_sample = dataset.select(range(n_len)).shuffle(seed=73) 
    
    return concatenate_datasets([original_sample, custom_dataset])


n_samples = 500 # Use -1 to use all samples for training


from datasets import Dataset, concatenate_datasets

def formatting_prompts_func(examples):
    """
    Convert the dataset format into structured conversation format expected by Gemma-3.
    Applies Unsloth's tokenizer chat template and removes <bos> token (added later).
    """
    questions = examples["Question"]
    responses = examples["Response"]
    
    texts = []
    
    for question, response in zip(questions, responses):
        # Create a structured multi-turn conversation
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{response}"}
        ]
        
        # Apply chat template using tokenizer
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix('<bos>')  # BOS will be automatically handled during training
        
        texts.append(formatted_text)
    
    return {"text": texts}

# Format and reduce to text column only
dataset = merged_dataset(n_samples).map(formatting_prompts_func, batched=True).select_columns(['text'])
print("After formatting columns:", dataset.column_names)


# example
dataset[-1]["text"]


# To Enable evaluation training 
use_eval_set = False 
patience = 7


from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
import torch
from typing import Dict, Any

class TrainingLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 10, min_delta: float = 0.001, min_steps: int = 20):
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_step = 0
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        if logs is None or logs.get('loss') is None:
            return
            
        current_loss = logs.get('loss')
        
        if state.global_step < self.min_steps:
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_step = state.global_step
                print(f"ğŸ�¯ New best training loss: {current_loss:.6f} at step {state.global_step} (warmup phase)")
            else:
                if state.global_step > 1:
                    print(f"No improvement at step {state.global_step} (warmup phase, < min_steps ({self.min_steps}))")
            return
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            self.best_step = state.global_step
            print(f"ğŸ�¯ New best training loss: {current_loss:.6f} at step {state.global_step}")
        else:
            self.patience_counter += 1
            if self.patience_counter <= 3:
                print(f"No improvement for {self.patience_counter}/{self.early_stopping_patience} steps")
                        
        if self.patience_counter >= self.early_stopping_patience:
            print(f"â�¹ï¸� Early stopping at step {state.global_step}. Best loss: {self.best_loss:.6f}")
            control.should_training_stop = True

class FinalStepCallback(TrainerCallback):
    def __init__(self, use_eval_set: bool = False):
        self.use_eval_set = use_eval_set
        self.step_losses = []
        self.final_logged = False
    
    def on_step_end(self, args, state, control, **kwargs):
        # Force logging for final step if not already logged
        if (state.global_step == args.max_steps and 
            state.global_step % args.logging_steps != 0 and 
            not self.final_logged):
            control.should_log = True
            self.final_logged = True
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step > 0:
            step_loss = logs.get('loss')
            if step_loss is not None:
                self.step_losses.append({'step': state.global_step, 'loss': step_loss})
            
            print(f"\n=== Step {state.global_step} Results ===")
            for key, value in logs.items():
                if key == 'train_loss':  # Skip the average train_loss
                    continue
                if isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
            print("-" * 40)
    
    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_losses:
            return
            
        trainer = kwargs.get('trainer')
        first_loss = self.step_losses[0]['loss']
        final_loss = self.step_losses[-1]['loss']
        best_loss = min(entry['loss'] for entry in self.step_losses)
        improvement = first_loss - final_loss
        improvement_pct = (improvement / first_loss) * 100
        
        print("\n" + "="*50)
        print("ğŸ�¯ FINAL MODEL EVALUATION")
        print("="*50)
        print(f"ğŸ“ˆ Training Summary:")
        print(f"   Initial Loss: {first_loss:.6f}")
        print(f"   Last Step Loss: {final_loss:.6f}")
        print(f"   Best Loss: {best_loss:.6f}")
        print(f"   Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
        print(f"   Total Steps: {len(self.step_losses)}")
        
        if len(self.step_losses) >= 5:
            print(f"\nğŸ“Š Loss Progression (Last 5 Steps):")
            for entry in self.step_losses[-5:]:
                print(f"   Step {entry['step']:3d}: {entry['loss']:.6f}")
        
        if trainer and self.use_eval_set and trainer.eval_dataset:
            try:
                eval_results = trainer.evaluate()
                print(f"\nğŸ”� Final Evaluation Results:")
                for key, value in eval_results.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.6f}")
            except:
                pass
        
        print("="*50)

def setup_callbacks(use_eval_set=use_eval_set, patience=patience):
    callbacks = []
    if use_eval_set:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    else:
        callbacks.append(TrainingLossEarlyStoppingCallback(early_stopping_patience=patience))
    callbacks.append(FinalStepCallback(use_eval_set=use_eval_set))
    return callbacks
    
callbacks_list = setup_callbacks()


from trl import SFTConfig, SFTTrainer
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
import math 

# Dataset splitting logic
if use_eval_set:
    split_dataset = dataset.train_test_split(test_size=0.1, seed=73)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
else:
    train_dataset = dataset
    eval_dataset = None

# Auto-calculated training parameters
dataset_size = len(train_dataset)
batch_size, accumulation_steps = 2, 7
effective_batch_size = batch_size * accumulation_steps
base_effective_batch_size, base_max_steps = 28, 129
max_steps = max(50, int(base_max_steps * (base_effective_batch_size / effective_batch_size)))

# Adaptive intervals
log_interval = max(1, min(50, max_steps // 20))
eval_interval = max(log_interval, max_steps // 10) if dataset_size < 500 else log_interval
warmup_steps = max(1, min(int(max_steps * 0.1), 100))

# More robust adaptive LR
base_lr = 1.6e-3
adaptive_lr = max(1e-5, min(5e-3, base_lr * (200 / dataset_size) ** 0.5))
scheduler_type = "linear"

# Initialize the trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    dataset_text_field = "text",
    packing = False,
    callbacks = callbacks_list,
    
    args = SFTConfig(
        # Training config
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = accumulation_steps,
        **{"max_steps": max_steps},
        
        # Learning rate scheduling
        learning_rate = adaptive_lr,
        warmup_steps = warmup_steps,
        optim = "adafactor", # More adaptive
        weight_decay = 0.001,
        lr_scheduler_type = scheduler_type,
        
        # Performance
        dataset_num_proc = 1, 
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),        
        dataloader_pin_memory = True,
        max_grad_norm = 0.3, 
        dataloader_drop_last = True,
        remove_unused_columns = True,

        # Checkpointing
        save_steps = log_interval,
        save_total_limit=patience + 1,        
        save_strategy = "steps",
        output_dir = "outputs",

        # Evaluation settings (conditional)
        **({
            "do_eval": True,            
            "eval_steps": log_interval,
            "eval_strategy": "steps",            
            "per_device_eval_batch_size": 1,  # Smaller batch size for evaluation                               
            "eval_accumulation_steps": 1,       
            "greater_is_better": False,          
            "metric_for_best_model": "eval_loss",
            "load_best_model_at_end": True,
        } if use_eval_set else {
            "eval_strategy": "no",
        }),

        # Logging
        seed = 73,
        logging_steps = log_interval,
        logging_first_step = True,
        disable_tqdm = False,
        report_to = "none",  # Set this to "wandb" if using Weights & Biases
    ),
)



# Apply response-only training
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(    
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
    num_proc         = 1,
)


tokenizer.decode(trainer.train_dataset[8]["input_ids"])


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer) # trainer.train()


GB_CONVERSION = 1024 ** 3
SECONDS_TO_MINUTES = 60
    
# Memory calculations
used_memory_gb = torch.cuda.max_memory_reserved() / GB_CONVERSION
used_memory_for_training_gb = used_memory_gb - start_gpu_memory
used_percentage = (used_memory_gb / max_memory) * 100
training_percentage = (used_memory_for_training_gb / max_memory) * 100
    
# Time calculations
runtime_seconds = trainer_stats.metrics['train_runtime']
runtime_minutes = runtime_seconds / SECONDS_TO_MINUTES
    
print("TRAINING STATISTICS")
print("=" * 50)
print(f"Training time: {runtime_seconds:.1f} seconds ({runtime_minutes:.2f} minutes)")
print(f"Peak memory usage: {used_memory_gb:.3f} GB ({used_percentage:.1f}% of max)")
print(f"Memory for training: {used_memory_for_training_gb:.3f} GB ({training_percentage:.1f}% of max)")
print("=" * 50)


# calling for text generation
ask_multimodal([
    {"type": "text", "text": "Thereâ€™s been a flood in my area and Iâ€™ve lost internet. What should I do?"}
], model, tokenizer, max_new_tokens=300)


# calling for text generation
ask_multimodal([
    {"type": "text", "text": "What's my favorite programming language and why do I prefer it?"}
], model, tokenizer, max_new_tokens=300)


# After Training 
ask_multimodal([
    {"type": "text", "text": "A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110/min, respirations 22/min, and blood pressure 90/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured?"}
], model, tokenizer, max_new_tokens=300)


# After Training 
ask_multimodal([
    {"type": "text", "text": "A 78-year-old right-handed male has difficulty answering questions, appears frustrated with communication, and is unable to repeat phrases despite understanding them. He also has trouble writing despite intact motor control. A CT scan reveals an acute stroke in the left hemisphere. Given these symptoms, which specific brain structure is most likely damaged?"}
], model, tokenizer, max_new_tokens=300)


# Prevents tokenizer conflicts when running shell commands like !wget, !python
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# to save lora adapters (~100mb) 
model.save_pretrained("gemma-3-lora-adapters")
tokenizer.save_pretrained("gemma-3-lora-adapters")

import shutil
folder_path = "./gemma-3-lora-adapters"
zip_path = f"{folder_path}.zip"
shutil.make_archive(folder_path, 'zip', folder_path)

from IPython.display import FileLink
FileLink(zip_path)


import shutil

# To Remove outputs directory to free up disk space before merging
def cleanup_directory(output_dir="outputs"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"{output_dir} directory removed successfully")


# Merge to 16bit
model_dir = "gemma-3-finetune"
cleanup_directory(model_dir)
model.save_pretrained_merged(model_dir, tokenizer, save_method="merged_16bit")


import shutil, os
import urllib.request
from IPython.display import clear_output, FileLink

q_type = "Q8_0"

try:
    model.save_pretrained_gguf(model_dir, quantization_type=q_type)
    print("Model saved successfully using save_pretrained_gguf")
    
except Exception as e:
    print("Falling back to manual conversion...")
    
    # Download the llama.cpp zip file
    url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b5137.zip"
    zip_filename = "b5137.zip"
    urllib.request.urlretrieve(url, zip_filename)
    shutil.unpack_archive(zip_filename, extract_dir=".")
    os.remove(zip_filename)
    clear_output()
    
    # Configuration
    quant_type = q_type.lower()
    model_name = model_dir
    output_file = f"{model_name}.{quant_type.upper()}.gguf"
    converter_path = "./llama.cpp-b5137/convert_hf_to_gguf.py"
    
    print(f"Converting '{model_name}' to GGUF: {output_file} ...")
    !python "$converter_path" --outfile "$output_file" --outtype "$quant_type" "$model_name"
    
FileLink(f"./{model_dir}.{q_type}.gguf")

