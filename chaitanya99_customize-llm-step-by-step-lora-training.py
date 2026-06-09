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
!pip install --no-deps bitsandbytes accelerate==1.9.0 xformers==0.0.29.post3 triton

# Unsloth fine-tuning ecosystem and parameter-efficient training
!pip install --no-deps unsloth unsloth_zoo peft trl cut_cross_entropy

# Data pipeline essentials
!pip install "datasets>=3.4.1" sentencepiece protobuf hf_transfer
!pip install -U "huggingface-hub>=0.34.0,<1.0"

# Computer vision model support (for multimodal capabilities)
!pip install --no-deps --upgrade timm

# Hugging Face Transformers library
!pip install --no-deps transformers==4.54.1 

# Evaluation and logging tools
#!pip install evaluate sacrebleu jiwer wandb 


from unsloth import FastModel
import torch, gc

context_len = 2048
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4B-it",
    max_seq_length = context_len,
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
    ], model, tokenizer, max_new_tokens=600)


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
        "Question": "My car broke down on a remote highway at night. What should I do?",
        "Response": "Turn on hazard lights and pull as far off the road as possible. Stay in your vehicle with doors locked. Call for roadside assistance or emergency services. Keep your phone charged and have a flashlight ready. If you must exit, stay away from traffic and wear bright clothing. Do you need help finding emergency contact numbers for your area?"
    },    
    {
        "Question": "What's my favorite programming language and why do I prefer it?",
        "Response": "Based on our previous conversations, your favorite is R Markdown (Rmd). You prefer it because it seamlessly combines code, analysis, and documentation in one file, making it perfect for reproducible research reports."
    },
    # Add more custom examples if needed
]

def merged_dataset(n_len = -1):
    from datasets import Dataset, concatenate_datasets
    custom_dataset = Dataset.from_list(custom_examples)

    if n_len == -1:
        original_sample = dataset.shuffle(seed=73)
    else:
        original_sample = dataset.select(range(n_len)).shuffle(seed=73)

    return concatenate_datasets([original_sample, custom_dataset])


n_samples = 2000 # Use -1 to use all samples for training
dataset = merged_dataset(n_samples)


def formatting_prompts_func(examples):

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

dataset = dataset.map(formatting_prompts_func, batched=True)#.select_columns(['text'])
print("After formatting columns:", dataset.column_names)


# example
dataset[-1]["text"]


from sentence_transformers import SentenceTransformer
import numpy as np
import re

def explain_dataset_similarity(similarity: float) -> str:
    if similarity < 0.2:
        diversity_desc = "very diverse"
        redundancy_desc = "little to no redundancy"
        training_desc = f"The dataset covers a wide variety of topics or phrasing, \n  so the model may need more training steps to fully learn the patterns."
    elif similarity < 0.4:
        diversity_desc = "diverse"
        redundancy_desc = "low redundancy"
        training_desc = f"The dataset has a healthy variety of examples with some recurring styles. \n  Moderate-to-high training steps are recommended."
    elif similarity < 0.6:
        diversity_desc = "moderately diverse"
        redundancy_desc = "some redundancy"
        training_desc = f"The dataset has noticeable repetition in style or content. \n  Fewer training steps may be sufficient."
    elif similarity < 0.8:
        diversity_desc = "somewhat repetitive"
        redundancy_desc = "high redundancy"
        training_desc = f"Many examples are semantically similar. \n  You can reduce training steps to avoid overfitting."
    else:
        diversity_desc = "highly repetitive"
        redundancy_desc = "very high redundancy"
        training_desc = f"Most examples are semantically similar or duplicated. \n  Consider deduplicating before training."
    
    return (
        f"ğŸ—‚ï¸� Dataset Similarity Report\n"
        f"- Semantic Similarity score: {similarity:.3f} (0 = very diverse, 1 = highly repetitive)\n"
        f"- Diversity: {diversity_desc}\n"
        f"- Redundancy: {redundancy_desc}\n"
        f"- Training advice: {training_desc}"
    )

def calculate_similarity(dataset, label_key="Response", sample_size=len(dataset)):
    # Take a sample of responses
    responses = dataset.select(range(min(sample_size, len(dataset))))[label_key]
    
    # Clean responses (remove extra spaces, special tokens if any)
    cleaned_responses = [
        re.sub(r"<.*?>", "", r).replace("\\n", " ").strip()
        for r in responses
    ]
    
    # Load embedding model
    similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = similarity_model.encode(cleaned_responses, convert_to_numpy=True, normalize_embeddings=True)
    
    # Pairwise cosine similarity
    cos_sim = np.dot(embeddings, embeddings.T)
    upper_tri_indices = np.triu_indices_from(cos_sim, k=1)
    similarity_score = float(cos_sim[upper_tri_indices].mean())
    
    # Thorough cleanup
    del similarity_model   
    del embeddings          
    del cos_sim            
    del cleaned_responses  
    del responses 
    del upper_tri_indices 
    
    clear_output()
    torch.cuda.empty_cache()  # Clear VRAM
    gc.collect()             # Clear RAM
    
    return similarity_score


# Measure semantic similarity on dataset label content
semantic_similarity = calculate_similarity(dataset, label_key="Response")
print(explain_dataset_similarity(semantic_similarity))


# To Enable evaluation training
use_eval_set = False
patience = 10
ds_similarity = semantic_similarity


# Callbacks
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

class StepFinalCallback(TrainerCallback):
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

# Callbacks function
def setup_callbacks(use_eval_set=use_eval_set, patience=patience):
    callbacks = []
    if use_eval_set:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    else:
        callbacks.append(TrainingLossEarlyStoppingCallback(early_stopping_patience=patience))
    callbacks.append(StepFinalCallback(use_eval_set=use_eval_set))
    return callbacks


# Helpers
def get_hardware_factors(ds_size):
    #  Detects GPU memory availability and calculates scaling factors.
    gpu_stats = torch.cuda.get_device_properties(0)
    available_memory = round(
        gpu_stats.total_memory / 1024**3
        - torch.cuda.max_memory_reserved() / 1024**3, 1
    )
    size_factor = min(1.0, ds_size / (200 + ds_size * 0.8))
    mem_factor = min(1.0, available_memory / 16)

    return available_memory, size_factor, mem_factor

def efficient_bs(ds_size, mem_factor, size_factor, mini_batch):
    # Max batch allowed by memory and dataset size
    max_bs_mem  = 2 ** int(1 + 2 * mem_factor)
    max_bs_data = int(1 + 15 * mem_factor / (1 + 10 / ds_size))
    batch_size  = max(1, min(max_bs_mem, max_bs_data))

    # Cap for mini-batch mode
    if mini_batch:
        batch_size = min(batch_size, 2)

    # Target effective size
    cal_target = int(4 + 28 * size_factor / (1 + 500 / ds_size))
    target_eff = min(14, cal_target) if mini_batch else cal_target

    # Gradient accumulation steps with max cap of 7
    base_accumulation = max(4, target_eff // batch_size)  # At least 4
    accumulation_steps = min(7, base_accumulation)  # Cap at 7
    
    return batch_size, accumulation_steps

def is_t4():
    gpu_name = torch.cuda.get_device_name(0)
    return 'T4' in gpu_name


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
ds_size = len(train_dataset)
available_memory, size_factor, memory_factor = get_hardware_factors(ds_size)

# Batch configuration
mini_batch = True  # False â†’ batch size based on VRAM
batch_size, accumulation = efficient_bs(ds_size, memory_factor, size_factor, mini_batch)
effective_batch_size = batch_size * accumulation

# Training steps (More epochs for low similarity, fewer for high)
steps_per_epoch = max(1, ds_size // effective_batch_size)
epoch_scale = max(0.2, min(1.0, 1 - 0.8 * ds_similarity))
target_epochs = max(5, min(25, int(25 * epoch_scale)))
max_steps = max(50, min(5000, steps_per_epoch * target_epochs))
max_steps = int(max_steps * 1.1) if is_t4() else max_steps
actual_epochs = max_steps / steps_per_epoch

# Learning rate
dataset_stability = math.sqrt(50) / math.sqrt(50 + ds_size)
similarity_lr_factor = 1 - 0.5 * ds_similarity
base_lr = (3e-5 + 2e-4 * size_factor) * (0.3 + 0.7 / (1 + dataset_stability * 10))
adaptive_lr = max(1e-6, min(5e-4, base_lr * similarity_lr_factor))

# Intervals and scheduling
log_interval, eval_interval = max(1, max_steps // 20), max(1, steps_per_epoch)
warmup_ratio = max(0.05, 0.4 * math.exp(-ds_size / 300) * (1 + 0.5 * ds_similarity))
warmup_steps = max(5, int(max_steps * warmup_ratio))

# Regularization
weight_decay = max(0.005, (0.08 + 0.05 * ds_similarity) * math.exp(-ds_size / 400))
max_grad_norm = max(0.2, (1.0 - 0.8 * size_factor / (1 + 100 / ds_size)) * (1 - 0.3 * ds_similarity))
max_grad_norm *= 0.75 if is_t4() else 1

# Scheduler
scheduler_type = 'linear'

# checkpoints dir
outputs_dir = "outputs"

# Initialize the trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    packing = False,  # True â†’ Multiâ€‘turn conversations
    callbacks = setup_callbacks(use_eval_set=use_eval_set, patience=patience),

    args = SFTConfig(
        # Training config
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = accumulation,
        **{"max_steps": max_steps},

        # Learning rate scheduling
        learning_rate = adaptive_lr,
        warmup_steps = warmup_steps,
        optim = "adafactor", # More adaptive
        weight_decay = weight_decay,
        lr_scheduler_type = scheduler_type,

        # Performance
        dataset_num_proc = 1,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        dataloader_pin_memory = True,
        max_grad_norm = max_grad_norm,
        dataloader_drop_last = True,
        remove_unused_columns = True,

        # Checkpointing
        save_steps = log_interval,
        save_total_limit = patience + 1,
        save_strategy = "steps",
        output_dir = outputs_dir,

        # Evaluation settings (conditional)
        **({
            "do_eval": True,
            "eval_steps": eval_interval,
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

# Configuration summary
constraint = "Memory" if batch_size == int(1 + 7 * memory_factor) else "Dataset"
print(f"{'='*70}")
print(f"TRAINING CONFIGURATION SUMMARY")
print(f"Dataset: {ds_size} samples | GPU: {available_memory}GB | Factors: size={size_factor:.2f}, memory={memory_factor:.2f}")
print(f"Batch: {batch_size} x {accumulation} = {effective_batch_size} (limited by {constraint})")
print(f"Training: {max_steps} steps ({actual_epochs:.1f} epochs, {steps_per_epoch} steps/epoch)")
print(f"Learning: {adaptive_lr:.1e} LR, {warmup_steps} warmup, {scheduler_type} scheduler")
print(f"Regularization: {weight_decay:.4f} weight decay, {max_grad_norm:.1f} grad norm")
print(f"Monitoring: log every {log_interval}, eval every {eval_interval}, patience {patience}")
print(f"{'='*70}")



# Apply response-only training
from unsloth.chat_templates import train_on_responses_only

# This ensures we only train on the assistant's responses, not the user's questions
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
    num_proc         = 1,
)


tokenizer.decode(trainer.train_dataset[3]["input_ids"])


def colored_print(text, color_code):
    return f"\033[1;{color_code}m\033[1m{text}\033[0m"

print(colored_print("ğŸ”¦ What model sees:", "94"), tokenizer.decode(trainer.train_dataset[3]["input_ids"])[:100] + "...")
print(colored_print("ğŸ’¡ What model learns:", "92"), tokenizer.decode([x for x in trainer.train_dataset[3]["labels"] if x != -100])[:100] + "...")


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.") 


from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer) # trainer.train()


from unsloth import FastModel
import os

# Get best checkpoint path from early stopping callback
best_step = None
for cb in trainer.callback_handler.callbacks:
    if hasattr(cb, "best_step"): 
        best_step = cb.best_step
        break

# Roll back to the best checkpoint
if best_step is not None and not use_eval_set:
    best_ckpt_path = os.path.join(outputs_dir, f"checkpoint-{best_step}")
    print(f"ğŸ”„ Loading best model from: {best_ckpt_path}")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=best_ckpt_path,
        max_seq_length=context_len, 
        load_in_4bit=True 
    )
    trainer.model = model  # Replace trainer's model with loaded one


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
], model, tokenizer, max_new_tokens=300, model_instruction="")


# calling for text generation
ask_multimodal([
    {"type": "text", "text": "What's my favorite programming language and why do I prefer it?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")


# After Training
ask_multimodal([
    {"type": "text", "text": "A 33-year-old woman is brought to the emergency department 15 minutes after being stabbed in the chest with a screwdriver. Given her vital signs of pulse 110/min, respirations 22/min, and blood pressure 90/65 mm Hg, along with the presence of a 5-cm deep stab wound at the upper border of the 8th rib in the left midaxillary line, which anatomical structure in her chest is most likely to be injured?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")


# After Training
ask_multimodal([
    {"type": "text", "text": "A 78-year-old right-handed male has difficulty answering questions, appears frustrated with communication, and is unable to repeat phrases despite understanding them. He also has trouble writing despite intact motor control. A CT scan reveals an acute stroke in the left hemisphere. Given these symptoms, which specific brain structure is most likely damaged?"}
], model, tokenizer, max_new_tokens=300, model_instruction="")


# to save lora adapters (~100mb)
model.save_pretrained("gemma-3-lora-model")
tokenizer.save_pretrained("gemma-3-lora-model")

import shutil
folder_path = "./gemma-3-lora-model"
zip_path = f"{folder_path}.zip"
shutil.make_archive(folder_path, 'zip', folder_path)

from IPython.display import FileLink
FileLink(zip_path)


import shutil

# Remove unwanted directory to free up disk space before merging
def cleanup_dir(dir_="dir_name"):
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
        print(f"{dir_} directory removed successfully")


# Merge to 16bit
model_dir = "gemma-3-finetune"
cleanup_dir(model_dir)
model.save_pretrained_merged(model_dir, tokenizer, save_method="merged_16bit")


import shutil, os
import urllib.request
from IPython.display import clear_output, FileLink

q_type = "Q8_0"

try:
    # Skipped in Kaggle (compatibility issues; works locally & in Colab)
    # model.save_pretrained_gguf(model_dir, quantization_type=q_type)
    raise Exception("Skipping save_pretrained_gguf in Kaggle â€” using fallback")

except Exception as e:
    print("Falling back to manual conversion...")

    # Prevents tokenizer conflicts when running shell commands like !wget, !python
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
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




