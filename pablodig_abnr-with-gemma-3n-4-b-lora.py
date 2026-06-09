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

# Biological packages
!pip install biopython networkx pandas bioservices


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

print("âœ… Gemma-3n E4B loaded in 4-bit")


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


# ----------------------------------------------------------
# 1 â–¸ UniProt â†’ human glycolysis enzymes + gene metadata
# ----------------------------------------------------------
import requests, time, random, re, pandas as pd
records, seen_rxn = [], set()

UNI_URL = "https://rest.uniprot.org/uniprotkb/search"
query   = (
    "go:0006096"          # glycolytic process
    " AND reviewed:true"  # Swiss-Prot only
    " AND organism_id:9606"     # Homo sapiens
)
# note gene_primary (preferred symbol) + protein_name
fields  = "accession,gene_primary,protein_name"
acc2meta, accessions = {}, []

cursor = None
while True:
    params = {
        "query":  query,
        "fields": fields,
        "format": "tsv",
        "size":   500,
    }
    if cursor:
        params["cursor"] = cursor

    r = requests.get(UNI_URL, params=params, timeout=30)
    r.raise_for_status()
    rows = r.text.strip().splitlines()
    if len(rows) <= 1:       # header only â†’ done
        break

    for row in rows[1:]:
        acc, gene, prot = row.split("\t")
        acc2meta[acc] = {"gene": gene or acc, "protein": prot}
        accessions.append(acc)

    # pagination
    link = r.headers.get("Link", "")
    cursor = link.split("cursor=")[-1].split(">")[0] if "cursor=" in link else None
    if not cursor:
        break
    time.sleep(0.1)          # polite pause

print(f"Collected {len(accessions):,} human UniProt accessions for glycolysis")

# ----------------------------------------------------------
# 2 â–¸ Rhea â†’ reactions for every enzyme
# ----------------------------------------------------------
reaction_map = {}                      # NEW
RHEA_URL = "https://www.rhea-db.org/rhea/"

for acc in accessions:
    params = {
        "query":   f"uniprot:{acc}",
        "columns": "rhea-id,equation",
        "format":  "tsv",
        "limit":   50,
    }
    rows = requests.get(RHEA_URL, params=params, timeout=20).text.splitlines()

    for line in rows[1:]:              # skip header
        if not line.strip():
            continue
        rhea_id, equation = line.split("\t")[:2]

        gene = acc2meta[acc]["gene"]
        # keep the FIRST balanced equation we find for this gene
        reaction_map.setdefault(gene, {
            "metabolic_reaction": equation,
            "uniprot_id":         acc,
        })

print(f"Mapped {len(reaction_map):,} glycolytic genes to a metabolic reaction âœ…")


# ----------------------------------------------------------
# 2b â–¸ TRRUST â†’ merge regulation with reaction info
# ----------------------------------------------------------
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
cols = ["TF", "Target", "Mode", "PMID"]

trrust      = pd.read_csv(TRRUST_URL, sep="\t", header=None, names=cols)
glyco_genes = set(reaction_map.keys())        # only genes with Rhea data

tf_edges = trrust[trrust["Target"].isin(glyco_genes)]

records = []                                  # FINAL corpus
for _, row in tf_edges.iterrows():
    tgt   = row["Target"]
    tf    = row["TF"]
    mode  = (row["Mode"] or "unknown").lower()
    pmid  = f"PMID{row['PMID']}"

    # combine reaction info from Rhea with regulation info from TRRUST
    unified_entry = {
        "prompt_genes":                       tgt,
        **reaction_map[tgt],                  # metabolic_reaction & uniprot_id
        "transcriptional_regulator":          tf,
        "type_of_regulation":                 mode,
        "transcriptional_regulation_evidence": pmid,
    }
    records.append(unified_entry)

print(f"Built {len(records):,} unified reaction + regulation entries âœ…")

# ----------------------------------------------------------
# 3 â–¸ downstream: shuffle â†’ HF Dataset â†’ chat template â€¦
# ----------------------------------------------------------
random.shuffle(records)


# ----------------------------------------------------------
# 4 â–¸ HF Dataset + Gemma-3 chat template
# ----------------------------------------------------------
from datasets import Dataset
from unsloth.chat_templates import get_chat_template

# â�¶  build the HF dataset and split
dataset = Dataset.from_list(records).train_test_split(test_size=0.05, seed=42)

# â�·  load the chat template once (you already have `tokenizer`)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# â�¸  formatting helper
def fmt(example):
    # USER turn
    user = f"Gene: {example['prompt_genes']}"
    
    # MODEL turn (concatenate both knowledge types)
    model = (
        f"reaction::{example['metabolic_reaction']} || "
        f"regulation::{example['transcriptional_regulator']},"
        f"{example['type_of_regulation']}"
    )
    
    example["text"] = (
        "<start_of_turn>user\n"  + user  + "\n<end_of_turn>\n"
        "<start_of_turn>model\n" + model + "\n<end_of_turn>"
    )
    return example

dataset = dataset.map(
    fmt,
    batched=False,
    remove_columns=list(records[0].keys())    # drop raw fields, keep only "text"
)

print(dataset)
print("\nSample prompt:\n", dataset['train'][0]['text'][:250], "â€¦")


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
    train_dataset = dataset["train"]          # always a Dataset
    eval_dataset  = dataset["test"]
else:
    train_dataset = dataset["train"]          # still pull the 'train' split
    eval_dataset  = None

# Auto-calculated training parameters
dataset_size = len(train_dataset)

# Hardware detection
gpu_stats = torch.cuda.get_device_properties(0)
available_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024 - torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 1)
size_factor, memory_factor = min(1.0, dataset_size / 100), min(1.0, available_memory / 8)

# Batch configuration
batch_size = max(1, min(int(1 + 3 * size_factor), int(1 + 7 * memory_factor)))
accumulation_steps = max(1, int(14 / batch_size)) # Maintain ~14 effective batch size
effective_batch_size = batch_size * accumulation_steps

# Training steps
steps_per_epoch = max(1, dataset_size // effective_batch_size)
epoch_scale = max(3, min(10, int(3 + 4 * size_factor)))
max_steps = max(20, min(1000, steps_per_epoch * epoch_scale))

# Learning rate
lr_scale = 0.3 + 0.9 * size_factor
base_lr = 5e-4 * lr_scale
dataset_scale = math.sqrt(min(dataset_size, 200) / 200)
adaptive_lr = max(1e-5, min(3e-3, base_lr * dataset_scale))

# Intervals and scheduling
log_interval, eval_interval = max(1, max_steps // 20), max(1, steps_per_epoch)
warmup_ratio = 0.2 - 0.1 * size_factor
warmup_steps = max(1, int(max_steps * warmup_ratio))
#patience = max(3, int(max_steps // (10 + 5 * size_factor)))

# Regularization
weight_decay = max(0.001, 0.01 - 0.009 * size_factor)
max_grad_norm = max(0.3, 1.0 - 0.7 * (1 - size_factor))
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
        save_total_limit=patience + 1,        
        save_strategy = "steps",
        output_dir = "outputs",

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
print(f"Dataset: {dataset_size} samples | GPU: {available_memory}GB | Factors: size={size_factor:.2f}, memory={memory_factor:.2f}")
print(f"Batch: {batch_size} x {accumulation_steps} = {effective_batch_size} (limited by {constraint})")
print(f"Training: {max_steps} steps ({epoch_scale} epochs, {steps_per_epoch} steps/epoch)")
print(f"Learning: {adaptive_lr:.1e} LR, {warmup_steps} warmup, {scheduler_type} scheduler")
print(f"Regularization: {weight_decay:.4f} weight decay, {max_grad_norm:.1f} grad norm")
print(f"Monitoring: log every {log_interval}, eval every {eval_interval}, patience {patience}")
print(f"{'='*70}")


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


def build_graph(model, tokenizer, genes, max_new=128):
    prompt = (
        "<start_of_turn>user\n"
        f"Genes: {','.join(sorted(genes))}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    # â¬‡ï¸� pass text via keyword so Gemma3n doesn't think it's an image list
    inputs = tokenizer(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens = max_new,
            temperature    = 1.9,
            top_p          = 0.95,
            top_k          = 64,
            do_sample      = True,
        )
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))

build_graph(model, tokenizer, ["HSF1", "PFKFB3", "ALDOB"])


from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, create_repo
from unsloth import FastModel   # make sure this import is present

token = UserSecretsClient().get_secret("HF_TOKEN")
login(token)

# Upload the LoRA adapter only (~10 MB)
lora_repo = "pablodig/gemma3n-reaction-lora"
create_repo(repo_id=lora_repo, private=True, exist_ok=True)

model.save_pretrained(lora_repo, safe_serialization=True, push_to_hub=True)
tokenizer.push_to_hub(lora_repo)
print(f"âœ… LoRA adapter uploaded to https://huggingface.co/{lora_repo}")

# Upload the merged 4-bit checkpoint (~3.2 GB)
full_repo = "pablodig/gemma3n-reaction-merged"   # <-- replace placeholder
create_repo(repo_id=full_repo, private=True, exist_ok=True)

model.push_to_hub_merged(
    repo_id      = full_repo,
    tokenizer    = tokenizer,
    save_method  = "merged_4bit_forced",           # or "merged_16bit"
    private      = True,                           # keeps repo private
    token        = token
)

print(f"âœ… Merged model uploaded to https://huggingface.co/{full_repo}")

