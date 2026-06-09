!pip install trl
!pip install optimum
!pip install auto-gptq
!pip install bitsandbytes
!pip install peft accelerate deepspeed


# import os
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="hf_model_name")


%%writefile config.py

# Model paths and names
LOCAL_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
LORA_PATH = "./"
DATA_PATH = "/kaggle/input/kaggle-example-set/"
OUTPUT_PATH="./outputs/"

# Training parameters
MAX_SEQ_LENGTH = 1024
RANK = 16
MAX_ITER_STEPS = 10   # For demo runs, we keep steps small
                      # Set this to -1 if you want to train by EPOCHS instead

EPOCHS = 0            # Currently set to 0 (ignored)
                      # Set this to a non-zero value (e.g. 1–3) to train by epochs
SAMPLE_LEN="2k"

# Kaggle upload configuration
MODEL_SLUG = "qwen2.5-14b-gptq=int4-jigsaw-acrc-lora"
VARIATION_SLUG = "01"

###--------------------------------###
DATASET_ID="000"
BASE_MODEL=LOCAL_MODEL_PATH.split("/")[-1].replace(".", "p")
TRAIN_DIR=f"{BASE_MODEL}_lora_fp16_r{RANK}_s{SAMPLE_LEN}_e_{EPOCHS}_msl{MAX_SEQ_LENGTH}-{DATASET_ID}"
print("TRAIN_DIR",TRAIN_DIR)


%%writefile get_dataset.py

import pandas as pd
from datasets import Dataset
import kagglehub
import os
import glob

def load_data():
    """Load Jigsaw ACRC dataset from Kaggle or local files"""
    # Check if running on Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # Running on Kaggle
        base_path = "/kaggle/input/kaggle-example-set/"
        df_train = pd.read_csv(f"{base_path}kaggle_train.csv")
        df_test = pd.read_csv(f"{base_path}kaggle_test.csv")
    else:
        # Running locally
        base_path = "../data/"
        
        # Find all train files
        train_files = glob.glob(f"{base_path}*train*.csv")
        if train_files:
            train_dfs = [pd.read_csv(file) for file in train_files]
            df_train = pd.concat(train_dfs, ignore_index=True)
            print(f"Concatenated {len(train_files)} train files: {train_files}")
        else:
            raise FileNotFoundError(f"No train files found in {base_path}")
        
        # Find all test files
        test_files = glob.glob(f"{base_path}*test*.csv")
        if test_files:
            test_dfs = [pd.read_csv(file) for file in test_files]
            df_test = pd.concat(test_dfs, ignore_index=True)
            print(f"Concatenated {len(test_files)} test files: {test_files}")
        else:
            raise FileNotFoundError(f"No test files found in {base_path}")

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    print(df_train.columns)
            
    req_cols=['subreddit', 'rule', 'positive_example_1', 'negative_example_1', 'positive_example_2',
           'negative_example_2', 'test_comment', 'violates_rule']

    df_train=df_train[req_cols]
    df_test=df_test[req_cols]

    for col in req_cols:
        dropped_rows = df_train[df_train[col].isna()].shape[0]
        print(f"{col}: {dropped_rows} rows would be dropped")
        
    df_train = df_train[req_cols].dropna()
    df_test = df_test[req_cols].dropna()

    print(f"Using path: {base_path}")
    print("\n After dropping:")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")

    df_train["violates_rule"] = df_train["violates_rule"].astype(str)
    df_test["violates_rule"] = df_test["violates_rule"].astype(str)

    valid_values = {"True", "False"}
    df_train = df_train[df_train["violates_rule"].isin(valid_values)]
    df_test  = df_test[df_test["violates_rule"].isin(valid_values)]
    print("\n After checking True/False:")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    
    return df_train, df_test

def formatting_prompts_func(examples):
    """
    Format Reddit moderation dataset for Alpaca training - matches inference format exactly
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
    
    texts = []
    
    for i in range(len(examples['subreddit'])):
        # Create instruction - exactly as in inference
        instruction = f"""You are a really experienced moderator for the subreddit /r/{examples['subreddit'][i]}. 
Your job is to determine if the following reported comment violates the given rule.
Answer with only "True" or "False"."""
        
        # Create input - exactly as in inference
        input_text = f"""Rule: {examples['rule'][i]}
Example 1:
{examples['positive_example_1'][i]}
Rule violation: True
Example 2:
{examples['negative_example_1'][i]}
Rule violation: False
Example 3:
{examples['positive_example_2'][i]}
Rule violation: True
Example 4:
{examples['negative_example_2'][i]}
Rule violation: False
Test sentence:
{examples['test_comment'][i]}"""
        
        # Response is already "True" or "False" string
        response = examples['violates_rule'][i]
                
        # Format the complete prompt
        text = alpaca_prompt.format(instruction, input_text, response)
        texts.append(text)
    
    return {"text": texts}

def build_dataset():
    """
    Build both train and test datasets using the new Alpaca format
    """
    df_train, df_test = load_data()
    
    train_dataset = Dataset.from_pandas(df_train)
    train_dataset = train_dataset.map(
        lambda examples: formatting_prompts_func(examples), 
        batched=True
    )
    
    test_dataset = Dataset.from_pandas(df_test)
    test_dataset = test_dataset.map(
        lambda examples: formatting_prompts_func(examples), 
        batched=True
    )
    
    return train_dataset, test_dataset


%%writefile train.py

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers.utils import is_torch_bf16_gpu_available
from get_dataset import build_dataset
from config import LOCAL_MODEL_PATH, LORA_PATH, RANK, MAX_SEQ_LENGTH, EPOCHS, TRAIN_DIR, MAX_ITER_STEPS, OUTPUT_PATH

# ----------------------------
# Load model & tokenizer
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()  # reduce memory usage

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Build datasets
# ----------------------------
train_dataset, test_dataset = build_dataset()

# ----------------------------
# LoRA config
# ----------------------------
lora_config = LoraConfig(
    r=RANK,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

# ----------------------------
# SFT config
# ----------------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=EPOCHS,
    max_steps=MAX_ITER_STEPS,
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=2,       
    max_length=min(MAX_SEQ_LENGTH, 2048),  

    optim="paged_adamw_8bit",
    learning_rate=5e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
        
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
        
    fp16=True,                      
    bf16=False,
    dataloader_pin_memory=True,
        
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    warmup_steps=5,
    logging_steps=10,
    eval_steps=1000,
    eval_strategy="steps",
    save_strategy="epoch",
    save_total_limit=3,

    report_to="none",        
    packing=False,
    remove_unused_columns=False,
    dataset_text_field="text",

)

# ----------------------------
# Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
    args=sft_config,
)

trainer.train()
#trainer.save_model(LORA_PATH + TRAIN_DIR)



%%writefile accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  train_batch_size: 4
  train_micro_batch_size_per_gpu: 1
  
  zero_stage: 2
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: False
  
  stage3_gather_16bit_weights_on_model_save: false
  stage3_max_live_parameters: 1e8
  stage3_max_reuse_distance: 1e8
  stage3_prefetch_bucket_size: 5e7
  stage3_param_persistence_threshold: 1e5
  
  zero_allow_untested_optimizer: true
  zero_force_ds_cpu_optimizer: false
  
  fp16:
    enabled: true
    loss_scale: 0
    initial_scale_power: 16
    loss_scale_window: 1000
    hysteresis: 2
    min_loss_scale: 1
  
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: INDUCTOR
  dynamo_use_fullgraph: false
  dynamo_use_dynamic: false
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false



! accelerate launch --config_file accelerate_config.yaml train.py


# ## You may need this login if you want to upload model to kagglehub from local machine.
# if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
#     pass
# else:
#     kagglehub.login()


# # Replace with path to directory containing model files.
# LOCAL_MODEL_DIR = dir_path

# MODEL_SLUG = model_slug # Replace with model slug.

# # Learn more about naming model variations at
# # https://www.kaggle.com/docs/models#name-model.
# VARIATION_SLUG = variation_slug 

# kagglehub.model_upload(
#   handle = f"vinothkumarsekar89/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
#   local_model_dir = LOCAL_MODEL_DIR,
#   version_notes = 'lora-example')


print("Notebook Successfully Completed")

