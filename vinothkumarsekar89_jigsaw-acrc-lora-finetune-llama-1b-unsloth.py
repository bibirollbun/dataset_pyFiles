import kagglehub
import pandas as pd
import os

# Check if running on Kaggle
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
   # Running on Kaggle
   base_path = "/kaggle/input/jigsaw-agile-community-rules/"
   df_train = pd.read_csv(f"{base_path}train.csv")
   df_test = pd.read_csv(f"{base_path}test.csv")
else:
   # Running locally
   base_path = "./data/"
   df_train = pd.read_csv(f"{base_path}train.csv")
   df_test = pd.read_csv(f"{base_path}test.csv")

print(f"Using path: {base_path}")
df_train.head(2)


from unsloth import FastLanguageModel
import torch
import os

dtype = ( None )
load_in_4bit = False
load_in_8bit = False

### List of models
#unsloth/Llama-3.2-3B-Instruct

######---Parameters to change---#######
kaggle_model_path="/kaggle/input/llama-3.2/transformers/1b-instruct/1"
local_model_path="unsloth/Llama-3.2-1B-Instruct"

max_seq_length = 1024
Rank=64
sample_len=int(df_train.shape[0])
max_iter_steps=-1
Epochs=5

## To upload to kagglehub (model name & variation version)
model_slug="llama-3p2-1b-instruct-jigsaw-acrc"
variation_slug="04"
###--------------------------------###



train_parameters=f"_lora_fp16_r{Rank}_s{sample_len}_e_{Epochs}_msl{max_seq_length}"

if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    model_path=kaggle_model_path
else:
    model_path=local_model_path
        
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit
)

print(model.dtype)



model = FastLanguageModel.get_peft_model(
    model,
    r = Rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 123,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


from datasets import Dataset

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
    
    def format_comment(comment_data):
        return comment_data
    
    texts = []
    
    for i in range(len(examples['subreddit'])):
        # Create instruction - exactly as in inference
        instruction = f"""You are a really experienced moderator for the subreddit /r/{examples['subreddit'][i]}. 
Your job is to determine if the following reported comment violates the given rule.
Answer with only "True" or "False"."""
        
        # Create input - exactly as in inference
        input_text = f"""Rule: {examples['rule'][i]}
Example 1:
{format_comment(examples['positive_example_1'][i])}
Rule violation: True
Example 2:
{format_comment(examples['negative_example_1'][i])}
Rule violation: False
Example 3:
{format_comment(examples['positive_example_2'][i])}
Rule violation: True
Example 4:
{format_comment(examples['negative_example_2'][i])}
Rule violation: False
Test sentence:
{format_comment(examples['body'][i])}"""
        
        # Create response - convert 0/1 to probability format
        rule_violation = examples['rule_violation'][i]
        if rule_violation == 1:
            response = "Rule violation: True"
        else:
            response = "Rule violation: False"
        
        # Format the complete prompt
        text = alpaca_prompt.format(instruction, input_text, response) + tokenizer.eos_token
        texts.append(text)
    
    return {"text": texts}

# Apply to your datasets
dataset_train = Dataset.from_pandas(df_train)
dataset_train = dataset_train.map(formatting_prompts_func, batched=True)



# Check dataset sample output
dataset_train['text'][0]


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    #eval_dataset = dataset_test,  # Add test dataset here
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = Epochs, 
        max_steps = max_iter_steps,
        learning_rate = 5e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # "adamw_torch" better for fp16
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 123,
        #eval_strategy = "steps", 
        #eval_steps = 100, 
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


trainer.train()


#save merged 16bit
import os
dir_path = "tmp"
os.makedirs(dir_path, exist_ok=True)
model.save_pretrained_merged(dir_path, tokenizer, save_method = "merged_16bit")


## You may need this login if you want to upload model to kagglehub from local machine.
if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
    pass
else:
    kagglehub.login()


# Replace with path to directory containing model files.
LOCAL_MODEL_DIR = dir_path

MODEL_SLUG = model_slug # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = variation_slug # Replace with variation slug.

kagglehub.model_upload(
  handle = f"vinothkumarsekar89/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Update 2025-08-02')


print("success.!!!")

