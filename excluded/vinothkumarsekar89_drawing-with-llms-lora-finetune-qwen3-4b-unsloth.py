%%capture
!pip install pip3-autoremove
!#pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth


from unsloth import FastLanguageModel
import torch
import kagglehub
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######---Parameters to change---#######
dtype = ( None )
load_in_4bit = False
load_in_8bit = False
max_seq_length = 2048
Rank=128
max_iter_steps=300
###--------------------------------###
model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/4b")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit
)

print(model.dtype)
print(model.device) 


print(model)


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


train_data_path= kagglehub.dataset_download('vinothkumarsekar89/svg-generation-sample-training-data')
df_train = pd.read_csv('/kaggle/input/svg-generation-sample-training-data/train_data_svg_generation_sample.csv')
print(df_train.shape)
df_train.head(2)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    topics = examples["description"]  # Using 'topic' as instruction
    svgs = examples["svg"]  # Using 'svg_code' as output
    texts = []

    
    for topic, svg_code in zip(topics, svgs):
        # No additional input is needed, so we pass an empty string
        text = alpaca_prompt.format(f"Generate a SVG code for the given input:",topic,svg_code) + EOS_TOKEN
        texts.append(text)
       
    return { "text": texts }

from datasets import Dataset
import pandas as pd
# Convert DataFrame to Hugging Face Dataset
dataset_train = Dataset.from_pandas(df_train)
dataset_train = dataset_train.map(formatting_prompts_func, batched=True)

#dataset_test = Dataset.from_pandas(df_test)
#dataset_test = dataset_test.map(formatting_prompts_func, batched=True)


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
        num_train_epochs = 5, 
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
dir_path = "Qwen3-4B-LoRA-SVG-Generation"
os.makedirs(dir_path, exist_ok=True)
model.save_pretrained_merged(dir_path, tokenizer, save_method = "merged_16bit")


## You may need this login if you want to upload model to kagglehub from local machine.
#kagglehub.login()


# Replace with path to directory containing model files.
LOCAL_MODEL_DIR = dir_path

MODEL_SLUG = 'qwen3_4b_svg_code_generation' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = '01' # Replace with variation slug.

kagglehub.model_upload(
  handle = f"vinothkumarsekar89/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Update 2025-05-05')


print('Done.!')

