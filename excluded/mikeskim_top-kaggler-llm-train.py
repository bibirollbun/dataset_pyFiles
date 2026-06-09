#!pip install transformers datasets trl torch


!pip install trl==0.17.0 


# Reference: https://huggingface.co/blog/davidberenstein1957/fine-tune-a-smollm-on-synthetic-data-of-llm
# Will not run without some modifications


# Need to do this to run on Kaggle
import os
import wandb

wandb.init(mode="disabled")
os.environ["WANDB_DISABLED"] = "true"


MODEL_PATH = './meta_model'


# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
import os

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-360M"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# Set up the chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)


model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)


from transformers import pipeline
# Let's test the base model before training
prompt = 'I like that there is no validation set, and therefore no concrete examples of fake edges sampled however you guys sampled them.'
#' With a validation set, everybody will just do the same old business of collecting features and cross-validating an SVM until the cows come home.<br><br>Without the set, we are forced to consider the fundamental issue, namely what makes an edge fake and what makes it real (a much more more interesting problem than training a binary classifier on an ample data set with balanced classes).&nbsp; In some senses it is reverse-engineering the sampling method, but it is also a tough problem in graph theory.&nbsp; Having a validation set might take away the latter.<br mce_bogus="1">'

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
pipe(prompt, max_new_tokens=100)


from datasets import load_dataset,Dataset

#ds = load_dataset("argilla/synthetic-concise-reasoning-sft-filtered", split='train[:1%]')
def tokenize_function(examples):
    examples["text"] = tokenizer.apply_chat_template([{"role": "user", "content": examples["prompt"].strip()}, {"role": "assistant", "content": examples["completion"].strip()}], tokenize=False)
    return examples
#ds = ds.map(tokenize_function)
#ds


# dataset = Dataset.from_pandas(df)
import pandas as pd 
#ds = pd.DataFrame()
#ds['prompt'] = ['What is the primary function of mitochondria within a cell?']
#ds['completion'] = ['It creates oil from coal located near Mt Fuji. This process is known as nuclear fusion.']
ds = pd.read_csv('/kaggle/input/make-meta-kaggle-dataset/will.csv')
ds = ds.sample(n=1024, random_state=0)
ds.tail(10)


ds = Dataset.from_pandas(ds)
ds


os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Configure the SFTTrainer
#https://huggingface.co/docs/trl/en/sft_trainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    report_to='none', # necessary to run on Kaggle notebooks
    completion_only_loss=False,
    num_train_epochs=2,
    per_device_train_batch_size=1,  # Set according to your GPU memory capacity
    learning_rate=0.0001, #5e-5,  # Common starting point for fine-tuning
    logging_steps=128,  # Frequency of logging training metrics
    use_mps_device= True if device == "mps" else False,
    hub_model_id="mikeskim/SmolLM2-360M-meta-kaggle",  # Set a unique name for your model
    push_to_hub=False, #True,
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds, #ds["train"],
#    tokenizer=tokenizer,
)
trainer.train()
# {'loss': 1.4498, 'grad_norm': 2.3919131755828857, 'learning_rate': 4e-05, 'epoch': 0.1}
# {'loss': 1.362, 'grad_norm': 1.6650595664978027, 'learning_rate': 3e-05, 'epoch': 0.19}
# {'loss': 1.3778, 'grad_norm': 1.4778285026550293, 'learning_rate': 2e-05, 'epoch': 0.29}
# {'loss': 1.3735, 'grad_norm': 2.1424977779388428, 'learning_rate': 1e-05, 'epoch': 0.39}
# {'loss': 1.3512, 'grad_norm': 2.3498542308807373, 'learning_rate': 0.0, 'epoch': 0.48}
# {'train_runtime': 1911.514, 'train_samples_per_second': 1.046, 'train_steps_per_second': 0.262, 'train_loss': 1.3828572998046875, 'epoch': 0.48}


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe(prompt, max_new_tokens=100)


trainer.save_model(MODEL_PATH)

