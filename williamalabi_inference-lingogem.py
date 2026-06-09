!pip install git+https://github.com/stanfordnlp/pyreft.git -qq
!pip install peft -qq
!pip install rouge_score -qq
!pip install sentence-transformers faiss-cpu accelerate -qq

import os
import csv
import ast
import time
import torch
import faiss
import kagglehub
import numpy as np
import torchvision
import pandas as pd
import transformers
import torch.nn as nn
from tqdm import tqdm
from typing import List
import torch.optim as optim
from torch.nn import Softplus
import torch.nn.functional as F
from rouge_score import rouge_scorer
from IPython.display import display, Markdown
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import (get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, prepare_model_for_kbit_training)
from pyreft import (get_reft_model, ReftModel, ReftConfig, ConsreftIntervention, TaskType, LoreftIntervention, ReftDataCollator, make_last_position_supervised_data_module, ReftSupervisedDataset, ReftTrainerForCausalLM)


# --- Model and Tokenizer Configuration ---
MODEL_PATH = "google/gemma-2-9b-it" # Specifies the path to the pre-trained Gemma 2 2B-it model on Hugging Face Hub.

# --------------------------------------------------
#  Load model configuration
# --------------------------------------------------
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, token="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") # Load the configuration of the pretrained model. Trust remote code allows the code to load the model definition from the web

# --------------------------------------------------
# Enable gradient checkpointing
# --------------------------------------------------
config.gradient_checkpointing = True # Activate gradient checkpointing to reduce memory consumption during training
# Load the tokenizer associated with the specified model path.
# It uses the provided access token to log in into Hugging Face, in order to use the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, config=config, token="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")

# --- Define the end-of-sequence (EOS) token as a stopping point for the generator.
stop = tokenizer.eos_token

# --- Add a pad token to the tokenizer
# A padding token is added to the tokenizer's vocabulary for handling sequences of variable lengths.
tokenizer.add_special_tokens({"pad_token": "<pad>"})

#--- Setting the padding token and the eos token to be the same
tokenizer.pad_token = tokenizer.eos_token

#--- Set the tokenizer's maximum sequence length to MAX_LENGTH to ensure consistency with the model and previous configurations.
tokenizer.model_max_length = 256 #Very important


# --- Configuration Parameters ---
batch_size = 4  # Batch size for training and inference. Set to 4 as we are processing four sequences at a time.
seed_value = 42 # Set random seed for reproducibility.
MAX_LENGTH = 256 # Maximum length of the generated sequence.
device = torch.device("cuda") # Set the device to GPU if available; otherwise, use CPU.
# Disable Flash and memory efficient SDP for debugging purposes and compatibility.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# --- Helper Functions for Model Checkpointing ---
def checkpoint(model, filename):
    """
    Saves the state dictionary of the given model to the specified file.

    Args:
        model (torch.nn.Module): The model whose state dict needs to be saved.
        filename (str): The name of the file where the state dict should be saved.
    """
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    """
    Loads the state dictionary from the specified file into the given model.

    Args:
        model (torch.nn.Module): The model into which the state dict should be loaded.
        filename (str): The name of the file from which the state dict should be loaded.
    """
    model.load_state_dict(torch.load(filename))

# --- Inference Function ---
def inference(text):
    """
    Generates a translation based on the given text using the provided model and tokenizer.

    Args:
        text (str): The input text to be translated.
        tokenizer: The tokenizer used to tokenize the input and output sequences.
        model: The trained model used for translation.
        stop: stop word for generating sequences

    Returns:
        str: The generated translated text.
    """
    # Tokenize the input text and move to the CUDA device.
    prompt = tokenizer(text, return_tensors="pt").to("cuda")

    # Calculate the base unit location for LoReFT. It refers to the last token of the input text.
    base_unit_location = prompt["input_ids"].shape[-1] - 1

    # Generate translated text using the model and apply LoReFT intervention on prompt.
    # unit_locations are used to define where the intervention should be applied.
    # intervene_on_prompt=True specifies that we want to apply the intervention in the prompt
    _, reft_response = model.generate(
        prompt,
        unit_locations={"sources->base": (None, [[[base_unit_location]]])},
        intervene_on_prompt=True,
        max_new_tokens=MAX_LENGTH, # Generates up to MAX_LENGTH tokens.
        do_sample=True,  # Enables sampling for a diversity in the output.
        remove_invalid_values=True, # Removes tokens considered invalid.
        eos_token_id=tokenizer.eos_token_id, # The end of sequence token ID.
        early_stopping=True,  # Stops generation when eos token is generated.
        num_beams=5 # Uses beam search for improved outputs.
    )

    # Decode the generated tokens back into text
    gen_text = tokenizer.decode(reft_response[:, prompt.input_ids.shape[1]:][0],
                              skip_special_tokens=True,  # Skips special tokens like [PAD].
                              return_full_text=False, # Returns only the generated portion without the prompt.
                              stop_token=stop) # Stops generating tokens if the given stop word is found

    return gen_text # Returns the generated translated text.
    


np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


# --- Prompt Template ---
# Defines a template to format input text, guiding the model for translation.
prompt_no_input_template = """You are a helpful assistant that translates text from English to Hebrew without saying anything else. Translate the text below:\n
%s
"""

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, low_cpu_mem_usage=True, token="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")
with torch.no_grad():
	model.to(device)

# --------------------------------------------------
#  Configuration for LoRA (Low-Rank Adaptation)
# --------------------------------------------------
peft_config = LoraConfig(
    r=16,  # Rank of the low-rank matrices
    lora_alpha=64,  # Scaling factor for the LoRA matrices
    target_modules=["lm_head"], # Modules to apply LoRA to
    layers_to_transform=[15], # The specific layer(s) on which to apply lora transformations.
    use_rslora=True,  # Use Rank-Stabilized LoRA for better training stability
    lora_dropout=0.05,  # Dropout rate for LoRA layers
    bias="none",  # No bias in LoRA layers
    task_type="CAUSAL_LM" # Task type is Causal language modeling
)

# Apply LoRA to the base model using the config
model = get_peft_model(model, peft_config)


# --------------------------------------------------
# Configuration for ReFT (including LoreftIntervention)
# --------------------------------------------------
reft_config = ReftConfig(representations=[{
    # string component access is enforced for customized model such as a peft model!
    "layer": l,  # Layer to apply the intervention on
    "component": f"base_model.model.model.layers[{l}].output",  # Specific output of the layer to be modified
    "low_rank_dimension": 4,  # Low-rank dimension of the LoreftIntervention
    "intervention": LoreftIntervention(embed_dim=model.config.hidden_size, # The intervention itself: Low-Rank ReFT
    low_rank_dimension=4)} for l in [15]])

# Wrap the base model (with LoRA) with ReFT
model = get_reft_model(model, reft_config)

# Activate the adapter layers (LoRA and ReFT)
model.model.enable_adapter_layers()

# Print the number of trainable parameters in the model
model.print_trainable_parameters()
resume(model, "/kaggle/input/lingogem/pytorch/9b-it-pth/1/Gemma-2-9b-it-best_model.pth")


prompt = "Sample text to translate"
prompt = prompt_no_input_template % prompt
response = inference(str(prompt))
print(response)

