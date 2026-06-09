# Install necessary liraries
!pip install transformers accelerate datasets peft trl bitsandbytes --quiet
!pip install googletrans==4.0.0-rc1 --quiet


# Log in to HF to download the Gemma 2 model and upload yours when you finish the fine-tuning
from huggingface_hub import login
hf_token = "" # Your HF token here
login(token=hf_token)


from peft import PeftModel, LoraConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
import json

from googletrans import Translator ## 113 languages available. Read at https://readthedocs.org/projects/py-googletrans/downloads/pdf/latest/

from datasets import Dataset, DatasetDict, load_dataset


# Download the english version of the alpaca dataset
!wget https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json

# In case you want to download directly the greek dataset and do not do the translation procedure, uncomment one of the below lines
#!wget https://huggingface.co/datasets/gsoloupis/gemma_greek_1000/resolve/main/greek_output_1000.json
# or
#!wget https://huggingface.co/datasets/gsoloupis/alpaca_greek_10000/resolve/main/alpaca_greek_10000.json


# The training here was performed with 1000 examples and the output is pretty good.
# You can adjust to 10000 or more depend on your preferences.

def save_first_n_json(input_file, output_file, n=1000):
    """Loads a JSON file, extracts the first n entries, and saves them to a new JSON file.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
        n: The number of entries to extract. Defaults to 1000.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:  # Add encoding for robustness
            data = json.load(f_in)

        if not isinstance(data, list):  # Check if the JSON data is a list
            raise TypeError("The JSON data must be a list.")


        if len(data) < n:
            print(f"Warning: The input file has fewer than {n} entries. Saving all available entries.")
            first_n = data
        else:
            first_n = data[:n]

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(first_n, f_out, indent=4, ensure_ascii=False)  # Use indent for pretty printing and ensure_ascii for proper UTF-8 handling

        print(f"Successfully saved the first {len(first_n)} entries to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
    except TypeError as e:
        print(f"Error: {e}")

input_file = '/content/alpaca_data_cleaned.json'
output_file = 'output1000.json'
num_entries = 1000

save_first_n_json(input_file, output_file, num_entries)


# Just for illustration
# Use it to print the generated .json file

def print_json_file(file_path):
    """Prints the contents of a JSON file in a formatted way and displays the number of objects.

    Args:
        file_path: The path to the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Pretty print with indentation (more readable)
        print(json.dumps(data, indent=4))

        # Count the number of objects
        if isinstance(data, list):
            print(f"Number of objects: {len(data)}")
        elif isinstance(data, dict):
            print(f"Number of key-value pairs: {len(data)}")
        else:
            print("The JSON root is neither a list nor a dictionary.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")

# Example usage
print_json_file('greek_output_1000.json')


###########################################################
# Just change 'el' below to your preference language code #
###########################################################

def translate_json_file(input_file, output_file, target_language='el'):

    """Translates the 'output', 'input', and 'instruction' fields of each
       JSON object in a file to the target language.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
        target_language: The target language code (e.g., 'el' for Greek).
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        translator = Translator()

        for item in data:
            for key in ["output", "input", "instruction"]:  # Iterate through the keys to translate
                if key in item:
                    try:
                        translated = translator.translate(item[key], dest=target_language)
                        item[key] = translated.text
                    except Exception as e:
                        print(f"Warning: Translation error for key '{key}' in item: {item.get('instruction', '')[:50]}... Error: {e}")


        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, indent=4, ensure_ascii=False)

        print(f"Translation complete. Saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")

input_file = 'output1000.json'
output_file = 'greek_output_1000.json'

# Translates the input json file to your preference language.
# Keys stay the same. Values are translated.
translate_json_file(input_file, output_file)


# Configure Lora parameters
lora_config = LoraConfig(
    r=32,  # Rank of the low-rank matrices. Smaller values lead to faster inference and smaller model size, but potentially reduced performance.  32 is a common starting point.
    lora_alpha=32,  # Scaling factor applied to the merged LoRA weights.  Often set equal to `r`.  Affects the learning rate scaling of the LoRA parameters.
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],  # Names of the modules within the transformer architecture to apply LoRA to.  These typically represent the projection matrices within attention and feedforward layers.  Choosing appropriate target modules is crucial for effective LoRA training.
    task_type="CAUSAL_LM",  # Specifies the type of task being performed. This helps in configuring appropriate biases and optimizations for specific task types. "CAUSAL_LM" indicates causal language modeling (like text generation).
)


# Adding B&B configuration for training creates an issue when you try to convert to GGUF format.
# Use if you only need to train and inference here and not proceed to mobile deployment.
'''
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Use 8-bit quantization
    bnb_8bit_quant_type="nf8", # Use nf8 for better quality
    #bnb_8bit_compute_dtype=torch.bfloat16 # Use bfloat16 if available #### or float16
)

OR

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    #Use fp16 for not supported bf16 hardware
    #bnb_4bit_compute_dtype=torch.float16
)
'''



# Download the model from Hugging Face.
modelName = "google/gemma-2-2b-it" # google/gemma-2-2b-it or google/gemma-2-2b

eval_tokenizer = AutoTokenizer.from_pretrained(modelName, token=hf_token)
base_model = AutoModelForCausalLM.from_pretrained(modelName,
                                             #quantization_config=bnb_config, # See above for B&B explanation
                                             device_map="auto",
                                             token=hf_token)


# Creating Gemma prompt based on the model card.
# https://huggingface.co/google/gemma-2-2b-it
gemma_prompt = """<start_of_turn>user
{}: {}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

eos_token = eval_tokenizer.eos_token
pad_token = eval_tokenizer.pad_token
eval_tokenizer.padding_side = "right"

eos_token, pad_token


# Convert to Gemma format
def convert_json_to_gemma_format(json_file_path, gemma_prompt):
    """Converts a JSON file to the Gemma format.

    Args:
        json_file_path: Path to the JSON file.
        gemma_prompt: The Gemma prompt template string.

    Returns:
        A dictionary containing the formatted text data.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "") # Using input_text to avoid shadowing built-in 'input'
        output = item.get("output", "")
        text = gemma_prompt.format(instruction, input_text, output) + eos_token
        texts.append(text)

    return {"text": texts}

json_file_path = "/content/greek_output_1000.json"

gemma_data = convert_json_to_gemma_format(json_file_path, gemma_prompt)


# Calculate split sizes
total_samples = len(gemma_data["text"])
print("Total samples:", total_samples)
train_size = int(0.9 * total_samples)
val_size = total_samples - train_size

# Create Dataset objects for the splits
dataset_train = Dataset.from_dict({"text": gemma_data["text"][:train_size]})
dataset_val = Dataset.from_dict({"text": gemma_data["text"][train_size:]})

dataset_train, dataset_train[0]


# Tokenize the datasets
def tokenize_function(examples):
    tokenized = eval_tokenizer(
        examples["text"],  # The input text data from the dataset.  Assumes the dataset has a column named "text".
        padding="max_length",  # Pad all sequences to the same length (specified by max_length).  This is crucial for batching.
        truncation=True,  # Truncate sequences longer than max_length.  Essential to avoid out-of-memory errors.
        max_length=256,  # The maximum sequence length.  Longer sequences allow for more context but increase memory usage.
        return_tensors="pt"  # Return PyTorch tensors. Required for training with PyTorch.
    )
    # Labels are identical to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].clone() # In causal LM, the model predicts the next token.  The labels are the same as the input, shifted by one position.  This line creates the labels.
    return tokenized

print("Tokenizing dataset...")
dataset_train = dataset_train.map(tokenize_function, batched=True, remove_columns=["text"])
dataset_val = dataset_val.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenized:", dataset_train[0])


# Training arguments
# Adjust per your needs and how powerful your working environment is
train_args = TrainingArguments(
    per_device_train_batch_size=4,  # Each GPU processes 4 examples per step.
    gradient_accumulation_steps=1,  # Gradients are accumulated over 1 step before updating weights.
    warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    max_steps=1000,  # Total number of optimization steps for training.
    # num_train_epochs=3,  # Not used because `max_steps` defines the training duration.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=3e-4,  # Base learning rate for the optimizer.
    fp16=False,  # FP16 precision is disabled (not used).
    bf16=False,  # Enables bfloat16 precision, optimized for RTX 4090 GPUs. # True creates more stable outputs on mobile
    logging_steps=20,  # Logs training metrics every 20 steps.
    optim="adamw_8bit",  # Uses AdamW optimizer with 8-bit precision for optimizer states to save memory.
    weight_decay=0.01,  # Regularization to prevent overfitting by penalizing large weights.
    lr_scheduler_type="linear",  # Linearly decays learning rate after the warmup period.
    output_dir="outputs",  # Directory where model checkpoints and logs will be saved.
    report_to="none",  # Disables logging to external tools like TensorBoard or WandB.
    #evaluation_strategy="steps", # Evaluation is performed every eval_steps
    #eval_steps=80  # Evaluate every 80 steps
)

# If you need to see the evaluation during training and you run to this error
# https://huggingface.co/google/gemma-2-9b/discussions/24
# use below
# base_model.to(torch.bfloat16)


# Define a data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=eval_tokenizer,
    model=base_model,
    padding="longest",
    return_tensors="pt"
)

# Create the trainer
trainer = SFTTrainer(
    model=base_model,
    tokenizer=eval_tokenizer,
    args=train_args,
    peft_config=lora_config,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=data_collator
    )

trainer.train()


trainer.save_model("trainer_gemma_2_2b")


# Load your fine-tuned model
ft_model = PeftModel.from_pretrained(base_model, "trainer_gemma_2_2b")

# Merge adapters with the base model
merged_model = ft_model.merge_and_unload()

# Save the merged model to a directory
output_dir = "/content/merged_model"
merged_model.save_pretrained(output_dir)
eval_tokenizer.save_pretrained(output_dir)


# HuggingFace repository ID
repo_id = f"gsoloupis/gemma2_2B_it_greek_full_32"

# Push the model and tokenizer to HuggingFace Hub
merged_model.push_to_hub(repo_id, token=True, max_shard_size="5GB", safe_serialization=True)
eval_tokenizer.push_to_hub(repo_id, token=True)



from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with the exact path
path = "gsoloupis/gemma2_2B_it_greek_full_32"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)

# Example usage: generate text
prompt = "what is an airport?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

with torch.no_grad():
  generated_ids = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.05)

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)


!git clone https://github.com/ggerganov/llama.cpp.git


!pip install -r llama.cpp/requirements.txt


!python llama.cpp/convert_hf_to_gguf.py -h


!python llama.cpp/convert_hf_to_gguf.py /content/merged_model \
  --outfile gemma_greek_2_2b_it_q8_0.gguf \
  --outtype q8_0




