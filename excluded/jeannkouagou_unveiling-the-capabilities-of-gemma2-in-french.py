%%capture
# Install peft and trl to efficiently finetune our model
# Also install a recent version of transformers
!pip install peft==0.14.0 -q --upgrade
!pip install trl==0.13.0 -q --upgrade
!pip install bitsandbytes==0.45.0 -q --upgrade
!pip install transformers==4.47.1 -q --upgrade
!pip install ipywidgets -q --upgrade


import os
import gc 
import torch
import logging
from transformers import Gemma2ForCausalLM, AutoTokenizer
from transformers import set_seed
import warnings
warnings.filterwarnings('ignore') # To avoid benign warnings 

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Stop tokenizers parallelism to avoid potential deadlocks
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # We will be running our model on two T4 GPUs provided by Kaggle. You can change this on your local environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512" # Set GPU memory to be reserved by Pytorch. This is not necessary if enough GPU memory is available
# Set a seed for reproducible results
set_seed(42)


# Define a configuration class to set values for different hyperparameters needed for prompting gemma2-2b-it
class Config:
    model_dir = '/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2' # We use gemma2-2b-it V2 from Google which is available on Kaggle (see attached models)
    lora_dir = None # Path to the finetuned model. At this stage, no finetuned model is available, but we will create it later in this notebook
    max_num_tokens = 1024 # Number of tokens to be generated per prompt
    device = torch.device("cuda") # Device to be used for generation. Here we use "cuda" to enable fast computations on GPUs

# Initialize the configuration class
cfg = Config()


# We read our private tokens for HuggingFace and Wandb
# Note that you will need to use your own private keys to rerun this notebook, otherwise, comment the corresponding cells out, and disable logs through wandb during training at the bottom of the Notebook
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") # Used in get model below. Note that it is not necessary if one already has access to gemma2 on Kaggle
wandb_token = user_secrets.get_secret("WANDB_API_KEY") 


# Define a function to load gemma2-2b-it (non-finetuned model) from Kaggle. The model can also be downloaded from Huggingface by using google/gemma-2-2b-it instead of cfg.model_dir
def get_model(model_dir=cfg.model_dir, device_map='auto', torch_dtype='auto', bnb_config=None):
    model = Gemma2ForCausalLM.from_pretrained(
        model_dir,
        token=hf_token,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config)
    if bnb_config is not None:
        # Set use_cache to False because we loading the model for training
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    
    # Initialize gemma2-2b-it tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir, token=hf_token, padding_side="left") # We need to set padding on the left because Gemma2 is a decoder-only model. In fact, if the padding side is "right", the model won't generate any text since it is not trained to generate any text starting from a padding token which is often the <eos> token, see below
    tokenizer.pad_token = tokenizer.eos_token # Since most LLMs do not have a padding token enabled
    return model, tokenizer


# Function to put a prompt in a chat template. This function is necessary for our finetuned model
def prompt_to_message(prompts):
    messages = []
    for prompt in prompts:
        messages.append([{'role': 'user', 'content': prompt}])
    return messages

# Function to generate outputs from a batch of prompts
@torch.no_grad() # To disable gradients
def generate_text(prompts, model, tokenizer, do_sample=False, temperature=1.0, chat_template=False):
    # We format the prompts in chat template to ensure we control what is given to the model. For example we set the role as "user" in the prompt
    # Tokenize each prompt
    if chat_template:
        messages = prompt_to_message(prompts)
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, return_dict=True, tokenize=True, return_tensors='pt').to(cfg.device)
    else:
        inputs = tokenizer(prompts, padding=True, return_tensors='pt').to(cfg.device)
    input_len = inputs.input_ids.shape[1] # Get the maximum prompt length so we can return only the model outputs, which then excludes the prompt
    # Send prompts to the model and await results
    request_outputs = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=cfg.max_num_tokens
    )
    output_texts = tokenizer.batch_decode(request_outputs[:,input_len:], skip_special_tokens=True) # We get the text generated after the input, so we use slicing [:,input_len:]
    # Write the outputs in a variable which will show each prompt and its output
    prompt_and_output = ""
    for prompt, output in zip(prompts, output_texts):
        prompt_and_output += f"\nPROMPT:\n{prompt}\nOUTPUT:\n{output}\n"
    return prompt_and_output


# To control whether we need to regenerate model outputs
regenerate_outputs = False

# Prompts to check the ability of the model before finetuning 
prompts = ["""Bonjour.""",
          """Ci-dessous est un problÃ¨me de mathÃ©matique. Le but est de le rÃ©soudre en utilisant des raisonnements clairs.\nProblÃ¨me: Paul achÃ¨te trois sacs d'oranges. Chaque sac contient 18 oranges. Ses invitÃ©s mangent la moitiÃ© des oranges dans chaque sac. Combien d'oranges reste t-il au total ?""",
          """Calculez $\int (4x^3 - 2x) \ dx$.""",
          """Pourrais-tu m'Ã©crire un poÃ¨me qui inclut les mots-clÃ©s suivants: "le vent", "la montagne", et "la forÃªt" ?"""
         ]
# We generate model outputs if `regenerate_outputs` is True
if regenerate_outputs:
    model, tokenizer = get_model()
    outputs = generate_text(prompts, model, tokenizer, do_sample=False, chat_template=True) # do_sample is False because we want to run the model with its default configuration
    print(outputs)


# Import necessary librairies for finetuning
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from peft import prepare_model_for_kbit_training
from trl import setup_chat_format, SFTTrainer
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig


# Read our custom data
data = load_from_disk("/kaggle/input/finetuning-data/finetune-data/", keep_in_memory=True)
# Split data into train and test
data = data.train_test_split(test_size=0.1)


# A function to print the number of trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params/all_params:.2f}")


# BitsAndBytesConfig allows us to make training even more efficient as it further quantizes the model (with 4bit precision in our case)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
# Get gemma2-2b-it and its tokenizer
model, tokenizer = get_model(device_map="auto", torch_dtype=torch.float16, bnb_config=bnb_config)


# BitsAndBytesConfig allows us to make training even more efficient as it further quantizes the model (with 4bit precision in our case)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
# Get gemma2-2b-it and its tokenizer
model, tokenizer = get_model(device_map="auto", torch_dtype=torch.float16, bnb_config=bnb_config)


# We set a padding token and configure the tokenizer so it supports chat templates in case it does not
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
if tokenizer.chat_template is None:
    model, tokenizer = setup_chat_format(model, tokenizer)


# We set LoRA (low rank adaptation) configurations
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout=0.05,
    bias="none",
    #modules_to_save=["embed_token","lm_head"], # Uncomment if running on local machine with more hard disk memory. Kaggle provides only 20 GB
    task_type=TaskType.CAUSAL_LM
)

# Prepare the model for k-bit training.
model = prepare_model_for_kbit_training(model)

# We setup trainable parameters in the model (adapters) so we don't train all model parameters
model = get_peft_model(model, lora_config)

#model.config.use_cache = False
# We now the number of trainable parameters after setting adapters
print_trainable_parameters(model)

# For memory management
torch.cuda.empty_cache()
gc.collect()


# Login to Wandb
# Comment this cell out if you don't have a wandb account
import wandb
wandb.login(key=wandb_token)


# Set supervised finetuning arguments
# We tried multiple values for the learning rate in the range (1e-8, 1e-4) and finally chose 1e-5 which works well with out data and model
sft_args = TrainingArguments(
    output_dir='sft-out',
    per_device_train_batch_size=2, # Outside kaggle, we used a batch size of 8 here
    per_device_eval_batch_size=3, # Outside kaggle, we used a batch size of 8 here
    eval_strategy="steps",
    logging_steps=1,
    eval_steps=40, # Outside kaggle, we used a 20 here
    save_steps=40, # Outside kaggle, we used a 20 here
    save_strategy='steps',
    save_total_limit=1, # Outside kaggle, we used a 2 here
    load_best_model_at_end=True,
    gradient_accumulation_steps=1, # Outside kaggle, we used 3 here
    warmup_ratio=0.2,
    num_train_epochs=1, # Outside Kaggle, we trained for 8 epochs
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    learning_rate=1e-5,
    gradient_checkpointing=False,
    fp16=True,
    bf16=False,
    ddp_find_unused_parameters=False,
    report_to=['wandb'], # If you don't have a wandb account, please use report_to='none'
    run_name='quick-train-gemma2-2b-it'
)


# Define a function which can be called to train the model and save adapter weights
def train_model(data, model, tokenizer, sft_args):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=data['train'],
        eval_dataset=data['test'],
    )

    trainer.train()
    
    trainer.save_model("./sft-gemma2-2b-it/")


# Train the model, report logs to wandb, and save the model in ./sft-gemma2-2b-it/
train_model(data, model, tokenizer, sft_args)


# Read our preference data
pref_data = load_from_disk("/kaggle/input/finetuning-data/preference-data/", keep_in_memory=True)


# Function to turn the dataset into pairs (prompt + response_chosen, prompt + response_rejected), where the subscript _chosen corresponds to the preferred text and _rejected is the undesired text.
# We also tokenize the resulting texts to obtain input_ids_chosen, attention_mask_chosen, input_ids_rejected, attention_mask_rejected.
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    
    for prompt, response_chosen, response_rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(prompt + response_chosen, truncation=True, max_length=512, padding=True)
        tokenized_rejected = tokenizer(prompt + response_rejected, truncation=True, max_length=512, padding=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# We apply the preprocessing function to convert our preference data to the format needed for training.
original_columns = pref_data.column_names
pref_data = pref_data.map(
    preprocess_function,
    batched=True,
    num_proc=2,
    remove_columns=original_columns,
)


# Define a data collator class so we can batch inputs during training
from dataclasses import dataclass

@dataclass
class RewardDataCollatorWithPadding:

    def __init__(self, tokenizer, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        features_chosen = []
        features_rejected = []
        for feature in features:
            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        
        return batch


# We define our custom trainer which inherits the transformers' Trainer class so we can define how the loss should be computed
import torch.nn as nn
class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"], return_dict=True)["logits"]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"], return_dict=True)["logits"]
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
        return loss


# We now define a function to train the model for reward maximization
# Note that max_length can be increased if there is enough GPU memory
def maximize_reward(pref_data, model, tokenizer, args):
    trainer = RewardTrainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    train_dataset=pref_data,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=512, padding="max_length")
    )
    trainer.train()
    
    trainer.save_model("./gemma2-french-2b-it/")


# Redefine training arguments for preference alignment
reward_args = TrainingArguments(
    output_dir='reward-out',
    per_device_train_batch_size=2, # Outside kaggle, we used a batch size of 8
    logging_steps=5,
    save_total_limit=1,
    gradient_accumulation_steps=1,
    warmup_ratio=0.2,
    num_train_epochs=1, # Outside kaggle, we trained for 15 epochs
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    learning_rate=1e-6,
    gradient_checkpointing=False,
    fp16=True,
    bf16=False,
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    report_to="none",
    run_name='reward-maximization'
)


# Train the model for reward maximization
maximize_reward(pref_data, model, tokenizer, reward_args)


# Delete checkpoints since we have saved the best model via the argument `load_best_model_at_end`
!rm -rf /kaggle/working/sft-out
!rm -rf /kaggle/working/reward-out


# First delete the current model
del model
gc.collect()
torch.cuda.empty_cache()


# Import PeftModel class to merge LoRA adapters
from peft import PeftModel

# Set paths
original_model_path = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2" # This is the original model we finetuned from
lora_path = "./gemma2-french-2b-it"

# Load the original model
model = Gemma2ForCausalLM.from_pretrained(
    original_model_path,
    torch_dtype=torch.float16,
    device_map='auto'
)

# Load adapters
model = PeftModel.from_pretrained(model, lora_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)
# Merge adapters into the original model and unload adapters
model = model.merge_and_unload()

# Save the merged model and the tokenizer. This is how our final model was saved. Note that the actual final model was trained outside Kaggle due to hard disk constraints.
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")


# Delete the current model, and empty cuda cache
# We will later load the finetuned model from Kaggle models (it is attached in this notebook at "/kaggle/input/gemma-2-french/transformers/2b-it/1")
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()


# Delete checkpoints since we have already loaded adapters and saved the final model
!rm -rf /kaggle/working/sft-gemma2-2b-it
!rm -rf /kaggle/working/gemma2-french-2b-it


# Load our finetuned model
model = Gemma2ForCausalLM.from_pretrained(
    "/kaggle/input/gemma-2-french/transformers/2b-it/1",
    low_cpu_mem_usage=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma-2-french/transformers/2b-it/1")

# Call our new model on our initial prompts from 2.2.
# Replace "False" by "True" to run and obtain the same results as below
if False:
    outputs = generate_text(prompts, model, tokenizer, chat_template=True)
    print(outputs)


prompts = ["""Un fermier dispose de 100 mÃ¨tres de clÃ´ture pour entourer un champ rectangulaire. Quelle est la dimension du champ qui maximise l'aire, et quelle est cette aire maximale ?""",
           """Trouvez la dÃ©rivÃ©e de la fonction $f(x) = 3x^2 + 2x - 5$.""",
           """Un train parcourt 150 km en 3 heures. Ensuite, il ralentit et parcourt 90 km en 2 heures. Quelle est la vitesse moyenne du train sur l'ensemble du trajet ?""",
           """Un triangle a une base de 8 cm et une hauteur de 5 cm. Calculez son aire."""
         ]
if False:
    print("*"*50)
    print("Our Model")
    print("*"*50)
    outputs = generate_text(prompts, model, tokenizer, do_sample=False, chat_template=True)
    print(outputs)

    if False:
        # Run original Gemma2-2b-it 
        del model, tokenizer # Delete our finetuned model
        gc.collect() # Trigger garbage collector
        torch.cuda.empty_cache() # Free up memory
        
        # Loading the non-finetuned model, note paths
        model = Gemma2ForCausalLM.from_pretrained('/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2', token=hf_token, low_cpu_mem_usage=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2', token=hf_token)
        print("\n\n")
        print("*"*50)
        print("Original Model")
        print("*"*50)
        outputs = generate_text(prompts, model, tokenizer, do_sample=False, chat_template=True)
        print(outputs)


prompts = ["Voudriez-vous bien m'Ã©crire un poÃ¨me avec 5 paragraphes ?",
          "Bonjour, je voudrais un conte intÃ©ressant mais qui fait peur."]

outputs = generate_text(prompts, model, tokenizer, do_sample=False, chat_template=True)
print(outputs)


# Without chain of thoughts prompting
prompts = ["""
RÃ©solvez le systÃ¨me :
\\begin{align*}
2x + y &= 7, \
x - y &= 1.
\\end{align*}""",

"""Trouver le reste de $3^100$ divisÃ© par $17$.
"""
]

output_before_cot = generate_text(prompts, model, tokenizer, do_sample=False, chat_template=True)
print(output_before_cot)


# With chain of thoughts prompting
prompts = ["""
RÃ©solvez le systÃ¨me :
\\begin{align*}
2x + y &= 7, \\
x - y &= 1.
\\end{align*}""",

"""Trouver le reste de $3^100$ divisÃ© par $17$.
"""
]

# Message to use for chain of thoughts prompting
cot_message = "Veuillez procÃ©der Ã©tape par Ã©tape pour rÃ©soudre le problÃ¨me."

# Prepend chain of thoughts message
prompts = [cot_message+"\n"+prmt for prmt in prompts]

output_with_cot = generate_text(prompts, model, tokenizer, do_sample=False, chat_template=True)
print(output_with_cot)

