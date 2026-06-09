import warnings
import logging


logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')


# Install essential libraries for NLP and efficient fine-tuning
!pip install -qqq -U transformers trl peft datasets huggingface_hub bitsandbytes accelerate sentence-transformers
# Install Kaggle-specific utilities for dataset and model management
!pip install -qqq --upgrade kagglehub[pandas-datasets,hf-datasets]
# Install `liger-kernel` Efficient Triton Kernels for LLM Training
!pip install liger-kernel


import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format, SFTConfig


import kagglehub
from kagglehub import KaggleDatasetAdapter
dataset_name = "abdullahabdelrhim/msa-prompts-stories" # The instruction dataset to use


# Adjust precision and attention based on GPU
if torch.cuda.get_device_capability()[0] >= 8:
    # Use bfloat16 and FlashAttention-v2 for newer GPUs (Compute Capability >= 8
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
    !pip install -qqq flash-attn   # Install FlashAttention-v2 library
else:
    # Use float16 and eager attention for older GPUs
    torch_dtype = torch.float16
    attn_implementation = "eager"


## config
from transformers import BitsAndBytesConfig, set_seed

# Set seed for reproducibility -- you don't need this unless you want full reproducibility
set_seed(42)

# BitsAndBytes configuration for memory-efficient model loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4", # Use Normal Float 4 data type
    bnb_4bit_compute_dtype=torch_dtype, # Set compute data type based on GPU
    bnb_4bit_use_double_quant=True, # Enable double quantization for more memory saving
)

base_model = "/kaggle/input/gemma/transformers/2b-it/3"
# Load model with quantization and optimized attention
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# Disable caching during training for gradient computation efficiency
model.config.use_cache = False
model.gradient_checkpointing_enable()


tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.padding_side = 'right' # Fix weird overflow issue with fp16 training
tokenizer.pad_token = tokenizer.eos_token


dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,  # indicates that the dataset is structured in a manner compatible with Hugging Face Datasets
    dataset_name,
    "Formatted-MSA-prompts-stories-for-fine-tuning.csv",
)

def format_chat_template(row):
    """
     transforming individual data samples (rows) into a structured format suitable for instruction-based fine-tuning.
    """
    row_json = [
                {"role": "user", "content": row["Prompt"]},
                {"role": "assistant", "content": row["Story"]}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = dataset.map(format_chat_template, num_proc=4)

split_dataset = dataset.train_test_split(train_size=0.9, test_size=0.1)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")


# show example
print(train_dataset["text"][2])


token_lengths = [len(tokenizer(text['Story'])['input_ids']) for text in train_dataset]


import matplotlib.pyplot as plt
plt.hist(token_lengths, bins=30, color='blue')
plt.title('Token Length Distribution')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.show()


## Low-Rank Adapter (LoRA) Configuration for Parameter-Efficient Fine-Tuning
adaptor_model_name = "gemma-2b-stories-arabic-finetuned"  # Identifier for the LoRA adapter
output_dir = f"/kaggle/working/{adaptor_model_name}"  # Directory for saving the fine-tuned adapter 

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        r=64,  # Rank of the low-rank matrix
        lora_alpha=128,  # Scaling factor for LoRA typically start with alpha=r and go upto alpha=2r
        lora_dropout=0.1, # Dropout probability for regularization
        bias="none", # Disables bias terms in LoRA layers
        target_modules="all-linear", # Applies LoRA to all linear layers
        task_type="CAUSAL_LM", # Specifies causal language modeling task
)


# Integration of Low-Rank Adapters into the Base Model: Applying Parameter-Efficient Fine-Tuning
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# Training arguments
training_arguments = SFTConfig(
    output_dir=output_dir,                  # directory to save and repository id
    logging_dir="./logs",                   # Directory for storing training logs
    # max_steps=100,                         # for debugging
    num_train_epochs=5,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=3,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    gradient_checkpointing_kwargs={"use_reentrant":False}, # Solve ==> RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    optim="paged_adamw_32bit",              # use paged fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    bf16=False,                             # use bfloat16 precision
    tf32=False,                             # Disable TF32 mixed precision
    learning_rate=2e-4,                     # Optimizer learning rate 
    max_grad_norm=0.3,                      # Gradient norm clipping threshold
    warmup_steps=50,                        # Linear learning rate warmup steps
    weight_decay = 0.01,
    lr_scheduler_type="cosine",             # Cosine learning rate decay schedule
    push_to_hub=False,                      # Prevent automatic push to Hugging Face Hub
    use_liger=True,                         # Enable Efficient Triton Kernels for LLM Training
    # neftune_noise_alpha=0.01,              # NEFTune is a technique to boost the performance of chat models. (didn't work)
    report_to="none",                       # Suppress reporting to external services
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)


# start training, the model saved to output directory
trainer.train()


# Save artifacts
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


def get_loss_logs(trainer, loss_type='loss'):
    logs = trainer.state.log_history
    loss = [(x["step"], x[loss_type]) for x in logs if loss_type in x.keys()]
    return [x[1] for x in sorted(loss, key=lambda x: x[0])]

train_loss = get_loss_logs(trainer, 'loss')

plt.plot(train_loss, label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Execute this cell only if you are having memory issues.
import torch
import gc

def clear_hardwares():
    gc.collect()
    if torch.cuda.is_available():
        torch.clear_autocast_cache()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    gc.collect()

# Flush memory
del trainer, model
clear_hardwares()


from peft import AutoPeftModelForCausalLM

# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")


del merged_model
clear_hardwares()


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

# Authenticate to Kaggle
os.environ["KAGGLE_USERNAME"] = user_secrets.get_secret("kaggle_username")
os.environ["KAGGLE_KEY"] = user_secrets.get_secret("kaggle_key")


import kagglehub
from kagglehub.config import get_kaggle_credentials

cre = get_kaggle_credentials()
username = cre.username

# For PyTorch framework & `2b` variation.
# Replace the framework with "jax", "other" based on which framework you are uploading to.
kagglehub.model_upload(f'{username}/gemma2-kaggle/pyTorch/gemma-2b-stories-arabic-finetuned', 
                       output_dir, version_notes='improved accuracy', ignore_patterns=["checkpoint-*/*",])

print(f"The fine-tuned model was successfully uploaded to <a href='https://www.kaggle.com/models/{username}/gemma2-kaggle/pyTorch/gemma-2b-stories-arabic-finetuned'>Kaggle Models</a>.")


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Model with PEFT adapter
peft_model_id = "/kaggle/working/gemma-2b-stories-arabic-finetuned/"
tokenizer = AutoTokenizer.from_pretrained(peft_model_id, use_fast=False)
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch_dtype)
model.to(device)
# Sets the module in evaluation mode.
model.eval()
# apply the torch compile transformation
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)


def generate_story(prompt_example: str, add_arabic_prefix: bool =False, generation_params: dict = None) -> str:
    """Generates an Arabic story based on the given prompt."""
    if not prompt_example:
        return "Empty prompt"
    arabic_prefix = ""
    if add_arabic_prefix is True:
        # Add prefix to enforce Arabic language
        arabic_prefix = "أنت راوى قصص باللغة العربية. اكمل القصة التالية باللغة العربية فقط"
    modified_prompt = arabic_prefix + prompt_example
    if add_arabic_prefix is True:
        modified_prompt = modified_prompt + "\n\n" + "اكمل باللغة العربية فقط."
    # Format prompt
    message = [
        {"role": "user", "content": modified_prompt}
    ]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    # Tokenize inputs and move to device
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    # Default generation parameters
    default_params = {
         "max_new_tokens": 640,
         "num_beams": 4, # Beam search for better creative writing
          # Prevent repetition issues
         "eos_token_id": tokenizer.eos_token_id,
         "pad_token_id": tokenizer.eos_token_id, 
         "no_repeat_ngram_size": 3,  
         "repetition_penalty": 1.2, # discourage the model from repeating tokens
         "num_logits_to_keep": 1, # Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size
         "use_cache": True # Enable caching for faster generation
     }
    
    # Merge default parameters with user-provided parameters
    if generation_params:
         params = {**default_params, **generation_params}
    else:
      params = default_params
        
    generated_story = ""
    # Generate the story with adjusted parameters
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                **params
            )
        # Decode and print the generated story
        generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except RuntimeError as e:
        print(f"An error occurred during generation: {e}")
    return generated_story


# (Examples showing adaptability to different prompt styles)
print("\n--- Testing Model Robustness without prompt prefix ---")
prompt_examples = [
    "يحكى أن صيادًا بسيطًا وجد مصباحًا سحريًا",
]

for prompt in prompt_examples:
    print(f"Prompt: {prompt}")
    story = generate_story(prompt)
    print(f"Generated Story: {story}")


# (Examples showing adaptability to different prompt styles)
print("\n--- Testing Model Robustness with Diverse Inputs ---")
prompt_examples = [
    "يحكى أن صيادًا بسيطًا وجد مصباحًا سحريًا",
    "يحكى أن في مدينة بغداد القديمة، كان هناك حكيم مشهور بذكائه وفطنته. جاء إليه شاب يائس يشكو",
     "اكتب قصة قصيرة تكون مفهومة للأطفال في عمر 4-7. نهاية القصة يجب أن تكون ذات عبرة. بلد الأحداث هي المغرب. يجب أن تحتوي القصة على شعور بالفضول.",
]

for prompt in prompt_examples:
    print(f"Prompt: {prompt}")
    story = generate_story(prompt, add_arabic_prefix=True)
    print(f"Generated Story: {story}")



print("\n--- Testing Model Robustness with Different Decoding Strategies---")

prompt = "يحكى أن في قديم الزمان كان هناك ملك يحب المغامرات."
print(f"Prompt: {prompt}")
# Example with different number of beams and repetition penalty
generation_params = {"num_beams": 5, "repetition_penalty": 1.3, "max_new_tokens": 500}
story = generate_story(prompt, add_arabic_prefix=True, generation_params=generation_params)
print(f"Generated Story (Advanced Beam Search): {story}")


from sentence_transformers import SentenceTransformer

# Load the MiniLM model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to get MiniLM embeddings using sentence-transformers
def get_embedding(text):
    # Directly encode the text using the loaded model
    return embedding_model.encode(text, convert_to_tensor=True)


# Function to retrieve top k similar examples based on cosine similarity
def get_top_k_similar_examples(question, dataset_df, k=3):
    question_embedding = get_embedding(question)
    similarities = []

    for _, row in dataset_df.iterrows():
        example_embedding = get_embedding(row["Prompt"])
        similarity = torch.cosine_similarity(question_embedding, example_embedding, dim=0).item()
        similarities.append((similarity, row))

    # Sort by similarity and select top k examples
    top_k_examples = sorted(similarities, key=lambda x: x[0], reverse=True)[:k]
    return [example[1] for example in top_k_examples]


story_few_shot_df = eval_dataset.to_pandas()

# Configuration for Few-Shot Prompting
few_shot_config = {
    "num_examples": 3,  # Number of examples to use for few-shot prompting
}
def generate_story_few_shot_prompting(prompt_example: str) -> str:
    if not prompt_example:
        return "Empty prompt"
    num_examples = few_shot_config.get("num_examples", 2)  # Get the number of examples, default to 2
    few_shot_examples = get_top_k_similar_examples(prompt_example, story_few_shot_df, k=num_examples)
    prompt_examples = "\n".join([f"مثال {_i}: {ex['Prompt']} \nقصة {_i}: {ex['Story']}" for _i, ex in enumerate(few_shot_examples, start=1)])
    # Add prefix to enforce Arabic language
    arabic_prefix= "أنت سارد قصص مُلهم. احكِ قصة باللغة العربية فقط"
    modified_prompt = arabic_prefix +"\n\n" + prompt_examples + "\n\n------------------\n"+ prompt_example + "\n\n" + "اكمل القصة باللغة العربية فقط."
    # Format prompt
    message = [
        {"role": "user", "content": modified_prompt}
    ]
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
    # Tokenize inputs and move to device
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
    generated_story = ""
    # Generate the story with adjusted parameters
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=640,
                # Prevent repetition issues 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_beams=4, # Beam search for better creative writing
                repetition_penalty=1.2, # discourage the model from repeating tokens
                num_logits_to_keep=1, #  Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size
                use_cache=True,
            )
        # Decode and print the generated story
        generated_story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except RuntimeError as e:
        print(f"An error occurred during generation: {e}")
    return generated_story


# (Examples showing adaptability to different prompt styles)
print("\n--- Testing Model Robustness with Few Shots ---")
prompt_examples = [
    "يحكى أن صيادًا بسيطًا وجد مصباحًا سحريًا",
    'تخيل حديقة سرية لا تفتح إلا في ليلة اكتمال القمر. اكتب قصة قصيرة عن طفل يكتشف هذه الحديقة وما يجده بداخلها. يجب أن يكون هناك شعور بالغموض والجمال.',
]

for prompt in prompt_examples:
    print(f"Prompt: {prompt}")
    story = generate_story_few_shot_prompting(prompt)
    print(f"Generated Story: {story}")


# --- Basic RAG Example ---
print("\n--- Basic Retrieval-Augmented Generation Example ---")
knowledge_base = {
    "غيلان": "الغيلان هي مخلوقات أسطورية في الفولكلور العربي، غالبًا ما توصف بأنها تسكن الأماكن المهجورة ولها قدرات خارقة.",
    "ألف ليلة وليلة": "ألف ليلة وليلة هي مجموعة قصصية عربية مشهورة عالميًا، تضم حكايات مثل علي بابا والأربعين حرامي وسندباد البحري.",
    "بساط الريح": "بساط الريح هو وسيلة نقل سحرية تظهر في العديد من القصص الفلكلورية العربية، قادرة على الطيران بسرعة فائقة.",
}

def retrieve_knowledge(query):
    relevant_info = []
    for keyword, info in knowledge_base.items():
        if keyword in query:
            relevant_info.append(info)
    return " ".join(relevant_info)

user_query_rag = "اكتب قصة قصيرة عن مغامرة باستخدام بساط الريح."
retrieved_info = retrieve_knowledge(user_query_rag)
arabic_prefix= "أنت سارد قصص مُلهم. احكِ قصة باللغة العربية فقط" + "\n\n"
augmented_prompt = arabic_prefix + f"المعلومات الأساسية: {retrieved_info}\n\nاكتب قصة قصيرة عن مغامرة باستخدام بساط الريح."
augmented_prompt += "\n\n" + "اكمل القصة باللغة العربية فقط."
message_rag = [
    {"role": "user", "content": augmented_prompt}
]
model_inputs = tokenizer.apply_chat_template(message_rag, add_generation_prompt=True, return_tensors="pt").to("cuda")
input_length = model_inputs.shape[1]

try:
    with torch.inference_mode():
        outputs_rag = model.generate(
            model_inputs,
            max_new_tokens=640,
            # to prevent repetition 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_beams=4,  # Beam search for better creative writing
            repetition_penalty=1.2,
            num_logits_to_keep=1,
            use_cache=True
        )
    print(tokenizer.batch_decode(outputs_rag[:, input_length:], skip_special_tokens=True)[0])
    #print(generated_story_rag)
except RuntimeError as e:
    print(f"An error occurred during generation: {e}")


print("\n--- Comparison Example with Reference Story ---")
first_example = eval_dataset[0]
comparison_prompt = first_example["Prompt"]
reference_story = first_example["Story"]

generated_story = generate_story(comparison_prompt)
print(f"Comparison Prompt:\n{comparison_prompt}\n")

if reference_story:
    print(f"Reference Story:\n{reference_story}\n")
else:
    print(f"Reference Story: Not found in dataset.\n")

print(f"Generated Story:\n{generated_story}\n")

