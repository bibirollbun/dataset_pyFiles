# Installation
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth

!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


# Get the model from Unsloth
from unsloth import FastLanguageModel
import torch

torch._dynamo.config.cache_size_limit = 64  # or higher  

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit", # Text only but it's for Ollama so it should be fine for now
    dtype = None, # None for auto detection
    max_seq_length = 2048, # Used for training for function call responses
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, 
    attn_implementation="flash_attention", 
    # token = "hf_...", # use one if using gated models
)


# Use this chat template for training tool calls
tokenizer.chat_template = (
    "{{ bos_token }}{% for message in messages %}{% if message['role'] != 'system' %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)


# Get model response before training
from transformers import TextStreamer

def model_infer(model, text):
    print("User input: ")
    print(text)
    
    messages = [{
        "role": "user",
        "content": [{
            "type" : "text",
            "text" : text,
        }]
    }]
    # Convert the messages to the correct format for generation
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")
    
    print("Model response: ")
    outputs = model.generate(
        **inputs,
        max_new_tokens = 2048, # Increase for longer outputs!
        # Recommended Gemma-3 settings!
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )
    print(tokenizer.batch_decode(outputs))


# Get the finetuning model
from peft import TaskType

ft_model = FastLanguageModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!
    target_modules=["gate_proj","q_proj","o_proj","k_proj","down_proj","up_proj","v_proj"],
    use_gradient_checkpointing="unsloth",
    task_type = TaskType.CAUSAL_LM, 
    bias= "none",
    use_rslora=False,
    loftq_config=None,
    r = 16,
    lora_alpha = 64,
    lora_dropout = 0.05,    
)


# Load the dataset
from datasets import load_dataset
dataset = load_dataset("lmassaron/hermes-function-calling-v1", split="train")
eval_dataset = load_dataset("lmassaron/hermes-function-calling-v1", split="test[:25]")
dataset[0]


# Convert the dataset to the correct format for finetuning
def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset =  dataset.map(formatting_prompts_func, batched = True)
eval_dataset =  eval_dataset.map(formatting_prompts_func, batched = True)
dataset[100]["text"]


# Setup the fine-tuning trainer
from trl import SFTTrainer, SFTConfig
training_arguments = SFTConfig(
        eval_strategy="steps",
        do_eval=True,
        optim="adamw_torch_fused",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.1,
        eval_steps=10,
        max_steps=120,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        report_to="none"        
)

trainer = SFTTrainer(
    model = ft_model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset= eval_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    bias = "none",
    args = training_arguments,  
)


# Train the model on the responses only, ignore user instructions
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


# Verify the chat template was applied correctly. Only 1 <bos> token should be present.
tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# Verify user instruction masked
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, "")


# Train the model
trainer_stats = trainer.train()
trainer_stats


FastLanguageModel.for_inference(ft_model)
e = eval_dataset[20]["conversations"][0]["content"]
model_infer(ft_model, e)

