%%capture
!pip install pip3-autoremove -q
!pip-autoremove torch torchvision torchaudio -y -q
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121 -q
!pip install unsloth -q
!pip install openai -q


# Uncomment this cell if you want generate datasets, track training and upload your models to HuggingFace

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("Huggingface API")
secret_value_1 = user_secrets.get_secret("OpenAI API key")
secret_value_2 = user_secrets.get_secret("WANDB_API_KEY")

import os
os.environ['WANDB_API_KEY'] = secret_value_2
os.environ['HF_ACCESS_TOKEN'] = secret_value_0


import torch
import re
from unsloth import FastLanguageModel

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset


# from openai import OpenAI
# import json
# import traceback

# client = OpenAI(api_key=secret_value_0)

# def generate_sentences(num_sentences=5):

#     try:
#         messages = [
#             {
#                 "role": "system",
#                 "content": "You are a helpful assistant that generates creative and diverse sentences."
#             },
#             {
#                 "role": "user",
#                 "content": f"Generate {num_sentences} unique and diverse sentences on any topic you choose."
#             }
#         ]

#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             temperature=0.8,
#             n=1
#         )

#         content = response.choices[0].message.content
#         sentences = content.split("\n")

#         return [sentence.strip() for sentence in sentences if sentence.strip()]

#     except Exception as e:
#         print(f"Error generating sentences: {e}")
#         print(traceback.format_exc())
#         return []

# def generate_instruction_dataset(original_texts, num_examples=10):

#     dataset = []

#     for original_text in original_texts:
#         original_text = re.sub(r"^\d+\.\s*", "", original_text)
#         messages = [
#             {
#                 "role": "system",
#                 "content": "Generate a rewritten version and instruction prompt for the given text. The output should have three sections - Original Text, Rewritten Text and Prompt (which is the prompt used to generate the Rewritten Text)."
#             },
#             {
#                 "role": "user",
#                 "content": f"Original text: {original_text}\n\nGenerate a rewritten version and an instruction prompt that an LLM would use to rewrite the original text."
#             }
#         ]

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=messages,
#                 temperature=0.7,
#                 n=num_examples
#             )

#             for choice in response.choices:
#                 content = choice.message.content

#                 if "Rewritten Text:" in content and "Prompt:" in content:
#                     try:
#                         rewritten = content.split("**Rewritten Text:**")[1].split("**Prompt:**")[0].strip()
#                         prompt = content.split("**Prompt:**")[1].strip()

#                         example = {
#                             "original": original_text,
#                             "rewritten": rewritten,
#                             "prompt": prompt
#                         }
#                         dataset.append(example)
#                     except Exception as parse_error:
#                         print(f"Error parsing content: {content}")
#                         print(f"Parse error: {parse_error}")
#                 else:
#                     print(f"Unexpected format in response: {content}")

#         except Exception as e:
#             print(f"Error generating dataset for '{original_text}': {e}")
#             print(traceback.format_exc())

#     return dataset


# def save_dataset(dataset, filename="instruction_dataset.jsonl"):
#     with open(filename, 'w') as f:
#         for example in dataset:
#             f.write(json.dumps(example) + '\n')

# generated_sentences = generate_sentences(num_sentences=5)
# dataset = generate_instruction_dataset(generated_sentences)
# save_dataset(dataset)


# Load model and tokenizer

max_seq_length = 512
dtype = None   # None for auto detection
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = secret_value_0
)


# Adding LoRA adapters

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,   # 0 is optimized
    bias = "none",      # "none" is optimized
    use_gradient_checkpointing = "unsloth", # "unsloth" for very long context
    random_state = 42,
)


from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def format_dataset(example):
    data = []
    
    # system
    data.append({"role": "system", "content": "You are an assistant whose job is to return the prompt used to transform the original text to the rewritten text."})

    # human
    human_content = "Original Text: " + example['original'] + ". Rewritten Text: " + example['rewritten'] + "."
    data.append({"role": "user", "content": human_content})

    # assistant
    assistant_content = "Prompt: " + example['prompt']
    data.append({"role": "assistant", "content": assistant_content})
    
    tokenized_output = tokenizer.apply_chat_template(data, tokenize=False, add_generation_prompt=False, return_tensors="pt")
    return {"text": tokenized_output}


# Use your generated dataset here by uncommenting the below code 

# from datasets import Dataset

# def load_jsonl_dataset(file_path):
#     with open(file_path, 'r') as f:
#         data = [json.loads(line) for line in f]
#     return Dataset.from_list(data)


# file_path = "/kaggle/working/instruction_dataset.jsonl"
# dataset = load_jsonl_dataset(file_path)
# formatted_dataset = dataset.map(format_dataset)
# print(formatted_dataset[0])


# Comment out this cell if you are using the generated dataset

from datasets import load_dataset
dataset = load_dataset("billa-man/llm-prompt-recovery", split = "train")

formatted_dataset = dataset.map(format_dataset)
print(formatted_dataset[0])


from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = True, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/kaggle/working/llama-3b-rewrite",
        report_to = "wandb",   # Uncomment this line if you want to track your training
    ),
)


# We use Unsloth's train_on_completions method to only train on the assistant outputs 
# and ignore the loss on the user's inputs.

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)


trainer_stats = trainer.train()


model = FastLanguageModel.for_inference(model) # Enable native 2x faster inference

def inference(original_text, rewritten_text):
    messages = [
        {"role": "user", "content": "Return the prompt that was used to tranform the original text into the rewritten text. Original Text: " + original_text +", Rewritten Text: " + rewritten_text}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    gen_idx = len(inputs[0])
    outputs = model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True,
                             temperature = 1.5, min_p = 0.1)
    response = tokenizer.batch_decode(outputs[:, gen_idx:], skip_special_tokens = True)[0]
    response = response[8:]
    
    return response


original_text = "Recent breakthroughs have demonstrated several ways to induce magnetism in materials using light, with significant implications for future computing and data storage technologies."
rewritten_text = "Light-induced magnetic phase transitions in non-magnetic materials have been experimentally demonstrated through ultrafast optical excitation, offering promising pathways for photomagnetic control in next-generation spintronic devices and quantum computing architectures."

inference(original_text, rewritten_text)


# Uncomment this cell if you want to upload to huggingface_hub

def save_model(model, 
               tokenizer, 
               output_dir: str, 
               repo_id: str,
               push_to_hub: bool = False):
  
  # model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

  if push_to_hub and repo_id:
    print(f"Saving model to '{repo_id}'")
    model.push_to_hub_merged(repo_id, tokenizer, save_method="lora", token=secret_value_0)


save_model(model, tokenizer, "/kaggle/working/llama-3b-rewrite-final", "billa-man/llm-prompt-recovery", True)


# import pandas as pd

# test = pd.read_csv("/kaggle/input/llm-prompt-recovery/test.csv")

# ids = []
# rewrite_prompt = []

# for index, row in test.iterrows():
#     original_text = row['original_text']
#     rewritten_text = row['rewritten_text']
#     prompt = inference(original_text, rewritten_text)
#     ids.append(row['id'])
#     rewrite_prompt.append(prompt)

# df_test = pd.DataFrame({'id': ids, 'rewrite_prompt': rewrite_prompt})
# df_test.to_csv('submission.csv', header=True, index=False)
# sub = pd.read_csv("/kaggle/working/submission.csv")
# sub['rewrite_prompt'][0]

