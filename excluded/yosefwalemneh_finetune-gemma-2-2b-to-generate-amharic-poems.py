! pip install -Uq transformers datasets evaluate accelerate trl peft


! pip uninstall -yq wandb


!rm -r gemma-2-2b-it-finetuned-amharic


from datasets import load_dataset

am_instruct = load_dataset("EthioNLP/Amharic_Instruction_dataset")
am_instruct = am_instruct.filter(lambda row: row["datasource"] in (
    # 'amharic_mezmur_completion',
    # 'amharic_mezmur_generation',
    'amharic_poem_completion',
    'amharic_poem_generation',
    'amharic_story_generation',
    'amharic_zefen_completion',
    'amharic_zefen_generation',
))
am_instruct


from collections import Counter

Counter(am_instruct["train"]["datasource"])


am_aya = load_dataset("Henok/aya_amharic_dataset")
am_aya


am_hermes = load_dataset("rasyosef/amharic-openhermes")
am_hermes


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)


from transformers import pipeline

gemma2_am = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device="cuda"
  )


messages = [{"role": "user", "content": "በአማርኛ ግጥም ስጠኝ"}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


messages = [{"role": "user", "content": """ተረት ንገረኝ

ጅብና አንበሳ"""}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


messages = [{"role": "user", "content": "የኢትዮጵያ ዋና ከተማ ስም ምንድን ነው?"}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


messages = [{"role": "user", "content": "የጃፓን ዋና ከተማ ስም ምንድን ነው?"}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


# Format instructions in ChatML
am_instruct = am_instruct.filter(lambda example: example['instruction'] and example['output'], num_proc=4)

am_instruct = am_instruct.map(
    lambda example: {"messages": [
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}" if example['input'] else example['instruction'].strip()},
        {"role": "assistant", "content": example['output']},
    ]}, num_proc=4
  )

# Apply chat template
am_instruct = am_instruct.map(
    lambda example: {"messages_templated": tokenizer.apply_chat_template(example["messages"], tokenize=False)}, num_proc=4
  )

am_instruct


# Format instructions in ChatML
am_aya = am_aya.map(
    lambda example: {"messages": [
        {"role": "user", "content": example['inputs']},
        {"role": "assistant", "content": example['targets']},
    ]}, num_proc=4
  )

# Apply chat template
am_aya = am_aya.map(
    lambda example: {"messages_templated": tokenizer.apply_chat_template(example["messages"], tokenize=False)}, num_proc=4
  )

am_aya


# Apply chat template
am_hermes = am_hermes.map(
    lambda example: {"messages_templated": tokenizer.apply_chat_template(example["messages"], tokenize=False)}, num_proc=4
  )

am_hermes


from datasets import DatasetDict, concatenate_datasets

am_dataset = DatasetDict({
    "train": concatenate_datasets([ 
        am_instruct["train"].shuffle(seed=42).select(range(6_000)),
        am_aya["train"],
        am_hermes["train"]
    ]),
})
am_dataset


samples = am_dataset["train"].shuffle().select(range(5))

for sample in samples:
  print(sample["messages_templated"])
  print("\n-----------------------------------------------------------\n")


import re

# Remove instructions that have too many English characters
am_dataset = am_dataset.filter(lambda x: len(re.findall('[a-zA-Z]', "".join([m["content"] for m in x['messages']]))) < 0.04*len(x['messages_templated']), num_proc=4)

# Truncate input 448 tokens in order not to exceed GPU memory
am_dataset = am_dataset.map(lambda x: {"messages_templated": tokenizer.decode(tokenizer(x['messages_templated'])["input_ids"][1:449])}, num_proc=4)
am_dataset


samples = am_dataset["train"].shuffle().select(range(5))

for sample in samples:
  print(sample["messages_templated"])
  print("\n-----------------------------------------------------------\n")


am_dataset_final = am_dataset["train"].shuffle(seed=42).train_test_split(test_size=0.025, seed=42)
am_dataset_final


print(model)


import torch
from peft import LoraConfig, get_peft_model, cast_mixed_precision_params

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Target all linear layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
)

model = get_peft_model(model, peft_config)
cast_mixed_precision_params(model, dtype=torch.float16)
model.print_trainable_parameters()


from trl import SFTConfig, SFTTrainer

eval_steps = 500
save_steps = eval_steps
logging_steps = eval_steps

print("Eval Steps:", eval_steps)
print("Save Steps:", save_steps)

new_model_id = "gemma-2-2b-it-finetuned-amharic"

sft_config = SFTConfig(
    output_dir=new_model_id,
    dataset_text_field="messages_templated",
    max_seq_length=448,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_steps=250,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,
    # packing=True,
    # push_to_hub=True, # uncomment if you want to save model checkpoints to huggingface
    # hub_private_repo=True,
    logging_strategy="steps",
    logging_steps=logging_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=1,
    seed=42,
    load_best_model_at_end=True
)

trainer = SFTTrainer(
    model,
    args=sft_config,
    train_dataset=am_dataset_final["train"],
    eval_dataset=am_dataset_final["test"],
    tokenizer=tokenizer
)


import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# Start training
trainer.train()


import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


trainer.save_model()


# import os
# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()
# os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_WRITE_YOSEFW")

# new_model_id = "gemma-2-2b-it-finetuned-amharic"

# model.push_to_hub(new_model_id)
# tokenizer.push_to_hub(new_model_id)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "unsloth/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# Load our fine-tuned LoRA Adapter from HuggingFace
# peft_model_id = "yosefw/gemma-2-2b-it-finetuned-amharic"

# Load our fine-tuned LoRA Adapter from a Local Directory
peft_model_id = "./gemma-2-2b-it-finetuned-amharic"
model.load_adapter(peft_model_id)


from transformers import pipeline

gemma2_am = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device="cuda"
  )


messages = [{"role": "user", "content": "በአማርኛ ግጥም ስጠኝ"}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


messages = [{"role": "user", "content": """ተረት ንገረኝ

ጅብና አንበሳ"""}]
response = gemma2_am(messages, max_new_tokens=512, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


messages = [{"role": "user", "content": "የኢትዮጵያ ዋና ከተማ ስም ምንድን ነው?"}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])


messages = [{"role": "user", "content": "የጃፓን ዋና ከተማ ስም ምንድን ነው?"}]
response = gemma2_am(messages, max_new_tokens=256, repetition_penalty=1.1)[0]["generated_text"][-1]
print(response["content"])




