
%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# Install Flash Attention 2 for softcapping support
import torch
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install --no-deps packaging ninja einops "flash-attn>=2.6.3"



from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gemma-2-2b-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)




model = FastLanguageModel.get_peft_model(
    model,
    r = 24,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 24,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

dataset = load_dataset("mshojaei77/Persian_QA")

# Define a template for question-answer formatting
persian_prompt = """سؤال:
{}

پاسخ:
{}"""

# Initialize the tokenizer to get the End of sequence token
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    """
    Formats questions and answers into a single string using the defined template.
    Appends an EOS token at the end of each formatted text to prevent infinite generation.
    """
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        text = persian_prompt.format(question, answer) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def preprocess_remove_zwnj(examples):
    """
    Removes the Zero Width Non-Joiner (\u200c) from the question and answer fields.
    """
    examples["question"] = [q.replace("\u200c", "") for q in examples["question"]]
    examples["answer"] = [a.replace("\u200c", "") for a in examples["answer"]]
    return examples




# Apply preprocessing to remove \u200c from the dataset
dataset = dataset.map(preprocess_remove_zwnj, batched=True)

# Apply the prompt formatting function to the dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

# Access the 'train' split of the dataset
train_dataset = dataset["train"]

# Apply formatting to the train split as well
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)




from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments
def is_bfloat16_supported():
    return torch.cuda.is_bf16_supported()

# Update the SFTTrainer with the processed train dataset
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,

    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # num_train_epochs=1, # Set this for 1 full training run.
        max_steps=200,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)




trainer_stats = trainer.train()



model.save_pretrained("/content/drive/MyDrive/hothere/phase1")
tokenizer.save_pretrained("/content/drive/MyDrive/hothere/phase1")


from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/hothere/phase1",  # Path to the previously fine-tuned model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)



from datasets import load_dataset,DownloadConfig

# بارگذاری داده‌های مربوط به ویکی‌پدیا
wikipedia_dataset = load_dataset("lifeweb-ai/Divan", data_files='wikipedia/*.parquet', token='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')



def preprocess_remove_zwnj(examples):
    """
    Removes the Zero Width Non-Joiner (\u200c) from the text field.
    """
    examples["text"] = [text.replace("\u200c", "") for text in examples["text"]]
    return examples

# Apply preprocessing to remove \u200c from the Wikipedia dataset
wikipedia_dataset = wikipedia_dataset.map(preprocess_remove_zwnj, batched=True)



def formatting_wikipedia_func(examples):
    """
    Formats the text field by appending the EOS token at the end of each text.
    """
    texts = examples["text"]
    formatted_texts = [text + tokenizer.eos_token for text in texts]
    return {"text": formatted_texts}

# Apply the formatting function to the Wikipedia dataset
wikipedia_dataset = wikipedia_dataset.map(formatting_wikipedia_func, batched=True)






from trl import SFTTrainer
from transformers import TrainingArguments

def is_bfloat16_supported():
    return torch.cuda.is_bf16_supported()
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=wikipedia_dataset["train"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=3,
    packing=False,

    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=15,
        max_steps=100,
        learning_rate=7e-6,
        fp16=not is_bfloat16_supported(),

        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant",
        seed=3407,
        output_dir="outputs_wikipedia",
        report_to="none",
    ),
)




trainer_stats = trainer.train()




model.save_pretrained("/content/drive/MyDrive/hothere/phase2")
tokenizer.save_pretrained("/content/drive/MyDrive/hothere/phase2")


from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/hothere/phase2",  # Path to the previously fine-tuned model
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)



from datasets import load_dataset,DownloadConfig

# بارگذاری داده‌های مربوط به ویکی‌پدیا
twitter_dataset = load_dataset("lifeweb-ai/Divan", data_files='twitter/*.parquet', token='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')



twitter_dataset


import re

def preprocess_text(examples):
    """
    Preprocess the text by:
    1. Removing Zero Width Non-Joiner (\u200c).
    2. Removing placeholders like [httplink].
    """
    processed_texts = []
    for text in examples["text"]:
        text = text.replace("\u200c", "")
        text = re.sub(r"\[httplink\]", "", text)
        processed_texts.append(text.strip())
    examples["text"] = processed_texts
    return examples

subset_twitter_dataset = twitter_dataset["train"].select(range(100000))

subset_twitter_dataset = subset_twitter_dataset.map(preprocess_text, batched=True)

def formatting_twitter_func(examples):
    """
    Formats the text by appending the EOS token at the end of each text.
    """
    texts = examples["text"]
    formatted_texts = [text + tokenizer.eos_token for text in texts]
    return {"text": formatted_texts}

subset_twitter_dataset = subset_twitter_dataset.map(formatting_twitter_func, batched=True)

train_dataset = subset_twitter_dataset




from trl import SFTTrainer
from transformers import TrainingArguments

def is_bfloat16_supported():
    return torch.cuda.is_bf16_supported()
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=subset_twitter_dataset,
    dataset_text_field="text",
    max_seq_length=248,
    dataset_num_proc=2,
    packing=False,

    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        max_steps=200,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        seed=3407,
        output_dir="outputs_twitter",
        report_to="none",
    ),
)




trainer_stats = trainer.train()




model.save_pretrained("/content/drive/MyDrive/hothere/phase3")
tokenizer.save_pretrained("/content/drive/MyDrive/hothere/phase3")



model.push_to_hub("ItsAbba3/wikipersian_gemma2_2b_final", token = "MY_token")
tokenizer.push_to_hub("ItsAbba3/wikipersian_gemma2_2b_final", token = "MY_token")



FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    persian_prompt.format(
        "آمل کجاست و چقدر جمعیت داره؟",
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output_text



FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    persian_prompt.format(
        "مردم در مورد حکومت چه فکر میکنند؟",
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output_text



FastLanguageModel.for_inference(model)
inputs = tokenizer(
[
    persian_prompt.format(
        "در چند جمله مردم ایران و فرهنگ سالهای اخیر ایران رو توضیح بده؟",
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 1024, use_cache = True)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
output_text







