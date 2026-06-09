%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
torch.cuda.empty_cache()


from unsloth import FastModel

BASE_MODEL_NAME = "unsloth/gemma-3n-E2B-it"

model, tokenizer = FastModel.from_pretrained(
    model_name = BASE_MODEL_NAME, 
    dtype = None, # None for auto detection
    max_seq_length = 1024, 
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False,
)


from transformers import TextStreamer
import gc
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


%%capture
pip install python-docx

import os
import re
from collections import defaultdict
from docx import Document
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


# === CONFIG ===
INPUT_DIR = "/kaggle/input/raw-docs"  # folder with your .docx files
AGE_GROUP_REGEX = re.compile(r"Ages?\s*([\d\-]+)", flags=re.IGNORECASE)
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
MAX_LENGTH = 256
IGNORE_INDEX = -100

# === Helpers ===
def infer_age_group_from_filename(path):
    fname = os.path.basename(path)
    m = AGE_GROUP_REGEX.search(fname)
    return m.group(1) if m else "unknown"

def is_topic_header(text):
    return bool(re.match(r"^[A-Z][a-zA-Z& ]+$", text)) and not text.endswith("?")

def clean_answer(text):
    text = text.replace("Sparky's Answer:", "").strip()
    return re.split(r"Wow! Fact:.*?Wow!$", text, flags=re.DOTALL)[0].strip()

def normalize_subject(subject):
    if not subject:
        return "unknown"
    s = subject.strip().lower()
    if "math" in s:
        return "math"
    if "science" in s:
        return "science"
    if "geography" in s:
        return "geography"
    if "history" in s:
        return "history"
    return s
def parse_docx_file(path):
    print(f"Parsing {path}")
    doc = Document(path)
    age_group = infer_age_group_from_filename(path)
    examples = []
    current_topic = None
    current_question = None
    current_answer_parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if is_topic_header(text):
            current_topic = text
        elif text.endswith("?") and "Sparky's Answer" not in text:
            if current_question and current_answer_parts:
                cleaned = clean_answer(" ".join(current_answer_parts))
                examples.append({
                    "question": current_question,
                    "answer": cleaned,
                    "subject": normalize_subject(current_topic),
                    "age_group": age_group,
                    "format": "open_ended",
                })
                current_answer_parts = []
            current_question = text
        elif "Sparky's Answer:" in text or current_answer_parts:
            current_answer_parts.append(text)

    if current_question and current_answer_parts:
        cleaned = clean_answer(" ".join(current_answer_parts))
        examples.append({
            "question": current_question,
            "answer": cleaned,
            "subject": normalize_subject(current_topic),
            "age_group": age_group,
            "format": "open_ended",
        })

    print(f"  → extracted {len(examples)} examples")
    return examples


# Collect all examples by age group
by_age = defaultdict(list)
for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if fname.lower().endswith(".docx"):
            path = os.path.join(root, fname)
            exs = parse_docx_file(path)
            for ex in exs:
                by_age[ex["age_group"]].append(ex)



# Build raw_datasets_by_age
by_age_raw = defaultdict(list)
for root, _, files in os.walk(INPUT_DIR):
    for fname in files:
        if not fname.lower().endswith(".docx"):
            continue
        path = os.path.join(root, fname)
        examples = parse_docx_file(path)  # your existing parser
        for ex in examples:
            by_age_raw[ex["age_group"]].append(ex)

raw_datasets_by_age = {}
for age_group, examples in by_age_raw.items():
    ds = Dataset.from_list(examples)
    try:
        train_testval = ds.train_test_split(
            test_size=1 - SPLIT_RATIOS["train"],
            seed=42,
            stratify_by_column="subject",
        )
        val_test = train_testval["test"].train_test_split(
            test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
            seed=43,
            stratify_by_column="subject",
        )
        print(f"-- Stratified succesfully for Age group {age_group}.\n")
    except Exception:
        train_testval = ds.train_test_split(test_size=1 - SPLIT_RATIOS["train"], seed=42)
        val_test = train_testval["test"].train_test_split(
            test_size=SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
            seed=43,
        )
        print(f"-- Stratification failed for Age group {age_group}.\n")

    raw_datasets_by_age[age_group] = DatasetDict({
        "train": train_testval["train"],
        "val": val_test["train"],
        "test": val_test["test"],
    })


def system_prompt_for_age(age_group):
  if age_group == "2-4":
    system_prompt = ("You are a gentle and playful tutor for toddlers aged 2 to 4."
        "Use very short, simple words and sentences. Speak like a friendly character "
        "from a children's show. Use repetition, sound effects, and lots of excitement!")
  elif age_group == "5-7":
    system_prompt  = ("You are a cheerful and friendly tutor for children aged 5 to 7."
    " Use simple words and fun metaphors to explain things clearly. Be playful and keep "
    "answers short and exciting. You can use characters like 'sugar bugs' or 'energy monsters' "
    "to make it fun.")
  elif age_group == "8-10":
    system_prompt  = ("You are a smart and encouraging tutor for children aged 8 to 10."
    " Explain things using clear, age-appropriate language. Add interesting facts or "
    "comparisons that make learning fun. You can use simple science words and "
    "real-world examples.")
  elif age_group == "1113":
    system_prompt  = ("You are a knowledgeable and relatable tutor for preteens aged 11 to 13."
    " Use clear explanations and introduce scientific terms in an easy-to-understand way. "
    "Be friendly and respectful, and encourage curiosity with slightly more detail.")
  elif age_group == "1415":
    system_prompt  = ("You are an insightful and respectful tutor for teenagers aged 14 to 15. "
    "Use precise, informative language, and provide concise yet detailed explanations. "
    "Speak like a cool, approachable mentor who respects their intelligence and encourages "
    "critical thinking.")
  else:
    system_prompt = ("You are an insightful and respectful tutor for people above age 16.")
  return system_prompt



# Add "text" field for SFTTrainer and tokenize
def add_text_field(ds_dict, age_group):
    def make_text(example):
        convo = [
            {"role": "system", "content": system_prompt_for_age(age_group)},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        ).removeprefix("<bos>")
        return example
    return ds_dict.map(make_text, batched=False)

# Combine datasets from all age groups
from datasets import concatenate_datasets 
combined_dataset = DatasetDict()
for age_group, raw_ds in raw_datasets_by_age.items():
    # Apply the modified add_text_field to each age group's dataset
    processed_ds = add_text_field(raw_ds, age_group)
    # Concatenate the splits
    for split in ["train", "val", "test"]:
        if split not in combined_dataset:
            combined_dataset[split] = processed_ds[split]
        else:
            combined_dataset[split] = concatenate_datasets([combined_dataset[split], 
                                                            processed_ds[split]])

ds = combined_dataset


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from trl import SFTTrainer, SFTConfig
import torch
import copy

# Load base model once (frozen except adapters)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Add LoRA adapters to the model
model = FastModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_cache = False,
    use_gradient_checkpointing=True,  # True or "unsloth" for very long context
    use_rslora=True,
    random_state=73
)

model.print_trainable_parameters()

# Create SFTTrainer for this age
trainer = SFTTrainer(
    model=model,
    tokenizer= tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds.get("val"),
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    ),
)


from unsloth import unsloth_train
trainer_stats = unsloth_train(trainer) 


GB_CONVERSION = 1024 ** 3
SECONDS_TO_MINUTES = 60

# Memory calculations
used_memory_gb = torch.cuda.max_memory_reserved() / GB_CONVERSION
used_memory_for_training_gb = used_memory_gb - start_gpu_memory
used_percentage = (used_memory_gb / max_memory) * 100
training_percentage = (used_memory_for_training_gb / max_memory) * 100

# Time calculations
runtime_seconds = trainer_stats.metrics['train_runtime']
runtime_minutes = runtime_seconds / SECONDS_TO_MINUTES

print("TRAINING STATISTICS")
print("=" * 50)
print(f"Training time: {runtime_seconds:.1f} seconds ({runtime_minutes:.2f} minutes)")
print(f"Peak memory usage: {used_memory_gb:.3f} GB ({used_percentage:.1f}% of max)")
print(f"Memory for training: {used_memory_for_training_gb:.3f} GB ({training_percentage:.1f}% of max)")
print("=" * 50)


# Merge lora adapters and save the fine-tuned model
merged_model_save_path = f"./merged_2B_model"
if not os.path.exists(merged_model_save_path):
    os.makedirs(merged_model_save_path)
model.save_pretrained_merged(merged_model_save_path, tokenizer)


# Quantize and convert to gguf format
model.save_pretrained_gguf(
    merged_model_save_path,
    quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
)

