!pip install pip3-autoremove -q
!pip install -q torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth -q
!pip install --upgrade -q transformers==4.53.2 "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.0"


import os
import pandas as pd
import numpy as np
import unsloth
import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
VER = 1
model_name ="unsloth/Qwen2.5-Math-1.5B-bnb-4bit"
EPOCHS = 1
DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)
MAX_LEN = 2048


# Load and preprocess the training data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception

# We use LabelEncoder to convert the 65 unique target strings into integers from 0 to 64.
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


# Feature Engineering
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
train['is_correct'] = train.apply(lambda x: "yes" if x['is_correct'] == 1 else "no", axis=1)


# Create a list of unique characters to represent each class. This is the core of the new strategy.
# The model will learn to output one of these characters.
special_character_list = [
    '■', '□', '▲', '△', '▼', '▽', '◆', '◇', '○', '●', '★', '☆', '♦', '♥', '♠', '♣',
    '§', '†', '‡', '※', '∞', '±', '≠', '≈', '√', '∑', '∏', '∆', 'Ω', 'μ', '∂', '→',
    '←', '↑', '↓', '↔', '↕', '〈', '〉', '『', '』', '│', '─', '┌', '┐', '└', '┘', '┼',
    '█', '▓', '▒', '£', '¥', '€', '₩', '©', '®', '™', '♪', '♫', '☀', '☁', '☂', '☃', '☎'
]

# Map the integer label to its corresponding special character
train['special_label'] = train['label'].apply(lambda x: special_character_list[int(x)])


# Load model and tokenizer using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)

# Prepare the model for LoRA fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=8, # Increased rank for better learning
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen2.5", # Using standard qwen2 chat template
)

# This detailed prompt explicitly tells the model the classification task and the exact output format.
# It provides the mapping from the special character to the full class name.
class_mappings = [f"{special_character_list[i]}: {le.classes_[i]}" for i in range(n_classes)]


SYS_PROMPT = f"""You are an expert at analyzing math student responses. Your task is to classify the student's explanation into one of the following Category:Misconception classes.

Respond with ONLY the single character corresponding to the correct classification.

Available classifications:
{', '.join(class_mappings)}

Analyze the given input and provide your classification.
"""

# Create user prompt
user_prompt_template = """Question: {QuestionText}
Answer: {MC_Answer}
Correct? {CorrectFlag}
Student Explanation: {StudentExplanation}
"""

dataset_chat = [
    [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": user_prompt_template.format(
                QuestionText=row["QuestionText"],
                MC_Answer=row["MC_Answer"],
                CorrectFlag=row["is_correct"],
                StudentExplanation=row["StudentExplanation"],
            )
        },
        {
            "role": "assistant",
            "content": row["special_label"]
        }
    ]
    for _, row in train.iterrows()
]

# Convert to tokenized dataset
def formatting_prompts_func(dataset):
    # This function is now simplified as we handle the chat template application directly.
    texts = [
        tokenizer.apply_chat_template(
            ex, tokenize=False, add_generation_prompt=False
        ) for ex in dataset
    ]
    return {"text": texts}

dataset_tokenized = formatting_prompts_func(dataset_chat)


dataset_tokenized = Dataset.from_dict(dataset_tokenized)


dataset_tokenized[0]


# Training Arguments with Unsloth optimizations
from unsloth import is_bfloat16_supported

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    # num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir=f"./{DIR}",
    # save_strategy="epoch",
    save_strategy="steps",
    save_steps=0.10,
    max_steps = 50,
    save_total_limit=5,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_tokenized,
    dataset_text_field="text",
    max_seq_length=MAX_LEN,
    dataset_num_proc=2,
    packing=False, # Important for classification tasks
    args=training_args,
)


from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)


# Train the model
trainer_stats = trainer.train()

