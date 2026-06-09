!sudo apt install python3-dev cmake libcurl4-openssl-dev


!pip install unsloth


from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = 2048,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)


from datasets import load_dataset, Dataset
from unsloth import to_sharegpt

# データセットを読み込む（これは DatasetDict を返す）
rawdataset = load_dataset("tatsu-lab/alpaca")

# "train" スプリットのみを取得（これは datasets.Dataset 型）
train_dataset = rawdataset["train"]

# to_sharegpt に変換
share_dataset = to_sharegpt(
    train_dataset,
    merged_prompt="{instruction}",
    merged_column_name="instruction",
    output_column_name="output",
    conversation_extension=3
)



import re

converted_share_dataset = [
    [
        {
            "role": "user" if message["from"] == "human" else "assistant",
            "content": re.sub(r"\('(.+?)',\)", r"\1", message["value"])
        }
        for message in item["conversations"]
    ]
    for item in share_dataset
]

conversations = tokenizer.apply_chat_template(
    converted_share_dataset,
    tokenize=False
)

targetdataset = Dataset.from_dict({"text": conversations})


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=targetdataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

trainer_stats = trainer.train()


messages = [
    {"role": "user", "content": "こんばんは"},
    {"role": "assistant", "content": "こんばんは"},
    {"role": "user", "content": "名前を教えて"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)


