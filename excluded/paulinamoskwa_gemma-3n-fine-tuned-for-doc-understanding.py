%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install --no-deps --upgrade transformers
!pip install --no-deps --upgrade timm
!pip install transformers -U


BASE_PATH = "/kaggle/input/high-quality-invoice-images-for-ocr"


import csv
import json
from pathlib import Path

def load_csv(csv_path: str) -> list[dict]:
    data = []
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            invoice = json.loads(row["Json Data"])["invoice"]
            for key in ("client_address", "seller_address"):
                invoice[key] = invoice[key].replace("\n", ", ")
            invoice.pop("due_date", None)
            invoice["img_name"] = row["File Name"]
            data.append(invoice)
    return data

# Load datasets
train_data = load_csv(f"{BASE_PATH}/batch_1/batch_1/batch1_1.csv")
test_data  = load_csv(f"{BASE_PATH}/batch_1/batch_1/batch1_2.csv")


train_data[0]


len(train_data)


from PIL import Image

instruction = "Extract all information from this invoice image and return it in JSON format with the following fields: client_name, client_address, seller_name, seller_address, invoice_number, and invoice_date."

def convert_data_to_conversation(sample: dict) -> dict:
    img_path = f"{BASE_PATH}/batch_1/batch_1/batch1_1/{sample['img_name']}"
    img = Image.open(img_path).convert("RGB")
    response_data = {
        "client_name":    sample["client_name"],
        "client_address": sample["client_address"],
        "seller_name":    sample["seller_name"],
        "seller_address": sample["seller_address"],
        "invoice_number": sample["invoice_number"],
        "invoice_date":   sample["invoice_date"]
    }
    response_text = json.dumps(response_data, indent=2)
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": img}]},
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
    ]
    return {"messages": conversation}

converted_dataset = []
for sample in train_data:
    converted_sample = convert_data_to_conversation(sample)
    converted_dataset.append(converted_sample)


converted_dataset[0]


len(converted_dataset)


from unsloth import FastVisionModel, get_chat_template
import torch

model, processor = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    dtype = None,
    max_seq_length = 2048,
    load_in_4bit = True,
    full_finetuning = False,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 32,
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    target_modules = "all-linear",
)


import re

def run_inference(img):
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": img}]}]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(img, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
    eos_id = processor.tokenizer.eos_token_id
    end_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    result = model.generate(
        **inputs,
        max_new_tokens=256,
        eos_token_id=[eos_id, end_turn_id],
        pad_token_id=eos_id,
        temperature=0.7,
        top_p=0.9,
        use_cache=True
    )
    decoded = processor.batch_decode(result, skip_special_tokens=True)[0]
    assistant_answer = decoded.split("model")[-1].strip()

    # Return a JSON if possible, otherwise the raw answer
    json_match = re.search(r"\{.*?\}", assistant_answer, re.S)
    if json_match:
        return json.loads(json_match.group(0))
    return assistant_answer


from tqdm import tqdm
import yaml 

def get_diff_in_dict(a, b, name_a, name_b):
    diff = {}
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            diff[k] = f"{name_a}: {a.get(k)}, {name_b}: {b.get(k)}"
    return diff

TEST_BASE_PATH = f"{BASE_PATH}/batch_1/batch_1/batch1_2"

perfect_pred = 0
wrong_pred = 0

for elem in tqdm(test_data[0:10]):
    img = Image.open(f"{TEST_BASE_PATH}/{elem['img_name']}")
    res = run_inference(img)
    if type(res) is dict:
        res['img_name'] = elem['img_name']
        if res == elem:
            print("Perfect")
            perfect_pred += 1
        else:
            print("Wrong JSON")
            diff = get_diff_in_dict(res, elem, "pred", "gt")
            print(yaml.dump(diff, sort_keys=False, allow_unicode=True))
            wrong_pred += 1
    else: 
        print(res)
        print('---')
        print(elem)
        print("Wrong type")
        wrong_pred += 1
    print(f"Perfect pred: {perfect_pred}/{perfect_pred + wrong_pred}")


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_grad_norm = 0.3,
        warmup_ratio = 0.03,
        # max_steps = 100,   # alternative to num_train_epochs
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 10,
        save_strategy="steps",
        save_steps=50,
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "invoice_ocr_outputs",
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    )
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")


FastVisionModel.for_inference(model)

perfect_pred = 0
wrong_pred = 0

for elem in tqdm(test_data[0:10]):
    img = Image.open(f"{TEST_BASE_PATH}/{elem['img_name']}")
    res = run_inference(img)
    if type(res) is dict:
        res['img_name'] = elem['img_name']
        if res == elem:
            print("Perfect")
            perfect_pred += 1
        else:
            print("Wrong JSON")
            diff = get_diff_in_dict(res, elem, "pred", "gt")
            print(yaml.dump(diff, sort_keys=False, allow_unicode=True))
            wrong_pred += 1
    else: 
        print(res)
        print('---')
        print(elem)
        print("Wrong type")
        wrong_pred += 1
    print(f"Perfect pred: {perfect_pred}/{perfect_pred + wrong_pred}")







