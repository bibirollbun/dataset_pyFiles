%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


import os
import json
import random
from PIL import Image
from pathlib import Path

#train_path = f"/kaggle/input/checkpoint-{checkpointN}/train.json"
#valid_path = f"/kaggle/input/checkpoint-{checkpointN}/valid.json"
#test_path = f"/kaggle/input/checkpoint-{checkpointN}/test.json"

#'''
# 클래스별 영어 안내 메시지
CLASS_MAP = {
    "0": "red pedestrian signal",
    "1": "green pedestrian signal",
    "2": "countdown green pedestrian signal",
    "3": "countdown blank pedestrian signal",
    "4": "no pedestrian signal"
}

def generate_alert(cls):
    if cls == "4":
        return None

    suggestion = ""
    if cls == "0":
        suggestion = "Please wait."
    elif cls == "1":
        suggestion = "You can go."
    else:
        suggestion = "You should wait for the next green signal."
        
        
    return f"There is a {CLASS_MAP[cls]}. {suggestion}"

def handle_missing():
    return "No pedestrian signal detected. Proceed with caution."

# 변환 함수
def process_split(annotation_csv, image_dir, output_json):
    data = []
    lines = open(annotation_csv, "r").read().splitlines()
    
    for line in lines:
        fname, cls, x1, y1, x2, y2, blocked = line.strip().split(",")
            
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue

        alert = generate_alert(cls)

        # Q&A 모드 예시 추가
        data.append({
            "image": f"{img_path}",
            "question": "Can I cross the crosswalk now?",
            "answer": alert if alert else handle_missing()
        })

    random.shuffle(data)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{annotation_csv} → {output_json} 변환 완료. 총 {len(data)} 샘플.")

image_dir = "/kaggle/input/ptl-dataset/PTL_Dataset/Images/"

train_annotation = "/kaggle/input/ptl-dataset/PTL_Dataset/training_file.csv"
train_path = "train.json"

valid_annotation = "/kaggle/input/ptl-dataset/PTL_Dataset/validation_file.csv"
valid_path = "valid.json"

test_annotation = "/kaggle/input/ptl-dataset/PTL_Dataset/testing_file.csv"
test_path = "test.json"

process_split(train_annotation, image_dir, train_path)
process_split(valid_annotation, image_dir, valid_path)
process_split(test_annotation, image_dir, test_path)
#'''


from datasets import load_dataset
dataset = load_dataset(
    'json',
    data_files={
        'train': f"{train_path}",
        'valid': f"{valid_path}",
        'test': f"{test_path}",
    }
)


def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["question"]},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]
    return {"messages": conversation}
pass


converted_dataset = [convert_to_conversation(sample) for sample in dataset["train"]]
converted_dataset_val = [convert_to_conversation(sample) for sample in dataset["valid"]]
converted_dataset_tes = [convert_to_conversation(sample) for sample in dataset["test"]]


#test_img = dataset["test"][0]["image"]
test_img = "/kaggle/input/ptl-dataset/PTL_Dataset/Images/heon_IMG_0917.JPG"


from PIL import Image

Image.open(test_img).convert('RGB')


FastVisionModel.for_inference(model)  # Enable for inference!

image = Image.open(test_img).convert('RGB')
instruction = "Can I cross the crosswalk now?"

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor, skip_prompt=True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    eval_dataset=converted_dataset_val,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,

        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
        warmup_ratio = 0.03,
        max_steps = 800,
        #num_train_epochs = 2,          # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 100,
        eval_strategy="steps", # 검증 스텝마다 평가
        eval_steps=100, # 검증 스텝 설정 (save_steps와 동일하게 설정하는 것이 일반적)
        save_strategy="steps",
        save_steps=400, # 몇 스텝마다 모델 저장
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",             # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    )
)


isError = False


try:
    trainer_stats = trainer.train()
except:
    isError = True
    pass


model.gradient_checkpointing_disable()
model.config.use_cache = False


if isError:
    trainer_stats = trainer.train()


Image.open(test_img).convert('RGB')


FastVisionModel.for_inference(model)  # Enable for inference!

image = Image.open(test_img).convert('RGB')
instruction = "Can I cross the crosswalk now?"

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(processor, skip_prompt=True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache=True, temperature = 1.0, top_p = 0.95, top_k = 64)


model.save_pretrained_merged("unsloth_finetune", processor,)

