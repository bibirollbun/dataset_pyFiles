!pip install transformers accelerate datasets peft trl bitsandbytes --quiet


pip install -U bitsandbytes


from peft import LoraConfig

lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    #Use fp16 for not supported bf16 hardware
    #bnb_4bit_compute_dtype=torch.float16
)


from transformers import AutoTokenizer, AutoModelForCausalLM

#If you add the model from Kaggle, use this line.
modelName = "/kaggle/input/gemma-2/transformers/gemma-2-2b/2"

tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForCausalLM.from_pretrained(modelName, 
                                             quantization_config=bnb_config, 
                                             device_map="auto")


from datasets import load_dataset
dataset = load_dataset("projecte-aina/catalanqa", split="train")
dataset, dataset[0]


gemma_prompt = """<start_of_turn>user
{}: {}<end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""
gemma_prompt


eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
tokenizer.padding_side = "right"

eos_token, pad_token


def formatting_prompts_func(examples):
    instructions = examples["question"]
    inputs       = examples["context"]
    outputs      = examples["answers"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = gemma_prompt.format(instruction, input, output[0]['text']) + eos_token
        texts.append(text)
    return { "text" : texts, }
pass


dataset = dataset.map(formatting_prompts_func, batched = True)
dataset


print(dataset["text"][2])


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

print("Tokenizing dataset...")
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("Dataset tokenized:", dataset[0])


from transformers import TrainingArguments

train_args = TrainingArguments(
    per_device_train_batch_size=1,  # Each GPU processes 4 examples per step.
    gradient_accumulation_steps=2,  # Gradients are accumulated over 4 steps before updating weights.
    warmup_steps=30,  # Learning rate warms up (gradually increases) for the first 30 steps.
    max_steps=5000,  # Total number of optimization steps for training.
    gradient_checkpointing=True,  # Saves memory by recomputing activations during backpropagation.
    learning_rate=1e-4,  # Base learning rate for the optimizer.
    fp16=True,  # FP16 precision is disabled (not used).
    bf16=False,  # Enables bfloat16 precision, optimized for RTX 4090 GPUs.
    logging_steps=125,  # Logs training metrics every 125 steps.
    optim="adamw_8bit",  # Uses AdamW optimizer with 8-bit precision for optimizer states to save memory.
    weight_decay=0.01,  # Regularization to prevent overfitting by penalizing large weights.
    lr_scheduler_type="linear",  # Linearly decays learning rate after the warmup period.
    output_dir="outputs",  # Directory where model checkpoints and logs will be saved.
    report_to="none",  # Disables logging to external tools like TensorBoard or WandB.
)


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer

# Define a data collator
# Since we tokenized our dataset and returned them as Torch tensors, you may not need it.
# If you did not tokenized the dataset, you must use Data Collator. 
#It uses tokenizer, tokenize your training data and returns them as tensors.
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="longest",
    return_tensors="pt"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    peft_config=lora_config,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()


# Save the model locally
trainer.save_model("gemma-2-2b-cat-5000step")


# Zip the model to download it:
import shutil
shutil.make_archive("gemma-2-2b-cat-5000step", 'zip', "gemma-2-2b-cat-5000step")


#upload model to kaggle
import kagglehub
from kagglehub.config import get_kaggle_credentials

kagglehub.login()


kagglehub.model_upload(f'kennethalonso/gemma-2-2b-cat/pyTorch/5000steps', 'gemma-2-2b-cat-2500step', 'Apache 2.0')


from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
base_model_name = "/kaggle/input/gemma-2/transformers/gemma-2-2b/2"
fine_tuned_model_name = "/kaggle/input/gemma-2-2b-cat/pytorch/5000steps/1"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, fine_tuned_model_name).to('cuda')




question = "A quants anys de presó ha estat condemnat Oriol Pujol?"

inputs = tokenizer(question, return_tensors="pt").to('cuda')

generated_ids_bm = base_model.generate(**inputs,
                              max_new_tokens=128,
                              do_sample=True,
                              temperature=1.0,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

generated_ids_ftm = model.generate(**inputs,
                              max_new_tokens=128,
                              do_sample=True,
                              temperature=1.0,
                              top_p=0.95,
                              top_k=50,
                              repetition_penalty=1.0)

print("Base Model")
print("----------")
print(tokenizer.batch_decode(generated_ids_bm, skip_special_tokens=True)[0])
print("Finetuned Model")
print("---------------")
print(tokenizer.batch_decode(generated_ids_ftm, skip_special_tokens=True)[0])


question = "Quants brigadistes van lluitar a la Guerra Civil?"

inputs = tokenizer(question, return_tensors="pt").to('cuda')

generated_ids_bm = base_model.generate(**inputs,
                              max_new_tokens=512,
                              do_sample=True,
                              temperature=1.0,
                              top_p=0.98,
                              top_k=50,
                              repetition_penalty=1.0)

generated_ids_ftm = model.generate(**inputs,
                              max_new_tokens=512,
                              do_sample=True,
                              temperature=1.0,
                              top_p=0.98,
                              top_k=50,
                              repetition_penalty=1.0)

print("Base Model")
print("----------")
print(tokenizer.batch_decode(generated_ids_bm, skip_special_tokens=True)[0])

print("Finetuned Model")
print("---------------")
print(tokenizer.batch_decode(generated_ids_ftm, skip_special_tokens=True)[0])




