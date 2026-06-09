%%capture
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth


import re
import torch
import pandas as pd
from datasets import load_dataset
from trl import SFTTrainer
from tqdm.auto import tqdm
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None # None for auto detection.
load_in_4bit = True # 4bit quantization to reduce memory usage. 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    #token = "" HF_Token for gated models
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)


label_map = {0: "Algebra",
            1: "Geometry and Trigonometry",
            2: "Calculus and Analysis",
            3: "Probability and Statistics",
            4: "Number Theory",
            5: "Combinatorics and Discrete Math",
            6: "Linear Algebra",
            7: "Abstract Algebra and Topology"}


train_actual = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
aug_train = pd.read_csv("/kaggle/input/math-problem-classification-data/train_augmented.csv")
train = pd.concat([train_actual[:9189], aug_train]).reset_index(drop=True)
train["instruction"] = "Classify this math problem into one of these eight topics: Algebra, Geometry and Trigonometry, Calculus and Analysis, Probability and Statistics, Number Theory, Combinatorics and Discrete Math, Linear Algebra, Abstract Algebra and Topology."
train["label"] = train["label"].map(label_map)
train = train.rename(columns={"label": "output", "Question": "input"})
train.to_csv("train_updated.csv", index=False)


dataset = load_dataset("csv", data_files="train_updated.csv", split="train")


prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched = True,)


dataset, dataset[0]


print(dataset[0]["text"])


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 640,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


train_actual = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

train_actual["instruction"] = "Classify this math problem into one of these eight topics: Algebra, Geometry and Trigonometry, Calculus and Analysis, Probability and Statistics, Number Theory, Combinatorics and Discrete Math, Linear Algebra, Abstract Algebra and Topology."
train_actual.rename(columns = {"Question": "input", "label": "output"}, inplace=True)


raw_outputs = []
for i in tqdm(range(9189,len(train_actual))):
  inputs = tokenizer(
  [
      prompt.format(
          train_actual.iloc[0]["instruction"], # instruction
          train_actual.iloc[i]["input"], # input
          "", # output - blank for generation!
      )
  ], return_tensors = "pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
  raw_outputs.append(tokenizer.batch_decode(outputs))


print(raw_outputs[0][0], train_actual.iloc[9189]["output"])


test_set = train_actual[9189:]
test_set["raw_outputs"] = [raw_output[0] for raw_output in raw_outputs]
test_set


def parse_output(output):
    re_match = re.search(r'### Response:\n(.*?)<\|end_of_text\|>', output, re.DOTALL)
    if re_match:
        response = re_match.group(1).strip()
        return response
    else:
        return ''

test_set["parsed_outputs"] = test_set["raw_outputs"].apply(parse_output)
test_set


test_set["output"] = test_set["output"].map(label_map)
test_set


print("Accuracy: ", sum(test_set["output"] == test_set["parsed_outputs"]))


model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")


if False:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )


public_set = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
public_set


FastLanguageModel.for_inference(model)

public_set["instruction"] = "Classify this math problem into one of these eight topics: Algebra, Geometry and Trigonometry, Calculus and Analysis, Probability and Statistics, Number Theory, Combinatorics and Discrete Math, Linear Algebra, Abstract Algebra and Topology."
public_set.rename(columns = {"Question": "input"}, inplace=True)

raw_outputs = []
for i in tqdm(range(len(public_set))):
  inputs = tokenizer(
  [
      prompt.format(
          public_set.iloc[0]["instruction"], 
          public_set.iloc[i]["input"], 
          "",
      )
  ], return_tensors = "pt").to("cuda")

  outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
  raw_outputs.append(tokenizer.batch_decode(outputs))


public_set["raw_outputs"] = [raw_output[0] for raw_output in raw_outputs]
public_set


public_set["parsed_outputs"] = public_set["raw_outputs"].apply(parse_output)
public_set


label2id = {v:k for k,v in label_map.items()}
label2id


public_set["label"] = public_set["parsed_outputs"].map(label2id)
public_set


public_set["label"] = public_set["label"].fillna(0).astype(int)
public_set.rename(columns = {"input": "Question"})
public_set


public_set.rename(columns = {"input": "Question"}, inplace=True)
public_set[["id", "label"]].to_csv("submission.csv", index=False)
pd.read_csv("submission.csv")




