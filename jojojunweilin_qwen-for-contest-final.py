import sys
import os
package_dir = r"/kaggle/input/mypackges/packages"
package_dir2 = r"/kaggle/input/mypackage2/packages2"
for file in os.listdir(package_dir):
    if file.endswith((".whl", ".tar.gz")):  # 只安装 wheel 或 tar.gz
        file_path = os.path.join(package_dir, file)
        !pip install {file_path} --no-deps  # --no-deps 可防止网络安装依赖
for file in os.listdir(package_dir2):
    if file.endswith((".whl", ".tar.gz")):  # 只安装 wheel 或 tar.gz
        file_path = os.path.join(package_dir2, file)
        !pip install {file_path} --no-deps  # --no-deps 可防止网络安装依赖
!pip install {r"/kaggle/input/tokenizor/tokenizers-0.22.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"} --no-deps
import numpy as np
import re
from tqdm import tqdm
import torch
from datasets import Dataset
import pandas as pd
import re
import unicodedata
import json
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from modelscope import snapshot_download, AutoTokenizer
!pip uninstall -y wandb


# 可选：是否清理文本
CLEAN_TEXT = True

def cleaner(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1️⃣ 修复 Unicode 并转换为标准形式
    text = unicodedata.normalize('NFKC', text)

    # 2️⃣ 去掉 URL（更全面的匹配 http, https, www, 结尾可能带 / 或 ? 或 #
    # url_pattern = r'\b(?:https?://|www\.)\S+\b'
    # text = re.sub(url_pattern, '<URL>', text)

    # 3️⃣ 去掉邮箱
    text = re.sub(r'\b\S+@\S+\.\S+\b', '<EMAIL>', text)

    # 4️⃣ 去掉电话号码（含 + 号、-、空格、括号）
    phone_pattern = r'(\+?\d[\d\-\(\) ]{7,}\d)'
    text = re.sub(phone_pattern, '<PHONE>', text)

    # 5️⃣ 去掉多余空格
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# 测试
sample = "Check out my website: https://example.com or email me at test@example.com. Call +1-800-123-4567!"
print(cleaner(sample))


def build_dataframe(data_path):
    # 读取训练和测试数据
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")
    train_subset_1 = train_df[['body', 'rule', 'subreddit', 'rule_violation']]
    train_subset_2 = train_df[['positive_example_1', 'rule', 'subreddit']]
    train_subset_2['rule_violation'] = 1
    train_subset_3 = train_df[['positive_example_2', 'rule', 'subreddit']]
    train_subset_3['rule_violation'] = 1
    train_subset_4 = train_df[['negative_example_1', 'rule', 'subreddit']]
    train_subset_4['rule_violation'] = 0
    train_subset_5 = train_df[['negative_example_2', 'rule', 'subreddit']]
    train_subset_5['rule_violation'] = 0
    train_subset_6 = test_df[['positive_example_1', 'rule', 'subreddit']]
    train_subset_6['rule_violation'] = 1
    train_subset_7 = test_df[['positive_example_2', 'rule', 'subreddit']]
    train_subset_7['rule_violation'] = 1
    train_subset_8 = test_df[['negative_example_1', 'rule', 'subreddit']]
    train_subset_8['rule_violation'] = 0
    train_subset_8 = test_df[['negative_example_2', 'rule', 'subreddit']]
    train_subset_8['rule_violation'] = 0

    train_list = [train_subset_1, train_subset_2, train_subset_3, train_subset_4,
                  train_subset_5, train_subset_6, train_subset_7, train_subset_8]

    for index in range(len(train_list)):
        train_list[index].columns = ['body', 'rule', 'subreddit', 'rule_violation']
        if index != 0:
            train_subset_1 = pd.concat([train_subset_1, train_list[index]], axis=0)
    train_subset = train_subset_1.reset_index(drop=True)
    train_subset = train_subset.drop(train_subset_1[train_subset_1.duplicated()].index,axis=0)
    # 初始化列表
    all_rows = []

    # 清洗文本
    if CLEAN_TEXT:
        for column in train_subset.columns:
            if column != 'rule_violation':
                train_subset[column] = train_subset[column].apply(cleaner)

    return train_subset


PROMPT = """Given the subreddit, the rule, and examples of rule violations/non-violations, determine if the following comment violates the rule.Answer with only "1" if the comment violates the rule, otherwise answer with only "0".
Output must and ONLY is in this way "the answer is 0 or 1" . Do NOT explain. Do NOT repeat. """
MAX_LENGTH = 2048
tokenizer = AutoTokenizer.from_pretrained(r"/kaggle/input/qwen-pretrained2/output/qwen3-reddit", use_fast=False, trust_remote_code=True)

def dataset_jsonl_transfer(df:pd.DataFrame, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    for index,row in df.iterrows():
        prompt = f"""
        Input: 
        Subreddit: {row['subreddit']}
        Rule: {row['rule']}
        Comment: {row['body']}
        """

        input = prompt
        output = "the answer is " + str(row['rule_violation'])
        message = {
            "instruction": PROMPT,
            "input": f"{input}",
            "output": output,
        }
        messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")


def process_func(example):
    """
    将数据集进行预处理
    """
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=MAX_LENGTH,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def transfer_word_to_num(word):
    import re
    the_num = int(re.findall(r"he answer is\s*([01])\b", word)[0])
    return the_num


os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # 启用同步模式以定位错误
test_jsonl_new_path = "val_format.jsonl"
# 加载模型
model_path = "/kaggle/input/qwen-pretrained2/output/qwen3-reddit"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")


import pandas as pd
data_path = r"/kaggle/input/jigsaw-agile-community-rules"
test_df = pd.read_csv(f"{data_path}/test.csv")
train_subset_6 = test_df[['positive_example_1', 'rule', 'subreddit']]
train_subset_6['rule_violation'] = 1
train_subset_7 = test_df[['positive_example_2', 'rule', 'subreddit']]
train_subset_7['rule_violation'] = 1
train_subset_8 = test_df[['negative_example_1', 'rule', 'subreddit']]
train_subset_8['rule_violation'] = 0
train_subset_9 = test_df[['negative_example_2', 'rule', 'subreddit']]
train_subset_9['rule_violation'] = 0
train_list = [train_subset_6, train_subset_7, train_subset_8,train_subset_9]

for index in range(len(train_list)):
    train_list[index].columns = ['body', 'rule', 'subreddit', 'rule_violation']
    if index != 0:
        train_subset_6 = pd.concat([train_subset_6, train_list[index]], axis=0)
train_subset = train_subset_6.reset_index(drop=True)
train_subset = train_subset.drop(train_subset_6[train_subset_6.duplicated()].index,axis=0)


# 初始化列表
all_rows = []

# 清洗文本
for column in train_subset.columns:
    if column != 'rule_violation':
        train_subset[column] = train_subset[column].apply(cleaner)


PROMPT = """Given the subreddit, the rule, and examples of rule violations/non-violations, determine if the following comment violates the rule.Answer with only "1" if the comment violates the rule, otherwise answer with only "0".
Output must and ONLY is in this way "the answer is 0 or 1" . Do NOT explain. Do NOT repeat. """
MAX_LENGTH = 2048


train_df = train_subset
train_jsonl_new_path = "train_format.jsonl"
dataset_jsonl_transfer(train_df, train_jsonl_new_path)


# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pandas as pd
# import numpy as np

# # 加载模型和 tokenizer
# model_path = "/kaggle/input/qwen-pretrained2/output/qwen3-reddit"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
# model.eval()

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.pad_token_id

# # 测试样本
# test_df = pd.read_json("/kaggle/input/testing/val_format.jsonl", lines=True)

# # 最终结果列表
# predictions = []
# real = []
# for idx, row in test_df.iterrows():
#     instruction = row['instruction']
#     input_value = row['input']

#     # 构建 prompt
#     prompt = f"{instruction}\n{input_value}\nAnswer:"

#     # 编码
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     input_ids = inputs["input_ids"]

#     # 前向计算 logits
#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits  # [1, seq_len, vocab_size]

#     # 取最后一个 token 的 logits
#     last_token_logits = logits[0, -1, :]
#     probs = torch.softmax(last_token_logits, dim=-1)

#     # 关心的 token
#     yes_id = tokenizer.convert_tokens_to_ids("1")
#     no_id = tokenizer.convert_tokens_to_ids("0")

#     # 计算概率
#     yes_prob = probs[yes_id].item()
#     no_prob = probs[no_id].item()

#     # 输出 1 或 0
#     predictions.append(yes_prob)
#     real.append(float(row['output'][-1]))

# print(predictions)
# print(real)
# from sklearn.metrics import roc_auc_score
# # predictions 是模型预测概率
# # real 是真实标签（0 或 1）
# auc = roc_auc_score(real, predictions)
# print("AUC:", auc)



#---------------------基于全局--------------
# import os
# os.environ["SWANLAB_MODE"] = "disabled"   # 强制关闭 SwanLab
# os.environ["SWANLAB_DISABLED"] = "true"   # 冗余保险

# import swanlab
# swanlab.SwanLabEnv.check = lambda *args, **kwargs: None
# args = TrainingArguments(
#     output_dir="/root/autodl-tmp/output/Qwen3-0.6B",
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4,
#     logging_steps=10,
#     num_train_epochs=2,
#     save_steps=400,
#     learning_rate=1e-4,
#     save_on_each_node=True,
#     gradient_checkpointing=True,
#     report_to=None,
#     run_name="qwen3-0.6B",
# )
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
# )

# trainer.train()
# trainer.save_model("./output/qwen3-reddit-new")
# tokenizer.save_pretrained("./output/qwen3-reddit-new")



#---------------基于Lora---------------

import pandas as pd
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers.utils import is_torch_bf16_gpu_available
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    num_train_epochs=5,
    
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    
    optim="paged_adamw_8bit",
    learning_rate=1e-4, #keep high, lora usually likes high. 
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    bf16=is_torch_bf16_gpu_available(),
    fp16=not is_torch_bf16_gpu_available(),
    dataloader_pin_memory=True,
    
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    save_strategy="no",
    report_to="none",

    completion_only_loss=True,
    packing=False,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    r"/kaggle/input/qwen-pretrained2/output/qwen3-reddit",
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
)

trainer.save_model("./output/qwen3-reddit-lora")
tokenizer.save_pretrained("./output/qwen3-reddit-lora")


test_df = pd.read_csv(f"{data_path}/test.csv")
test_subset = test_df[['body', 'rule', 'subreddit']]
messages = []
new_path = "test_format.jsonl"
for index,row in test_df.iterrows():
    prompt = f"""
    Input: 
    Subreddit: {row['subreddit']}
    Rule: {row['rule']}
    Comment: {row['body']}
    """

    input = prompt
    message = {
        "instruction": PROMPT,
        "input": f"{input}",
    }
    messages.append(message)

# 保存重构后的JSONL文件
with open(new_path, "w", encoding="utf-8") as file:
    for message in messages:
        file.write(json.dumps(message, ensure_ascii=False) + "\n")


# # -----全参数预测结果------

# new_model_path = "./output/qwen3-reddit-new"
# new_tokenizer = AutoTokenizer.from_pretrained(new_model_path, trust_remote_code=True,padding=True,
# truncation=True)
# new_model = AutoModelForCausalLM.from_pretrained(new_model_path, torch_dtype=torch.bfloat16, device_map="auto")

# # 用测试集的前3条，主观看模型
# test_df_clean = pd.read_json(new_path, lines=True)
# test_text_list = []
# output_text_list = []
# for index, row in tqdm(test_df_clean.iterrows()):
#     instruction = row['instruction']
#     input_value = row['input']

#     # 构建 prompt
#     prompt = f"{instruction}\n{input_value}\nAnswer:"

#     # 编码
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     input_ids = inputs["input_ids"]

#     # 前向计算 logits
#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits  # [1, seq_len, vocab_size]

#     # 取最后一个 token 的 logits
#     last_token_logits = logits[0, -1, :]
#     probs = torch.softmax(last_token_logits, dim=-1)

#     # 关心的 token
#     yes_id = tokenizer.convert_tokens_to_ids("1")
#     no_id = tokenizer.convert_tokens_to_ids("0")

#     # 计算概率
#     yes_prob = probs[yes_id].item()
#     no_prob = probs[no_id].item()

#     # 输出 1 或 0
#     predictions.append(yes_prob)
# print(predictions)


# # 用测试集的前3条，主观看模型
# test_df_clean = pd.read_json(new_path, lines=True)
# test_text_list = []
# output_text_list = []
# for index, row in tqdm(test_df_clean.iterrows()):
#     instruction = row['instruction']
#     input_value = row['input']



#     messages = [
#         {"role": "system", "content": f"{instruction}"},
#         {"role": "user", "content": f"{input_value}"}
#     ]


#     response = predict(messages, model, tokenizer)

#     response_text = f"""
#     Question: {input_value}

#     LLM:{response}
#     """
#     test_text_list.append(response)
# print(test_text_list)


# ---------------rola预测结果

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
from sklearn.metrics import roc_auc_score

# ------------------ 模型和 LoRA 权重路径 ------------------
base_model_path = "/kaggle/input/qwen-pretrained2/output/qwen3-reddit"  # 原始基础模型
lora_model_path = "./output/qwen3-reddit-lora"  # 已训练好的 LoRA 模型

# ------------------ 加载 tokenizer ------------------
tokenizer = AutoTokenizer.from_pretrained(lora_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------ 加载模型 + LoRA 权重 ------------------
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)
model = PeftModel.from_pretrained(base_model, lora_model_path, device_map="auto")
model.eval()  # 推理模式

# ------------------ 测试集 ------------------
test_df = pd.read_json(new_path, lines=True)  # 你的测试文件
predictions = []
real = []

for idx, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    
    # 构建 prompt
    prompt = f"{instruction}\n{input_value}\nAnswer:"
    
    # 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # 前向计算 logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 取最后一个 token 的 logits
    last_token_logits = logits[0, -1, :]
    probs = torch.softmax(last_token_logits, dim=-1)
    
    # 关心的 token，例如 "1" 和 "0"
    yes_id = tokenizer.convert_tokens_to_ids("1")
    no_id = tokenizer.convert_tokens_to_ids("0")
    
    yes_prob = probs[yes_id].item()
    no_prob = probs[no_id].item()
    
    predictions.append(yes_prob)

# ------------------ 输出结果 ------------------
print("预测概率:", predictions)



test_df_orin = pd.read_csv(f"{data_path}/test.csv")
test_df_orin['rule_violation'] = predictions
test_df_orin[['row_id','rule_violation']]


test_df_orin[['row_id','rule_violation']].to_csv("submission.csv", index=False)




