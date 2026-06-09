pip install /kaggle/input/peft-n-friends/huggingface_hub-0.27.0-py3-none-any.whl


!pip install /kaggle/input/hf-libraries/peft/peft-0.14.0-py3-none-any.whl


!pip install   accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files


pip install /kaggle/input/hf-libraries/tokenizers/tokenizers-0.21.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


pip install /kaggle/input/hf-libraries/transformers/transformers-4.47.1-py3-none-any.whl


# !pip install transformers accelerate bitsandbytes \
#     -U --no-index --find-links /kaggle/input/lmsys-wheel-files


import transformers
transformers.__version__


import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import sklearn
import numpy as np
import pandas as pd
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel


import peft


peft.__version__


assert torch.cuda.device_count() == 2


@dataclass
class Config:
    # gemma_dir = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it-4bit/1/gemma-2-9b-it-4bit'
    gemma_dir = '/kaggle/input/unsloth-gemma-2-9b-it-bnb-4bit/unsloth_gemma-2-9b-it-bnb-4bit'
    
    lora_dir = '/kaggle/input/base-version-2048/checkpoint-6055'
    # lora_dir = '/kaggle/input/fix-wsdm-lmsys/checkpoint-6055_fix'
    
    max_length = 1800
    batch_size = 4
    device = torch.device("cuda")    
    tta = True  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input

cfg = Config()


test = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')


def process_text(text: str) -> str:
    return " ".join(eval(text, {"null": ""}))

def process_text(text: str) -> str:
    return text


test.loc[:, 'prompt'] = test['prompt'].apply(process_text)
test.loc[:, 'response_a'] = test['response_a'].apply(process_text)
test.loc[:, 'response_b'] = test['response_b'].apply(process_text)

display(test.head(5))


def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    if spread_max_length:
        prompt = tokenizer(prompt, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_a = tokenizer(response_a, max_length=max_length//3, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length//3, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        attention_mask = [[1]* len(i) for i in input_ids]
    else:
        text = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
        tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
    return input_ids, attention_mask


%%time

tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"

data = pd.DataFrame()
data["id"] = test["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test["prompt"], test["response_a"], test["response_b"])
data["length"] = data["input_ids"].apply(len)

aug_data = pd.DataFrame()
aug_data["id"] = test["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test["prompt"], test["response_b"], test["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)


# Load base model on GPU 0
device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_0,
    use_cache=False,
)

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_1,
    use_cache=False,
)
model_0 = PeftModel.from_pretrained(model_0, cfg.lora_dir)
model_1 = PeftModel.from_pretrained(model_1, cfg.lora_dir)


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win = [], []
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        tmp = df.iloc[start_idx:end_idx]
        input_ids = tmp["input_ids"].to_list()
        attention_mask = tmp["attention_mask"].to_list()
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        outputs = model(**inputs.to(device))
        proba = outputs.logits.softmax(-1).cpu()
        
        a_win.extend(proba[:, 0].tolist())
        b_win.extend(proba[:, 1].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    
    return df


st = time.time()

# sort by input length to fully leverage dynaminc padding
data = data.sort_values("length", ascending=False)


data


# 确保 sub_1 和 sub_2 中的总 token 数大致相同
sub_1 = data.iloc[0::2].copy()  # 从数据中每隔一行取一个，复制到 sub_1
sub_2 = data.iloc[1::2].copy()  # 从数据中每隔一行取一个，复制到 sub_2

# 使用线程池并行处理 sub_1 和 sub_2
with ThreadPoolExecutor(max_workers=2) as executor:
    # 将 inference 函数应用于 sub_1 和 sub_2，分别使用 model_0 和 model_1 以及 device_0 和 device_1
    results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

# 将并行处理的结果合并为一个 DataFrame
result_df = pd.concat(list(results), axis=0)

# 提取特定列的概率值
proba = result_df[["winner_model_a", "winner_model_b"]].values

# 打印程序运行的总耗时
print(f"elapsed time: {time.time() - st}")



st = time.time()

if cfg.tta:
    data = aug_data.sort_values("length", ascending=False)  # sort by input length to boost speed
    sub_1 = data.iloc[0::2].copy()
    sub_2 = data.iloc[1::2].copy()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

    tta_result_df = pd.concat(list(results), axis=0)
    # recall TTA's order is flipped
    tta_proba = tta_result_df[["winner_model_b", "winner_model_a"]].values 
    # average original result and TTA result.
    proba = (proba + tta_proba) / 2

print(f"elapsed time: {time.time() - st}")


result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
submission_df = result_df[["id", 'winner_model_a', 'winner_model_b']]

# 假设 result_df 是您的 DataFrame
submission_df['winner'] = submission_df.apply(
    lambda row: 'model_a' if row['winner_model_a'] > 0.5 
                else ('model_b' if row['winner_model_b'] > 0.5 else 'tie'),
    axis=1
)

selected_df = submission_df[['id', 'winner']]

selected_df.to_csv('submission.csv', index=False)
display(submission_df)

display(selected_df)

