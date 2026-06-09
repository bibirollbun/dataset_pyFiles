! python -m pip install --no-index --find-links=/kaggle/input/make-wheel-v1 -r /kaggle/input/make-wheel-v1/requirements.txt


import os
DEBUG = False
USE_QLORA=True
PROB = 0.2


import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import torch
import sklearn
import polars as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel


@dataclass
class Config:
    gemma_dir: str = "/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2"         #  "/opt/meituan/dolphinfs_xiachuankun/huggingface.co/google/gemma-2-9b-it"
    lora_dir: str = "/kaggle/input/20250115-lorav6/LoRA-v6"
    max_length: int = 2048
    batch_size_list = [2, 16]
    tta = False
    fold_idx = 0
    
    
cfg = Config()


tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"


# 导入数据
data = pd.read_csv("/kaggle/input/20240102-data1/train_split.csv")

if DEBUG:
    data = data.head(1000)

print(data.shape)

data = data.dropna(subset=["winner"]) 
print(data.shape)

# 验证集
test_df = data.loc[data['fold']==cfg.fold_idx].reset_index(drop=True)
print(test_df.shape)


for col in ['prompt', 'response_a', 'response_b']:
    test_df[col] = test_df[col].fillna('')
    text_list = []
    if col == 'prompt':
        max_no = 402
        s_no = 200
        e_no = -201
    else:
        max_no = 702
        s_no = 350
        e_no = -351
    for text in tqdm(test_df[col]):
        encoded = tokenizer(text, return_offsets_mapping=True)
        if len(encoded['input_ids']) > max_no:
            start_idx, end_idx = encoded['offset_mapping'][s_no]
            new_text = text[:end_idx]
            start_idx, end_idx = encoded['offset_mapping'][e_no]
            new_text = new_text + "\n(snip)\n" + text[start_idx:]
            text = new_text
        text_list.append(text)
    test_df[col] = text_list


def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length):
    prompt = ["<prompt>: " + t for t in prompt]
    response_a = ["\n\n<response_a>: " + t for t in response_a]
    response_b = ["\n\n<response_b>: " + t for t in response_b]
    texts = [p + r_a + r_b for p, r_a, r_b in zip(prompt, response_a, response_b)]
    tokenized = tokenizer(texts, max_length=max_length, truncation=True)
    return tokenized['input_ids'], tokenized['attention_mask']


%%time

data = pd.DataFrame()
data["id"] = test_df["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test_df["prompt"], test_df["response_a"], test_df["response_b"])
data["length"] = data["input_ids"].apply(len)
data['index'] = np.arange(len(data), dtype=np.int32)

# 翻转
aug_data = pd.DataFrame()
aug_data["id"] = test_df["id"]

aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test_df["prompt"], test_df["response_b"], test_df["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)
aug_data['index'] = np.arange(len(aug_data), dtype=np.int32)


qlora = {}
if USE_QLORA:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", #nf4 or fp4
        bnb_4bit_use_double_quant = False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules = ["score"]
    )
    qlora['quantization_config'] = bnb_config
    print("Using QLoRA")


# Load base model on GPU 0
device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_0,
    use_cache=False,
    num_labels=2,
    **qlora
)

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_1,
    use_cache=False,
    num_labels=2,
    **qlora
)


# Get peft
model_0 = PeftModel.from_pretrained(model_0, model_id=cfg.lora_dir).to(device_0) 
model_0.eval()

model_1 = PeftModel.from_pretrained(model_1, model_id=cfg.lora_dir).to(device_1)
model_1.eval()


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size, max_length=cfg.max_length):
    winners = []
    
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
        
        winners.extend(proba[:, 1].tolist())
    
    df['winner'] = winners
    return df


def fenbushi_predict(data):
    data_dict = {}
    data_dict[0] = data[data["length"] > 1024].reset_index(drop=True)
    data_dict[1] = data[data["length"] <= 1024].reset_index(drop=True)
    print(data_dict[0].shape)
    print(data_dict[1].shape)
    result_df = []
    for i, batch_size in enumerate(Config.batch_size_list):
        if len(data_dict[i]) == 0:
            continue
        sub_1 = data_dict[i].iloc[0::2].copy()
        sub_2 = data_dict[i].iloc[1::2].copy()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1), (batch_size, batch_size))
            
        result_df.append(pd.concat(list(results), axis=0))
    
    result_df = pd.concat(result_df).sort_values('index').reset_index(drop=True)
    return result_df
        


%%time
data = data.sort_values("length", ascending=False)
result_df = fenbushi_predict(data)
print(result_df.shape)
result_df.head(2)


tmp = result_df.copy()

tmp = tmp.merge(test_df[['id','label']], on='id', how='left')
tmp['winner1']  = tmp['winner'].apply(lambda x:1 if x>0.5 else 0)

from sklearn.metrics import accuracy_score
print(accuracy_score(tmp['winner1'], tmp['label']))


# 分成两部分，对其中预测不准确的一部分，进行预测
res1 = result_df[abs(result_df["winner"] - 0.5) < PROB].copy().reset_index(drop=True)
res2 = result_df[abs(result_df["winner"] - 0.5) >= PROB].copy().reset_index(drop=True)

print(res1.shape)
print(res2.shape)

need_index = res1['index'].tolist()
aug_res1 = aug_data[aug_data.index.isin(need_index)].copy().reset_index(drop=True)
print(aug_res1.shape)


%%time
aug_res1 = fenbushi_predict(aug_res1)


# 这里一定要取反
aug_res1['winner'] = 1-aug_res1['winner']  
# 概率取平均
res1['winner'] = (res1['winner'] + aug_res1['winner']) / 2
# 两部分拼接
final_res = pd.concat([res1,res2])
final_res = final_res.sort_values('index').reset_index(drop=True)



# 合并标签后评估效果
final_res = final_res.merge(test_df[['id','label']], on='id', how='left')
final_res['winner1']  = final_res['winner'].apply(lambda x:1 if x>0.5 else 0)

from sklearn.metrics import accuracy_score
print(accuracy_score(final_res['winner1'], final_res['label']))




