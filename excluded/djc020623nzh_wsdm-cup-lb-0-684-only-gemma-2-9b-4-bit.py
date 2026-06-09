!pip install transformers peft accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files


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
    gemma_dir = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it-4bit/1/gemma-2-9b-it-4bit'
    lora_dir = '/kaggle/input/gemma2-v6/gemma2-v6'
    max_length = 2048
    batch_size_list = [2, 16]
    tta = True
cfg = Config()


tokenizer = GemmaTokenizerFast.from_pretrained(cfg.gemma_dir)
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"


test_df = pl.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet').to_pandas()
# test_df = test_df[1000:2000]
test_df


for col in ['prompt', 'response_a', 'response_b']:
    test_df[col] = test_df[col].fillna('')
    text_list = []
    if col == 'prompt':
        max_no = 502
        s_no = 250
        e_no = -251
    else:
        max_no = 702
        s_no = 350
        e_no = -351
    for text in tqdm(test_df[col]):
        encoded = tokenizer(text, return_offsets_mapping=True)
        if len(encoded['input_ids']) > max_no:
            start_idx, end_idx = encoded['offset_mapping'][s_no]
            new_text = text[:end_idx]
            # print(len(tokenizer(text[:end_idx])['input_ids']))
            start_idx, end_idx = encoded['offset_mapping'][e_no]
            # print(len(tokenizer(text[start_idx:])['input_ids']))
            new_text = new_text + "\n(snip)\n" + text[start_idx:]
            # print(len(tokenizer(new_text)['input_ids']), new_text)
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
data["gap"] = test_df["response_a"].str.len() - test_df["response_b"].str.len()
# print(data.head())

aug_data = pd.DataFrame()
aug_data["id"] = test_df["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test_df["prompt"], test_df["response_b"], test_df["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)
aug_data["gap"] = test_df["response_a"].str.len() - test_df["response_b"].str.len()


# Load base model on GPU 0
device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_0,
    use_cache=False,
)
model_0.score = torch.nn.Linear(in_features=3584, out_features=2, bias=False).to(device_0)

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_1,
    use_cache=False,
)
model_1.score = torch.nn.Linear(in_features=3584, out_features=2, bias=False).to(device_1)


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


st = time.time()
# sort by input length to fully leverage dynaminc padding
if cfg.tta:
    data = pd.concat([data, aug_data]).reset_index(drop=True)
    data['index'] = np.arange(len(data), dtype=np.int32)
    data = data.sort_values("length", ascending=False)
else:
    data['index'] = np.arange(len(data), dtype=np.int32)
    data = data.sort_values("length", ascending=False)

# print(data.head())

data_dict = {}
data_dict[0] = data[data["length"] > 1024].reset_index(drop=True)
data_dict[1] = data[data["length"] <= 1024].reset_index(drop=True)
result_df = []
for i, batch_size in tqdm(enumerate(Config.batch_size_list),total = len(Config.batch_size_list)):
    if len(data_dict[i]) == 0:
        continue
    sub_1 = data_dict[i].iloc[0::2].copy()
    sub_2 = data_dict[i].iloc[1::2].copy()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1), (batch_size, batch_size))
        
    result_df.append(pd.concat(list(results), axis=0))

result_df = pd.concat(result_df).sort_values('index').reset_index(drop=True)
if cfg.tta:
    proba = result_df['winner'].values[:len(aug_data)]
    tta_proba = 1 - result_df['winner'].values[len(aug_data):]
    proba = (proba + tta_proba) / 2
    result_df = result_df[:len(aug_data)].copy()
    result_df['winner'] = proba

    
    result_df.loc[((result_df['winner'] < 0.5) & (result_df['winner'] > 0.4)) &
                    (result_df["gap"] > 1000),"winner"] = result_df['winner'] + 0.05
    result_df.loc[((result_df['winner'] < 0.6) & (result_df['winner'] > 0.5)) &
                    (result_df["gap"] < -1000),"winner"] = result_df['winner'] - 0.05
    
print(f"elapsed time: {time.time() - st}")


submission_df = result_df[['id', 'winner']].copy()
submission_df.to_csv('submission_prob.csv', index=False)
submission_df['winner'] = np.where(submission_df['winner'] < 0.5, 'model_a', 'model_b')
submission_df.to_csv('submission.csv', index=False)
display(result_df[['id', 'winner']])


submission_df = result_df[['id', 'winner']].copy()
submission_df['winner'] = np.where(submission_df['winner'] < 0.5, 'model_a', 'model_b')
submission_df.to_csv('submission.csv', index=False)
display(result_df[['id', 'winner']])

