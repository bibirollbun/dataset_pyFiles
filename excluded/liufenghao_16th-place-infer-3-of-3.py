!pip install transformers peft accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files


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


assert torch.cuda.device_count() == 2


@dataclass
class Config:
    gemma_dir = '/kaggle/input/16th-place-quantize-2-of-3/merged-v159-4bit'
    max_length = 3072
    batch_size = 2
    device = torch.device("cuda")    
    tta = True  # test time augmentation. <prompt>-<model-b's response>-<model-a's response>
    spread_max_length = False  # whether to apply max_length//3 on each input or max_length on the concatenated input

cfg = Config()


test = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')


display(test.head(5))


def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length, spread_max_length=cfg.spread_max_length
):

    text = []
    for pp,aa,bb in zip(prompt,response_a,response_b):
        
        tmp = (
            f"<start_of_turn>prompt\n{pp}<end_of_turn>\n"
            + f"<start_of_turn>response_a\n{aa}<end_of_turn>\n"
            + f"<start_of_turn>response_b\n{bb}<end_of_turn>"
        )
        
        if len(tokenizer(tmp)["input_ids"]) >= max_length:
            tmp = (
            f"<start_of_turn>prompt\n{pp[-(max_length // 3):]}<end_of_turn>\n"
            + f"<start_of_turn>response_a\n{aa[-(max_length // 3):]}<end_of_turn>\n"
            + f"<start_of_turn>response_b\n{bb[-(max_length // 3):]}<end_of_turn>"
        ) # modify the three text len to fit the length limitation
        text.append( tmp )

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


print(tokenizer.decode(data["input_ids"][1]))


print(tokenizer.decode(aug_data["input_ids"][0]))


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


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    # a_win, b_win, tie = [], [], []
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
        # tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    # df["winner_tie"] = tie
    
    return df


st = time.time()

# sort by input length to fully leverage dynaminc padding
data = data.sort_values("length", ascending=False)
# the total #tokens in sub_1 and sub_2 should be more or less the same
sub_1 = data.iloc[0::2].copy()
sub_2 = data.iloc[1::2].copy()

with ThreadPoolExecutor(max_workers=2) as executor:
    results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

result_df = pd.concat(list(results), axis=0)
proba = result_df[["winner_model_a", "winner_model_b"]].values

print(f"elapsed time: {time.time() - st}")
#elapsed time: 4.2011353969573975


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
#elapsed time: 3.4222445487976074


result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
# result_df.loc[:, "winner_tie"] = proba[:, 2]
# submission_df = result_df[["id", 'winner_model_a', 'winner_model_b']]
# submission_df.to_csv('submission.csv', index=False)
# display(submission_df)
submission_df = result_df[["id", "winner_model_a"]].copy()
submission_df.rename(columns={"winner_model_a": "winner"}, inplace=True)
submission_df['winner'] = np.where(submission_df['winner'] > 0.5, 'model_a', 'model_b')
submission_df.to_csv('submission.csv', index=False)
display(submission_df)

