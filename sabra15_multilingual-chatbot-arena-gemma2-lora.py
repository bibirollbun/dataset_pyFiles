!pip install transformers peft accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files --quiet


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


class Config:
    gemma_dir = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it-4bit/1/gemma-2-9b-it-4bit'
    lora_dir = '/kaggle/input/peft-model/peft_model'
    max_length = 2300
    batch_size = 4

cfg = Config()


test = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')

def process_text(text: str) -> str:
    return text.replace("null", "")

test.loc[:, 'prompt'] = test['prompt'].apply(process_text)
test.loc[:, 'response_a'] = test['response_a'].apply(process_text)
test.loc[:, 'response_b'] = test['response_b'].apply(process_text)

display(test.head(5))


def tokenize(
    tokenizer, prompt, response_a, response_b, max_length=cfg.max_length
):
    prompt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]

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
@torch.amp.autocast('cuda')
def inference(df, model, device, batch_size=cfg.batch_size, max_length=cfg.max_length):
    a_win, b_win, tie = [], [], []
    
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
        tie.extend(proba[:, 2].tolist())
    
    df["winner_model_a"] = a_win
    df["winner_model_b"] = b_win
    df["winner_tie"] = tie
    
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
proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")


result_df.loc[:, "winner_model_a"] = proba[:, 0]
result_df.loc[:, "winner_model_b"] = proba[:, 1]
result_df.loc[:, "winner_tie"] = proba[:, 2]

result_df['winner'] = result_df.apply(lambda row: 'model_a' if row['winner_model_a'] > row['winner_model_b'] else 'model_b', axis=1)

submission_df = result_df[['id', 'winner']]


submission_df.to_csv('submission.csv', index=False)
display(submission_df)

