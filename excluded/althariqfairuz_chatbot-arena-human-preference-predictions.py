# import kagglehub

# # Download latest version
# path_1 = kagglehub.model_download("seguride/gemma2-9b-it-bnb-4bit-lora-tune/transformers/default")
# path_2 = kagglehub.model_download("seguride/gemma-2-9b-it-bnb-4bit-unsloth/transformers/default")
# path_3 = kagglehub.dataset_download("emiz6413/lmsys-wheel-files")


!pip install transformers peft accelerate\
    -U --no-index --find-links /kaggle/input/lmsys-wheel-files


!pip install bitsandbytes \
    -U --no-index --find-links /kaggle/input/bitsandbytes-0-43-2-py3-none-manylinux-2-24-x86-64


import numpy as np 
import pandas as pd
import os
import time
from dataclasses import dataclass
import torch
import sklearn
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast, BitsAndBytesConfig
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import PeftModel
from concurrent.futures import ThreadPoolExecutor


assert torch.cuda.device_count() == 2


train_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/train.csv")
test_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/test.csv")
sample_df = pd.read_csv("/kaggle/input/lmsys-chatbot-arena/sample_submission.csv")


train_df.info()


train_df.head(5)


test_df.info()


test_df.head(5)


sample_df.info()


sample_df.head()


@dataclass
class Config:
    gemma_dir = '/kaggle/input/gemma-2-9b-it-bnb-4bit-unsloth/transformers/default/1'
    lora_dir = '/kaggle/input/gemma2-9b-it-bnb-4bit-lora-tune/transformers/default/1'
    max_length = 2048
    batch_size = 4
    device = torch.device("cuda")    
    tta = True
    spread_max_length= False

cfg= Config()


def process_text(text:str) -> str:
    stripped_str = text.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return ' '.join(sentences)

test_df.loc[:, 'prompt'] = test_df['prompt'].apply(process_text)
test_df.loc[:, 'response_a'] = test_df['response_a'].apply(process_text)
test_df.loc[:, 'response_b'] = test_df['response_b'].apply(process_text)

display(test_df.head(5))


def tokenize (tokenizer, prompt, response_a, response_b, max_length= cfg.max_length, spread_max_length=cfg.spread_max_length):
    propmt = ["<prompt>: " + p for p in prompt]
    response_a = ["\n\n<response_a>: " + r_a for r_a in response_a]
    response_b = ["\n\n<response_b>: " + r_b for r_b in response_b]
    if spread_max_length:
        prompt = tokenizer(prompt, max_length = max_length*2//10, truncation = True, padding = False).input_ids
        response_a = tokenizer(response_a, max_length=max_length*4//10, truncation=True, padding=False).input_ids
        response_b = tokenizer(response_b, max_length=max_length*4//10, truncation=True, padding=False).input_ids
        input_ids = [p + r_a + r_b for p, r_a, r_b in zip (prompt, response_a, response_b)]
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
data["id"] = test_df["id"]
data["input_ids"], data["attention_mask"] = tokenize(tokenizer, test_df["prompt"], test_df["response_a"], test_df["response_b"])
data["length"] = data["input_ids"].apply(len)

aug_data = pd.DataFrame()
aug_data["id"] = test_df["id"]
# swap response_a & response_b
aug_data['input_ids'], aug_data['attention_mask'] = tokenize(tokenizer, test_df["prompt"], test_df["response_b"], test_df["response_a"])
aug_data["length"] = aug_data["input_ids"].apply(len)


aug_data = aug_data.loc[data.index]

max_length = aug_data["length"].max()
max_length_indices = aug_data[aug_data["length"] != max_length].index


print(tokenizer.decode(data["input_ids"][0]))


print(tokenizer.decode(aug_data["input_ids"][0]))


# Load base model on GPU 0
device_0 = torch.device('cuda:0')
model_0 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_0,
    num_labels=3,
    use_cache=False,
)

# Load base model on GPU 1
device_1 = torch.device('cuda:1')
model_1 = Gemma2ForSequenceClassification.from_pretrained(
    cfg.gemma_dir,
    device_map=device_1,
    num_labels=3,
    use_cache=False,
)


model_0 = PeftModel.from_pretrained(model_0, cfg.lora_dir)
model_1 = PeftModel.from_pretrained(model_1, cfg.lora_dir)


@torch.no_grad()
@torch.amp.autocast(device_type='cuda')

def inference (df, model, device, batch_size = cfg.batch_size, max_length = cfg.max_length):
    a_win, b_win, tie = [], [], []

    for start_idx in range (0, len(df), batch_size):
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
# proba = result_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

print(f"elapsed time: {time.time() - st}")


result_df.head(5)


st = time.time()

if cfg.tta:
    # Sort by input length to boost speed
    aug_data = aug_data.loc[data.index]
        
    # Select the rows with the maximum length
    max_length = aug_data["length"].max()
    aug_data = aug_data[aug_data["length"] != max_length]

    sub_1 = aug_data.iloc[0::2].copy()
    sub_2 = aug_data.iloc[1::2].copy()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(inference, (sub_1, sub_2), (model_0, model_1), (device_0, device_1))

    tta_result_df = pd.concat(list(results), axis=0)
    # Recall TTA's order is flipped
#     tta_proba = tta_result_df[["winner_model_b", "winner_model_a", "winner_tie"]].values 
    
#     # Average original result and TTA result for the corresponding indices.
#     proba[max_length_indices] = (proba[max_length_indices] + tta_proba) / 2
    result_df.loc[tta_result_df.index, ['winner_model_a','winner_model_b','winner_tie']] = (result_df.loc[tta_result_df.index][['winner_model_a','winner_model_b','winner_tie']].values + tta_result_df.loc[tta_result_df.index][['winner_model_b','winner_model_a','winner_tie']].values)/2
print(f"Elapsed time: {time.time() - st}")


submission_df = result_df[["id", 'winner_model_a', 'winner_model_b', 'winner_tie']]
submission_df.to_csv('submission.csv', index=False)
display(submission_df)

