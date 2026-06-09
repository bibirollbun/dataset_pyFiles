import sys 
import torch
import random
import numpy as np
import pandas as pd
import gc
import time
import random
from tqdm import tqdm
import os
from IPython.display import display

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel, GPT2LMHeadModel, GPT2TokenizerFast


modelgpt2 = GPT2LMHeadModel.from_pretrained('/kaggle/input/xlm-roberta-large/xlm-roberta-large').to('cuda')
tokenizergpt2 = GPT2TokenizerFast.from_pretrained('/kaggle/input/xlm-roberta-large/xlm-roberta-large')


# https://huggingface.co/spaces/PirateXX/Sentencewise-Perplexity/blob/main/app.py
def calculatePerplexity(text):
    encodings = tokenizergpt2(text, return_tensors="pt")
    max_length = modelgpt2.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = modelgpt2(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

    return ppl.item()


test = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
pred = []
for _,row in test.iterrows():
    p_a = calculatePerplexity(row['response_a'])
    p_b = calculatePerplexity(row['response_b'])
    pred.append(p_a/(p_a+p_b))
sub = pd.read_csv('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv')
sub['winner'] = pred
sub.to_csv('submission.csv', index=False)

