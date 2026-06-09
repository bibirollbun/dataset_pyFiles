!pip install torch transformers peft accelerate bitsandbytes \
    -U --no-index --find-links /kaggle/input/unsloth-wheel-file


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from datetime import datetime
import pytz
print('LOGGING TIME OF START (SGT):',  datetime.strftime(datetime.now(pytz.timezone('Asia/Singapore')), "%Y-%m-%d %H:%M:%S"))

import numpy as np
import pandas as pd
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor
import gc

from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForCausalLM,
    GemmaTokenizerFast,
)
from peft import (
    PeftModel,
)

from transformers.data.data_collator import pad_without_fast_tokenizer_warning
import torch.nn.functional as F
import torch.nn as nn

import torch
import transformers
import peft
import bitsandbytes
import accelerate

print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)


## helper ! ---
class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)

    else:
        raise NotImplementedError


print('import ok!!!')



MODE = 'local'  #'submit'
 
if MODE == 'local':
    valid_df = pd.read_parquet(f'/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
    valid_df = valid_df.fillna('')
    valid_df.loc[:, 'label'] = valid_df['winner'].map({'model_a': 1, 'model_b': 0})
    valid_df = valid_df[0::5]  #fold-0
    valid_df = valid_df[:500] #500

if MODE == 'submit':
    valid_df = pd.read_parquet(f'/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
    valid_df = valid_df.fillna('')

num_valid = len(valid_df)
print('num_valid', num_valid)

#----
cfg = dotdict(
    model_id= \
        '/kaggle/input/unsloth-gemma/Gemma',

    lora_dir= \
    '/kaggle/input/gemma119',
    #'/kaggle/input/wsdm-hengck23-weight-01a/checkpoint-1211-unsloth-fix',
        

    max_length=2400, #2048,
    max_prompt_length=560, #512,
    batch_size=2,
    float_type=torch.float16,
)


print('config ok!!!')



def load_tokenizer(model_id):
    tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
    tokenizer.add_eos_token = True
    #tokenizer.add_eos_token = False
    tokenizer.padding_side = 'right'
    return tokenizer


def load_model(model_id, lora_dir, device):

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_compute_dtype=cfg.float_type,
    #     bnb_4bit_use_double_quant=True,
    # )
    model = Gemma2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=cfg.float_type,
        device_map=device,
        #_attn_implementation="flash_attention_2",
        #quantization_config=bnb_config,
    )
    par = torch.nn.Parameter(torch.vstack([model.lm_head.weight[no_token_id, :], model.lm_head.weight[yes_token_id, :]]))
    print(par)
    model.lm_head=nn.Linear(in_features=3584, out_features=2, bias=False)
    #loaded_weight = torch.load("/kaggle/input/lm-head/lm_head_weight.pth")
    #model.lm_head.weight.data= loaded_weight
    model.lm_head=par
    print(model.lm_head.weight)
    model.lm_head=model.lm_head.to(device)

    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.eval()
    return model


tokenizer = load_tokenizer(cfg.model_id)
yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

## modeling #####################################################################
multi_model = [
    load_model(cfg.model_id, cfg.lora_dir, 'cuda:0'),
    load_model(cfg.model_id, cfg.lora_dir, 'cuda:1'),
]


print('model ok!!!')



## dataset #####################################################################
 

def process_one_row(row):
    prompt = (
        "You are a reviewer who is dedicated to evaluating the quality of the conversation. "
        "Given a question and two answers:" + row['prompt']
    )
    response_a = "model_a:" + row['response_a']
    response_b = "model_b: " + row['response_b']
    final = (
        "Please only output model_a or model_b, whichever answer you think is better. "
        "Yes means model_a is better and No means model_b is better. "
        'SOLUTION The correct answer is: "{}"'
    )

    p = tokenizer(prompt, add_special_tokens=False)['input_ids']
    a = tokenizer(response_a, add_special_tokens=False)['input_ids']
    b = tokenizer(response_b, add_special_tokens=False)['input_ids']
    f = tokenizer(final, add_special_tokens=False)['input_ids']

    if len(p) > cfg.max_prompt_length:
        p = p[-cfg.max_prompt_length:]

    response_length = (cfg.max_length - len(p) - 2 - 50) // 2

    input_ids = (
        [tokenizer.bos_token_id]
        + 
        p
        + a[-response_length:]
        + b[-response_length:]
        + f
        + [tokenizer.eos_token_id]
    )

    length = len(input_ids)
    attention_mask = [1] * length
    return input_ids, attention_mask, length


valid_df[['input_ids', 'attention_mask', 'lengths']] = \
    valid_df.apply(process_one_row, axis=1, result_type='expand')

valid_df = valid_df.sort_values('lengths', ascending=False).reset_index(drop=True)

multi_df = [
    valid_df[0::2].reset_index(drop=True),
    valid_df[1::2].reset_index(drop=True),
]
valid_df = pd.concat(multi_df).reset_index(drop=True)

print(valid_df['lengths'].max())
print('data ok!!!')




## inference #####################################################################

def make_submission(valid_df, probability):
    # id,winner
    predict = probability.argmax(-1)
    winner =[
        {1:'model_a',0:'model_b',}[p] for p in predict
    ]
    submit_df = pd.DataFrame({
        'id' : valid_df['id'],
        'winner': winner,
    })
    return submit_df

def do_infer(model, valid_df, name=''):
    num_valid = len(valid_df)
    start_timer = timer()
    probability = []
    for i in range(0, num_valid, cfg.batch_size):
        print('\r', f'{name}: {i}', time_to_str(timer() - start_timer, 'min'), end='', flush=True)

        B = min(i +  cfg.batch_size, num_valid) - i
        d = valid_df[i:i + B]
        batch = {
            'input_ids': d['input_ids'].tolist(),
            'attention_mask': d['attention_mask'].tolist(),
        }
        batch = pad_without_fast_tokenizer_warning(
            tokenizer,
            batch,  #{"input_ids": input_ids, "attention_mask": attention_mask},
            padding='longest',
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
        with torch.amp.autocast('cuda', enabled=True, dtype=cfg.float_type):
            with torch.no_grad():
                attention_mask = batch['attention_mask'].to(model.device)
                input_ids = batch['input_ids'].to(model.device)
                batch_size = len(attention_mask)
                length = attention_mask.sum(-1)
                batch_index = torch.arange(batch_size, device=model.device)
                last_index = length - 1 
                
                output = model(
                    input_ids=input_ids,
                    attention_mask=input_ids,
                )
                output.logits = output.logits[batch_index, last_index]
                p = F.softmax(output.logits, dim=1)
                probability.append(p.data.cpu().numpy())
                #print(probability.shape)
                #todo tta
    
    torch.cuda.empty_cache()
    print('')
    probability = np.concatenate(probability)
    return probability


if 0:
    #debug
    probability = do_infer(multi_model[0], multi_df[0], name='')
    print(probability.shape)
    exit(0)


start_timer = timer()
with ThreadPoolExecutor(max_workers=2) as executor:
    result = executor.map(do_infer, multi_model , multi_df, ('process0', 'process1'))
probability = np.concatenate(list(result))
time_taken = timer() - start_timer

submit_df = make_submission(valid_df, probability)
submit_df.to_csv('submission.csv', index=False)
np.save('probability.npy', probability)

print(submit_df)
print(probability.shape)
print(f'time for 10_000 (max  4.5 hr): {10_000/num_valid*time_taken/60/60:4.1f}')
print(f'time for 25_000 (max 12.0 hr): {25_000/num_valid*time_taken/60/60:4.1f}')
print('MODE', MODE)
print('submit ok!!!')




try:
    multi_model[0]= multi_model[0].to('cpu')
    multi_model[1]= multi_model[1].to('cpu')
    del multi_model 
    gc.collect()
    torch.cuda.empty_cache()
except:
    pass

print('free memory ok!!!')



if MODE=='local':
    truth   = valid_df['label'].values.astype(np.int32)
    predict = np.argmax(probability, -1)
    acc = (truth == predict).mean()

    logp = np.log(np.clip(probability, 1e-5, 1 - 1e-5))
    logloss = truth * logp[:, 1] + (1 - truth) * logp[:, 0]
    logloss = (-logloss).mean()
    print('acc:', acc)
    print('logloss:', logloss)



'''
 

'''

