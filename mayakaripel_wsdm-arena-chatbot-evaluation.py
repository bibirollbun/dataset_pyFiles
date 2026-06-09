# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip uninstall bitsandbytes -y
!pip cache purge


import os
import shutil

# Define paths to potential bitsandbytes installations
bitsandbytes_paths = [
    '/usr/local/lib/python3.10/dist-packages/bitsandbytes',
    '/usr/local/lib/python3.10/dist-packages/bitsandbytes-0.45.1.dist-info',
    '/usr/local/lib/python3.10/dist-packages/bitsandbytes-0.43.1.dist-info'
]

# Remove the directories if they exist
for path in bitsandbytes_paths:
    if os.path.exists(path):
        shutil.rmtree(path)


!pip install /kaggle/input/bitsandbytes-0-45-1-wheel-file/bitsandbytes-0.45.1-py3-none-manylinux_2_24_x86_64.whl


import bitsandbytes as bnb
print(bnb.__version__)


import time  # Import the time module
import numpy as np
import pandas as pd
import gc
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

# Helper function to convert time to string
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

# Define necessary variables
class CFG:
    def __init__(self):
        self.model_id = '/kaggle/input/lmsys-checkpoints-0-0805'
        self.lora_dir = '/kaggle/input/wsdm-gemma-2-2b-lora-training/gemma-2-2b-finetuned-seq-cls-wsdm/checkpoint-9687'
        self.max_length = 512
        self.max_prompt_length = 512
        self.batch_size = 1
        self.memory_threshold = 12.0
        self.bits = 4
        self.seed = 42
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_target_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
        self.device = "cuda:0"
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

cfg = CFG()

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model directly with quantization and dispatch
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(cfg.model_id, quantization_config=quantization_config, device_map={"": cfg.device})
print(f"Model initialized and dispatched on {cfg.device}")
end_time = time.time()
print(f"Base model loading time: {end_time - start_time:.2f} seconds")

# Load LoRA after the base model has been loaded
start_time = time.time()
print(f"Loading Lora model from {cfg.lora_dir}")
lora_config = LoraConfig(
    r=cfg.lora_r,
    lora_alpha=cfg.lora_alpha,
    target_modules=cfg.lora_target_modules,
    lora_dropout=cfg.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Load the LoRA model with shape mismatch handling
try:
    model = PeftModel.from_pretrained(model, cfg.lora_dir, config=lora_config, ignore_mismatched_sizes=True)
except RuntimeError as e:
    print(f"An error occurred during LoRA loading: {e}")
    raise e

model.eval()
end_time = time.time()
print(f"LoRA model loading time: {end_time - start_time:.2f} seconds")


# Define the one_data_process function
def one_data_process(row):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    prompt = '<prompt>: ' + row['prompt']
    response_a = '\n\n<response_a>: ' + row['response_a']
    response_b = '\n\n<response_b>: ' + row['response_b']
    p = tokenizer(prompt, add_special_tokens=False)['input_ids']
    a = tokenizer(response_a, add_special_tokens=False)['input_ids']
    b = tokenizer(response_b, add_special_tokens=False)['input_ids']

    if len(p) > cfg.max_prompt_length:
        p = p[-cfg.max_prompt_length:]
    reponse_length = (cfg.max_length - len(p)-2) // 2  ##-2
    input_ids = \
        [tokenizer.bos_token_id] + \
        p + a[-reponse_length:] + b[-reponse_length:] + \
        [tokenizer.eos_token_id]
    length = len(input_ids)
    attention_mask = [1] * length
    return input_ids, attention_mask, length

# Load validation data
MODE = 'submit'  # 'submit'

if MODE == 'local':
    valid_df = pd.read_parquet(f'/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
    valid_df = valid_df.fillna('')
    valid_df.loc[:, 'label'] = valid_df['winner'].map({'model_a': 0, 'model_b': 1})
    valid_df = valid_df[0::5]  # fold-0
    valid_df = valid_df[:1000]  # 500

if MODE == 'submit':
    valid_df = pd.read_parquet(f'/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
    valid_df = valid_df.fillna('')

num_valid = len(valid_df)
print('num_valid', num_valid)

# Prepare Data
valid_df[['input_ids', 'attention_mask', 'length']] = \
    valid_df.apply(one_data_process, axis=1, result_type='expand')
valid_df = valid_df.sort_values('length', ascending=False).reset_index(drop=True)
multi_df = [
    valid_df[0::2].reset_index(drop=True),
    valid_df[1::2].reset_index(drop=True),
]
valid_df = pd.concat(multi_df).reset_index(drop=True)
print('data ok!!!')

# Inference function
def check_memory_usage():
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_used = int(result.stdout.decode('utf-8').strip().split('\n')[0])
    return memory_used / 1024  # Convert to GB

def do_infer(model, valid_df, name=''):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)  # Load tokenizer within the function
    num_valid = len(valid_df)
    start_timer = timer()
    probability = []
    for i in range(0, num_valid, cfg.batch_size):
        print('\r', f'{name}: {i}', time_to_str(timer() - start_timer, 'min'), end='', flush=True)
        
        # Check memory usage and implement early stopping if necessary
        memory_used = check_memory_usage()
        if memory_used > cfg.memory_threshold:
            print(f"\nMemory usage exceeded threshold: {memory_used} GB. Stopping inference.")
            break
        
        B = min(i + cfg.batch_size, num_valid) - i
        d = valid_df[i:i + B]
        batch = {
            'input_ids': d['input_ids'].tolist(),
            'attention_mask': d['attention_mask'].tolist(),
        }
        batch = pad_without_fast_tokenizer_warning(
            tokenizer,
            batch,
            padding='longest',
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
        with torch.amp.autocast('cuda', enabled=True):  # Mixed precision
            with torch.no_grad():
                output = model(
                    input_ids=batch['input_ids'].to(cfg.device),
                    attention_mask=batch['attention_mask'].to(cfg.device),
                )
                mean_logits = torch.mean(output.logits, axis=1)
                p = F.softmax(mean_logits[:, :2], dim=1)
                probability.append(p.data.cpu().numpy())
            
        del batch  # deallocate memory
        del d
        torch.cuda.empty_cache()
        
    print('')
    probability = np.concatenate(probability)
    return probability

def make_submission(valid_df, probability):
    # id,winner
    print("Shape of probability:", probability.shape)  # Print the shape of probability
    predict = probability.argmax(axis=1) # Take the argmax along axis 1.
    print("Shape of predict:", predict.shape)  # Print the shape of predict
    print("Type of elements in predict:", predict.dtype) #print the data type of the elements of predict
    winner = [
        {0: 'model_a', 1: 'model_b'}[int(p)] for p in predict
    ]
    submit_df = pd.DataFrame({
        'id': valid_df['id'].to_list(),
        'winner': winner,
    })
    return submit_df

# Perform Inference and Create Submission (using accelerate)
start_timer = timer()
torch.cuda.empty_cache()  # Clear GPU memory

# Perform inference
probability = do_infer(model, valid_df, 'process0')
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


# Clean Up Memory (no changes needed)
try:
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()
except:
    pass

print('free memory ok!!!')

