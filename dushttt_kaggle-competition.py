# !pip install -q torch torchvision torchaudio
# !pip install -q transformers
# !pip install -q peft
# !pip install -q pandas
# !pip install -q datasets
# !pip install -q scikit-learn
# !pip install -q polars


!nvidia-smi





import os

import pandas as pd

import os
import re
import torch
import argparse
from torch import nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split

from torch.optim import AdamW  # Use PyTorch's implementation

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
from datasets import load_dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import torch.nn as nn
from tqdm import tqdm
# from cfg import CFG
from transformers import BatchEncoding
import pandas as pd

import torch
import torchvision.transforms.functional as F

# ===============================
seed = 1223
torch.manual_seed(seed)
# ===============================

import polars as pl

import kaggle_evaluation.aimo_2_inference_server


import torch
# from transformers import BitsAndBytesConfig
from peft import LoraConfig

class CFG_class():
    def __init__(self):
        self.batch_size = 4
        self.lr = 3e-4
        self.gc_lr = 1e-4
        self.num_epochs = 20
        self.eval_epoch = 0
        self.seed = 0
        self.weight_decay = 1e-4
        self.max_tokens = 256 #Difference in 128 to 256 is around 1GB
        self.print_every = 1
        self.debug = True
        self.name = "exp_test"
       
        self.generation_config = {
            "min_new_tokens": 10,
            "max_new_tokens": 40,
            "num_beams": 1,
            # "dola_layers": "high",
            "temperature": 0.7,
            "repetition_penalty": 1.03,
            "top_k": 30,
            "top_p": 0.9,
            "no_repeat_ngram_size": 3,
            "do_sample": True
        }

#         self.quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
# #             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            # target_modules=["q_proj", "v_proj", "mlp"],
            target_modules = ['q_proj','v_proj','gate_proj','down_proj','up_proj','lm_head'],
            # target_modules=["o_proj", "qkv_proj"],
            # target_modules=["qkv_proj"],
            lora_dropout=0.1,
            bias="none",
            # init_lora_weights="gaussian",
            # task_type="CAUSAL_LM"
        )
   
    def log_config(self, log):
        log.info("Config details:")
        d = vars(self)
        for attr in d:
            if type(d[attr]) is dict:
                log.info(f"{attr}:")
                for subattr in d[attr]:
                    log.info(f"  {subattr}: {d[attr][subattr]}")
            else:
                log.info(f"{attr}: {d[attr]}")
       
       
CFG = CFG_class()



# import os

# model_path = "/kaggle/input/huggingface-qwen2.5-math-7b-instruct/pytorch/default/1/models--Qwen--Qwen2.5-Math-7B-Instruct/blobs"
# print(os.listdir(model_path))


class MathReasoningModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(MathReasoningModel, self).__init__()
        # self.model = AutoModelForCausalLM.from_pretrained(text_model_name)
        # "meta-llama/Llama-3.1-8B",
        self.args = args
        self.tokenizer = tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            # args.text_model_name,
            "/kaggle/input/qwen-final/pytorch/base/1/qwen",
            # quantization_config=CFG.quantization_config,
            low_cpu_mem_usage=True,
            # trust_remote_code=True,
            device_map="cuda:0",
            # device_map="auto",
            # local_files_only=True,
            # # device_map="auto",  # Automatically distribute layers across available devices
            use_cache=False,
            torch_dtype=torch.float16,
            # cache_dir="/kaggle/input/huggingface-qwen2.5-math-7b-instruct/pytorch/default/1/models--Qwen--Qwen2.5-Math-7B-Instruct/blobs",
            # trust_remote_code=True,
        )      
        
        self.model.gradient_checkpointing_enable({"use_reentrant": False})
        self.model = get_peft_model(self.model, CFG.lora_config)            

    def forward(self, input_ids, attention_mask, labels=None):        
        generate_ids = self.model.generate(
            # **inputs,
            input_ids=input_ids,
            attention_mask=attention_mask, 
            temperature=self.args.temperature,
            do_sample=self.args.do_sample,
            top_k=5,               
            top_p=0.9,
            num_beams=2,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.5,
            # length_penalty=1.2,                                                
            # max_length=299,
            max_new_tokens=self.args.max_new_tokens,
            )            
        return generate_ids 


args_dict = {
    "batch_size": 1,
    "max_length": 700,  # Example required argument
    "text_model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
    "max_new_tokens": 3300,
    "temperature": 0.7,  # Example required argument
    "do_sample": True,
    "padding_side": "left",
    # "cache_dir": "/kaggle/input/huggingface-qwen2.5-math-7b-instruct/pytorch/default/1",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# /kaggle/input/huggingface-qwen2.5-math-7b-instruct/pytorch/default/1/models--Qwen--Qwen2.5-Math-7B-Instruct

args = argparse.Namespace(**args_dict)


model = None
tokenizer = None

def load_model(args):
    global tokenizer, model
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/kaggle/input/tokenizer/pytorch/default/1/tokenizer",
        add_bos_token=False, 
        use_fast=True,
        # cache_dir=args.cache_dir,
    )   

    tokenizer.use_default_system_prompt = False
    tokenizer.padding_side = args.padding_side

    print('Model Loading...')
    model = MathReasoningModel(args, tokenizer).to(args.device)        

    weight_path = "/kaggle/input/qwen2.5-math-7b-instruct/pytorch/default/1/phi4_reasoning_class_text_lora_final_model_wts_input_padding_left_2.pt"
    model.load_state_dict(torch.load(f"{weight_path}", map_location="cpu", weights_only=True), strict=True)
    model.to(args.device)        
    model.eval()

    # /kaggle/input/qwen2.5-math-7b-instruct/pytorch/default/1/phi4_reasoning_class_text_lora_final_model_wts_input_padding_left_2.pt
    print('Model Loading Completed.')
    
    if model is None:
        raise ValueError("LLM initialization failed")


# for i in model.named_parameters():
#     print(i)


# # Define a sample dataset
# class SampleDataset(Dataset):
#     def __init__(self, args, tokenizer, inputs, max_length):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.inputs = inputs
#         self.max_length = max_length
        
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token if pad_token is not defined
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        
#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
  
#         input_messages = [
#             {
#                 "role": "system",
#                 "content": "You are a mathematical expert providing correct solutions with logical reasoning. Please reason step by step, and put your final numeric answer within \\boxed{}."
#             },
#             {
#                 "role": "user", 
#                 "content": f"{self.inputs[idx]}",
#             },
#         ]
        
#         # print(f"input_messages: {input_messages}")
        
#         input_prompt= self.tokenizer.apply_chat_template(
#             input_messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )       
                
#         # print(f"input_prompt: {input_prompt}")

#         # print(self.max_length)

#         tokenized_inputs = self.tokenizer(            
#             input_prompt, 
#             padding="max_length", 
#             truncation=True, 
#             max_length=self.max_length, 
#             return_tensors="pt",
#         )             
        
#         # print(f"tokenized_inputs: {tokenized_inputs}")
        
#         tokenized_inputs = BatchEncoding({
#             key: (value.squeeze(0) if isinstance(value, torch.Tensor) else value) 
#             for key, value in tokenized_inputs.items() if value is not None
#         })                                 
                
#         # print(f"tokenized_inputs: {tokenized_inputs.input_ids.shape}")
        
#         return tokenized_inputs


# Define a sample dataset
def SampleDataset(args, tokenizer, question, max_length):
    
    if isinstance(question, list):
        input_messages = [
            {
                "role": "system",
                "content": "You are a mathematical expert providing correct solutions with logical reasoning. Please reason step by step, and put your final numeric answer within \\boxed{}."
            },
            {
                "role": "user", 
                "content": f"{question[0]}",
            },
        ]
        
    elif isinstance(question, str):
        input_messages = [
            {
                "role": "system",
                "content": "You are a mathematical expert providing correct solutions with logical reasoning. Please reason step by step, and put your final numeric answer within \\boxed{}."
            },
            {
                "role": "user", 
                "content": f"{question}",
            },        
        ]
    
    # print(f"input_messages: {input_messages}")
    
    input_prompt= tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )       
            
    # print(f"input_prompt: {input_prompt}")

    # print(self.max_length)

    tokenized_inputs = tokenizer(            
        input_prompt, 
        padding="max_length", 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt",
    )             
    
    # print(f"tokenized_inputs: {tokenized_inputs}")
    
    # tokenized_inputs = BatchEncoding({
    #     key: (value.squeeze(0) if isinstance(value, torch.Tensor) else value) 
    #     for key, value in tokenized_inputs.items() if value is not None
    # })                                 
            
    # print(f"tokenized_inputs: {tokenized_inputs.input_ids.shape}")
    
    return tokenized_inputs   
    



# def extract_first_numeric(text: str):
#     match = re.search(r'-?\d*\.\d+|-?\d+', text)  # Match both floats and integers
#     return int(match.group()) if match else int(0)  # Convert to float if found, else return None

# def get_answer(question: str):
#     dataset = SampleDataset(args, tokenizer, [question], max_length=args.max_length)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)#, collate_fn=custom_collate_fn)
    
#     with tqdm(dataloader, unit="batch") as tepoch:
#         for inputs in tepoch: 
#             with torch.no_grad():      
#                 # print(inputs.input_ids.shape)                   
#                 generate_ids = model(
#                     **inputs.to("cuda:0"),
#                     )

#                 # print(generate_ids.shape)
                
#                 generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                
#                 results = tokenizer.decode(
#                     generate_ids[0], 
#                     skip_special_tokens=True, 
#                     clean_up_tokenization_spaces=True)
                
#         # print(results)
                
#         answer = extract_first_numeric(results.split('boxed{')[-1].split('}')[0]) % 1000
                
#     return answer


def extract_first_numeric(text: str):
    match = re.search(r'-?\d*\.\d+|-?\d+', text)  # Match both floats and integers
    return int(match.group()) if match else int(0)  # Convert to float if found, else return None

def get_answer(question: str):
    dataset = SampleDataset(args, tokenizer, question, max_length=args.max_length)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)#, collate_fn=custom_collate_fn)
    
    # for inputs in dataset: 
    #     print(dataset[inputs])
    with torch.no_grad():      
        # print(inputs.input_ids)                   
        generate_ids = model(
            **dataset.to(args.device),
            )

        # print(generate_ids)
        
        generate_ids = generate_ids[:, dataset['input_ids'].shape[1]:]
        
        results = tokenizer.decode(
            generate_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True)
                
        # print(results)
                
        answer = int(extract_first_numeric(results.split('boxed{')[-1].split('}')[0]) % 1000)
                
    return answer


# def predict_answer(question):
#     global model, tokenizer

#     if model is None:
#         print('entered')
#         load_model(args)    
    
#     return get_answer(question)

# question = "what is 2+2"
# result = predict_answer(question)
# print(result)

# # exit(0)


# question = ["""<|im_start|>system
# You are a mathematical expert providing correct solutions with logical reasoning. Please reason step by step, and put your final numeric answer within \boxed{}.<|im_end|>
# <|im_start|>user
# # How many vertical asymptotes does the graph of $y=\frac{2}{x^2+x-6}$ have?<|im_end|>
# <|im_start|>assistant"""]

# result = predict_answer(question)
# print(result)


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    global model, tokenizer

    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    # Make a prediction

    if model is None:
        print('entered')
        load_model(args)         
    
    prediction = get_answer(question)
    
    return pl.DataFrame({'id': id_, 'answer': prediction})


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
        )
    )




