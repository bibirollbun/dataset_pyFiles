import os
import polars as pl
import kaggle_evaluation.aimo_2_inference_server


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # "0,1,2,3"


%%time
!pip uninstall -y torch torchaudio tourchvision
!pip install -U --no-index --find-links=/kaggle/input/vllm-whl -U vllm
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl


# ====================================================
# Library
# ====================================================
import gc
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
import math
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm
import re

import vllm


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3" # "0,1,2,3"

MODEL_PATH = "/kaggle/input/deepseek-finetune/transformers/gpt_ds_coder/1/competition_gpt_ds_code"


llm = vllm.LLM(
    model = MODEL_PATH, 
    tensor_parallel_size=4, # 2, 4 
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    swap_space=2, # L4×4
)
tokenizer = llm.get_tokenizer()


def generate_text_vllm(requests, tokenizer, model):
    sampling_params = vllm.SamplingParams(
        temperature=0.9,
        seed=42, 
        max_tokens=1024
    )
    responses = model.generate(requests, sampling_params=sampling_params, use_tqdm=False)
    response_text_list = []
    for response in responses:
        # total_tokens += len(response.outputs[0].token_ids)
        response_text_list.append(response.outputs[0].text)
    return response_text_list


def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)


tool_instruction = '.\nPlease solve the problem above, and put your final answer within latex \\boxed{}.'


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.Series, question: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # print(id_, question)
    # print(type(id_), type(question))
    id_ = id_.item(0)
    question = question.item(0)
    # print(id_, question)
    # print(type(id_), type(question))
    prompt = question + tool_instruction
    print('Prompt', prompt)
    generate_text = generate_text_vllm([prompt], tokenizer, llm)[0]
    answer = -111
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', generate_text)
        # print(result_output)
        # print('Text: ', generate_text)
        if len(result_output) > 0:
            no = naive_parse(result_output[0])
            if len(no) > 0:
                answer = int(no) % 1000
        
        else:
            result_output = generate_text[-1]
            answer = int(float(result_output))
        print(answer)
    except:
        print('error')
    return pl.DataFrame({'id': id_, 'answer': answer})


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
        )
    )

