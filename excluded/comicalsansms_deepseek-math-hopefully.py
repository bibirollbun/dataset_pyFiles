!pip install torch
!pip install transformers

!pip install accelerate --no-index --find-links=file:///kaggle/input/offline-bitsandbytes-packages/accelerate/
!pip install bitsandbytes --no-index --find-links=file:///kaggle/input/offline-bitsandbytes-packages/bitsandbytes/


import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoConfig,
    set_seed
)


set_seed(42)

MODEL_PATH = "/kaggle/input/deepseek-math"

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

config = AutoConfig.from_pretrained(MODEL_PATH)
config.gradient_checkpointing = True


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, max_length=512)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
    # config=config
)


model.dtype


import gc
device = 'cuda'

import transformers

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1000,
    torch_dtype='auto',
    device_map="auto",
)

print(f"Transformers Version: {transformers.__version__}")


pipeline.device


question = "TEST"


import pandas as pd

def get_response(question):
    prompt = f"""
        Question: {question} 
        
        Instructions:
        what are the concepts needed to solve this question? list them out and remember the concepts.
        
        use these concepts to solve the question.
        
        If the answer is greater than 1000, the answer should be given as a non-negative modulo 1000. 
        Make sure that the final number in your response is the answer.
        Please integrate natural language reasoning with programs to solve the problem above.
        
        """
    response = pipeline(prompt)
    print(response)
    return response[0]['generated_text']

def get_responses(data):
    responses = []
    for i in data['problem']:
        answer = get_response(i)
        responses.append(answer[0]['generated_text'])
        
    return responses


def naive_parse(response):
    out = []
    start = False
    end = False
    for l in reversed(list(response)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    _out = reversed(out)
    out = ''.join(_out)
    print(out)
    return out


def _predict(question):
    response = get_response(question)
    answer = naive_parse(response)
    return answer


import os
import polars as pl

import kaggle_evaluation.aimo_2_inference_server


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    # Make a prediction
    prediction = _predict(question)
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




