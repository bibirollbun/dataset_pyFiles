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
df = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
df.head()


# Copied from Carlos Henrique C. Matos
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import numpy as np
import pandas as pd
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

import sys 
import torch
import random
import time
from IPython.display import display

if (not torch.cuda.is_available()): print("Sorry - GPU required!")
    
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

test_df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
if 'id' not in test_df.columns or 'topic' not in test_df.columns:
    raise ValueError("Input CSV must contain 'id' and 'topic' columns.")

# Clear GPU memory and delete existing objects if they exist
if torch.cuda.is_available():
    torch.cuda.empty_cache()
for obj in ['model', 'pipe', 'tokenizer']:
    if obj in globals():
        del globals()[obj]

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1")

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/phi-3.5-mini-instruct/pytorch/default/1",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

max_new_tokens = 180
temperature=0.9
top_p = 0.9

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p = top_p,
    do_sample=True   
)

def get_response(template, topic):
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": topic}
    ]

    # Generate answer
    outputs = pipe(messages)
    response = outputs[0]['generated_text'][-1]['content']
    print(response)

    # Remove leading and trailing spaces
    response = response.strip()
    
    # Find last punctuation mark
    last_period = response.rfind('.')
    last_question = response.rfind('?') 
    last_exclamation = response.rfind('!')
    last_close_bracket = response.rfind(']')
    
    # Find the last occurring punctuation mark
    last_punct = max(last_period, last_question, last_exclamation, last_close_bracket)
    
    # If we found punctuation, trim to it; otherwise return full response
    if last_punct != -1:
        return response[:last_punct + 1]
    return response

def create_essay(topic):
    essay_template = """
    You are an expert essay writer. Write a 100-word essay about a given topic. 
    Make your essay spark different views from a group of judges with various beliefs.
    
    Note that:
        - Some judges like organized, fair opinions
        - others prefer imaginative, positive, or emotional writing. 
        - A few focus on deep thinking and strong reasoning. 
    
    For writing the essay choose a style that some judges will like, but not all (e.g., formal, casual, critical, or hopeful). 
    Add parts that can be seen in different ways, like statements that can be debated.
    
    Your writing should be about 100 words, clear, and make sense, and avoid repetition and plagiarism.
    """
    return get_response(essay_template, f"Your topic is: {topic}")

def get_essays(verbose=True):

    # Load test data and create submission DataFrame
    submission = pd.DataFrame()
    submission['id'] = test_df['id']
    submission['essay'] = ''
    
    # Generate essay for each topic
    for i, row in test_df.iterrows():
        
        if verbose:
            print(f"\n{'*'*5}{row['topic']}{'*'*5}\n")
        
        essay = create_essay(row['topic'])
       
        submission.loc[i, 'essay'] = essay
       
    return submission

verbose = not bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
submission = get_essays(verbose = verbose)    
submission

submission.to_csv('submission.csv', index=False)

















!pip install vllm


%%writefile run_vllm.py
import sys
import re
import gc
import vllm
print('vllm version=',vllm.__version__)
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

df = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')

llm = vllm.LLM(model=os.getenv("llm_path"),
          dtype='half',
          enforce_eager=True,
          gpu_memory_utilization=0.98,
          max_model_len=1024,
          tensor_parallel_size=2,
          trust_remote_code=True)
tokenizer = llm.get_tokenizer()

def apply_template(topic, tokenizer):
        messages = [
        {"role": "system", 
         "content": '''You are an expert essay writer. Write a comprehensive essay on the given topic.
IMPORTANT: Limit the essay to approximately 100-150 words.'''
        },{
            "role": "user", 
            "content": topic
        }]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

df["topic"] = df['topic'].apply(lambda x: apply_template(x, tokenizer))
print('Example input-\n',df["topic"][0])

responses = llm.generate(
    df["topic"].values,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0.7,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=199,  # Maximum number of tokens to generate per output sequence.
    ),
    use_tqdm = True
)

df["essay"] = [x.outputs[0].text for x in responses]
df.to_csv("submission.csv", columns=["id", "essay"], index=False)


!python run_vllm.py


model.save_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained(model_name)


tokenizer.save_pretrained("tokenizer")



# Sample Prompts
prompts = [
    "Explain the concept of quantum computing in simple terms.",
    "Generate a Python script to calculate the Fibonacci sequence.",
    "Write a short story about a robot discovering emotions.",
    "Provide a summary of the benefits of using renewable energy."
]


test["essay"] = essays
test[["id", "essay"]].to_csv("submission.csv", index=False)
test[["id", "essay"]].head()

