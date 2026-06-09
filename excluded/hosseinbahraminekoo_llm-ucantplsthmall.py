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

