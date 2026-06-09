import os
import time
from datetime import datetime, timedelta

os.environ['TZ'] = 'Asia/Kolkata'
time.tzset()
print('Starting')
start_time = datetime.now()

is_interactive = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
is_non_interactive = not is_interactive
is_rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN')
hide_output_and_error="2>&- >&-" if is_non_interactive else "" 
hide_output_and_error2="2>&- >&-"

!touch /kaggle/working/submission.parquet
print('submission.parquet created')
from time import sleep
if is_non_interactive:sleep(10)


!python -m pip install  /kaggle/input/browser-notifications-in-a-kaggle-kernel/*.whl 
!python -m pip install jupyter -q --no-index --find-links=/kaggle/input/browser-notifications-in-a-kaggle-kernel/ 
!python -m pip install -q /kaggle/input/browser-notifications-in-a-kaggle-kernel/jupyter-notify/dist/jupyternotify-0.1.15-py2.py3-none-any.whl --no-index


%reload_ext jupyternotify


import os
import subprocess

import pandas as pd
import polars as pl

import kaggle_evaluation.aimo_2_inference_server


import openai


if is_interactive:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ['GROQ_API_KEY'] = user_secrets.get_secret("GROQ_API_KEY")
    openai.api_key = os.getenv('GROQ_API_KEY')
    openai.base_url = 'https://api.groq.com/openai/v1/'
    model_path = 'Qwen-2.5-Coder-32b'
else:
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    log_file = open("/kaggle/working/vllm_output.log", "w")
    
    openai.api_key = 'test'
    openai.base_url = 'http://localhost:8000/v1/'
    
    model_path = "/kaggle/input/qwen2.5-coder/transformers/32b-instruct/1"
    # model_path = "/kaggle/input/qwen2.5-math/transformers/72b-instruct/1"
    
    command = [
        "python3.10", 
        "-m", 
        "vllm.scripts", 
        "serve",
        model_path,
        "--tensor_parallel_size", "4",
        "--gpu_memory_utilization", "0.99",
        "--enforce_eager",
        "--enable_prefix_caching",
        "--rope-scaling", '{"factor": 4.0, "original_max_position_embeddings": 32768, "rope_type": "yarn"}',
        "--max_model_len", "64000",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes"
        
    ]
    
    process = subprocess.Popen(command, stdout=log_file, stderr=log_file, start_new_session=True)
    
    print(f"Background process started with PID: {process.pid}")

    import requests
    import time
    try:
        while True:
            try:
                !tail -1 /kaggle/working/vllm_output.log
                requests.get('http://localhost:8000/v1/models')
                break
            except Exception as e:
                print(end='.')
                time.sleep(30)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')


test_file='/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv'
reference_file='/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'

pd.read_csv(
    reference_file
).drop('answer', axis=1).to_csv('reference.csv', index=False)


import re
import subprocess

def get_response(prompt):
    response = openai.chat.completions.create(
    model=model_path,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=1000,
    temperature=0.0,
    seed=42,
    stop = ['```output'],
)
    response_text = response.choices[0].message.content
    code_regex = r'```python(.*?)```'
    code_match = re.findall(code_regex, response_text, re.DOTALL)[-1]
    return code_match


prefix_code = """
old_print = print
import sympy
import sympy as sp
import numpy as np
def print(*args, **kwargs):
    global result
    args=list(args)
    for k,arg in enumerate(args):
        old_print(type(arg))
        if isinstance(arg,(int, float, sympy.Number)):
            args[k] = (int(arg) % 1000)
            result = args[k]
    old_print(*args, **kwargs)
"""

suffix_code = """
print(result)
"""
def d_print(*args,**kwargs):
    if debug:
        print(*args,**kwargs)
        print('---'*25)
    

def solve_problem(prompt, attempt=1):
    global e
    if attempt ==1:
        # problem index 7
        # prompt += '\n\nWrite efficient python program using recursion correctly without using combinations, lru_cache, permutations. Enhance the Arbitrary upper limit for search'
        # problem index 1,2
        prompt += '\n\nWrite a python program.'
    d_print(prompt)
    code_match = get_response(prompt)
    # code_match = 'dummy'
    if code_match:
        d_print(code_match)
        try:
            if code_match != 'dummy':
                with open('solution.py', 'w') as f:
                    f.write(prefix_code + code_match + suffix_code)
            output = subprocess.check_output(['python', 'solution.py'], text=True, timeout=10, stderr=subprocess.STDOUT).strip()
            result=output.split('\n')[-1]
            return int(result)
        except subprocess.TimeoutExpired:
            if attempt<3:
                new_prompt = 'The code is timedout. This is your last attempt.'
                if 'arbitrary' in code_match.lower():
                    new_prompt = 'Just modify the arbitrary upper limit to dynamic values.'
            
                return solve_problem('\n\n'.join((prompt,code_match, new_prompt )),attempt+1)
            print('Timed out')
        except subprocess.CalledProcessError as e:
            print('CalledProcessError', e.returncode,e.output)
        except Exception as e:
            print(f"Error executing code: {e}")
    else:
        print("No code found in the response.")
    return 0



if is_interactive:
    df=pd.read_csv(reference_file)
    debug=1
    q=df.problem[2]
    solve_problem(q)


count=0
debug=1
import re
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    global count
    count +=1
    id_ = id_.item(0)
    question = question.item(0)
    if count !=1 and 0:return pl.DataFrame({'id': id_, 'answer': 0})
    print(count, question)
    # Make a prediction
    try:
        answer = solve_problem(question)
        print(answer)
    except Exception as e:
        print(f"Error executing code: {e}")
        try:
            response= get_response(question)
            boxed = re.search(r'\\boxed\{(.*)\}', response)
            if boxed:
                answer = int(boxed.group(1))%1000
            else:
                answer = int(response)%1000
        except Exception as e:
            answer = 0
    print('---')
    return pl.DataFrame({'id': id_, 'answer': answer})


inference_file='reference.csv'


count=0
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
inference_file='reference.csv'
# inference_file=test_file

if is_rerun:
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            inference_file,
        )
    )


try:
    import pandas as pd
    df = pd.read_parquet('submission.parquet')
    df2 = pd.read_csv(reference_file)
    merge = pd.merge(df2, df, on='id', how='left')
    merge['result'] = merge['answer_x'].astype(str) == merge['answer_y'].astype(str)
    display(merge)
    print(merge['result'].value_counts())
except Exception as e:
    print(e)


%%notify
2

