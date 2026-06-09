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


from transformers import set_seed
set_seed(42)


import os
import gc
import time
import warnings

import pandas as pd
import polars as pl

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 45) * 60


from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm_model_pth = '/kaggle/input/qwq-32b-preview/transformers/default/1'

llm = LLM(
    llm_model_pth,
    #dtype="half",                -> Changed this
    #max_num_seqs=128,            -> Changed this
    max_model_len=32768,#4096*10,         
    trust_remote_code=True,     
    tensor_parallel_size=4,      
    gpu_memory_utilization=0.96, 
)


tokenizer = llm.get_tokenizer()


import re

#prompts
thoughts = [
    'Please use chained reasoning and put the answer in \\boxed{}.',
    'Please reflect and verify while reasoning and put the answer in \\boxed{}.',
    'Solve the following problem using concise and clear reasoning and put the answer in \\boxed{}.',
    'Please reason carefully, with the help of a python programme, and finally put the answer into \\boxed{}.',
    'You are a helpful and reflective math assistant, please reason step by step concisely to put the answer in \\boxed{}.',
    # 'Solve the problem through concise and clear logical reasoning, and put the answer in \\boxed{}.',
    'You are the smartest math expert in the world, please spike this problem and put the answer in \\boxed{}.'
]

#create single prompt
def make_next_prompt(text,round_idx):
    default_prompt = thoughts[(round_idx+1)%len(thoughts)]
    default_python_code = f"print('{default_prompt}')"
    return default_python_code

#extract python code from response
def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        ans = "\n\n".join(matches)
        #print(f'Extracted python code: {ans}')
        return ans
    return ""

#extract all code segments
def extract_python_code_list(text):
    pattern = r'```python\s*(.*?)\s*```'
    ans=[]
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        ans.append(m)
    return ans

#process the code
def process_python_code(query):
    query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
    current_rows = query.strip().split("\n")
    new_rows = []
    for row in current_rows:
        new_rows.append(row)
    ans = "\n".join(new_rows)
    print(f'Processed python code: {ans}')
    return ans

import re

#extract the answer from the boxes
def extract_boxed_texts(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return []
    ans = []
    for content in matches:
        if content.isdigit():
            num = int(content)
        else:
            nums = re.findall(r'\d+', content)
            if not nums:
                continue
            num = int(nums[-1])
        ans.append(num % 1000)
    return ans
    
#extract the integer answer modulo 1000 from the boxes
def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return -1
    content = matches[0]
    if content.isdigit():
        num = int(content)
    else:
        nums = re.findall(r'\d+', content)
        if not nums:
            return -1
        num = int(nums[-1])
    return num % 1000

#select the final answer based on the frequency/ majoity voting
from collections import Counter
def select_answer(answers):
    valid_answers = []
    for answer in answers:
        try:
            if int(answer) == float(answer):
                if 1 < int(answer) < 999 and int(answer) % 100 > 0:
                    valid_answers.append(int(answer))
        except:
            pass
    if not valid_answers:
        return 49
    _, answer = sorted([(v,k) for k,v in Counter(valid_answers).items()], reverse=True)[0]
    return answer%1000


import os
import tempfile
import subprocess

#Python REPL to execute code. taken from NuminaMath Solution
class PythonREPL:
    def __init__(self, timeout=8):
        self.timeout = timeout

    def __call__(self, query):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            
            try:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self.timeout} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                # This makes the error message cleaner and more user-friendly
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                return False, combined_output


#sanity check
list_of_texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    for messages in [[{"role": "user", "content": "hi"}]]
]


# #define the sampling parameters
# sampling_params = SamplingParams(
#     temperature=1.0,              # Controls randomness in generation: higher values (e.g., 1.0) produce more diverse output.
#     min_p=0.01,                   # Minimum cumulative probability for nucleus sampling, filtering out unlikely tokens.
#     skip_special_tokens=True,     
#     # max_tokens=1800,            
#     max_tokens=32768,             # Sets a very high limit for token generation to handle longer outputs.
#     # stop=["```output"],       
# )

sampling_params = SamplingParams(
    temperature=0.01,
    repetition_penalty=1.0,
    top_k=20,
    top_p=0.8,
    max_tokens=32768,
)

#generate prompts in batch
def batch_message_generate(list_of_messages) -> list[list[dict]]:
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    
    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    
    for messages, single_request_output in zip(list_of_messages, request_output):
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        print(messages[-1])

    return list_of_messages


#filter answers from the responses

def batch_message_filter(list_of_messages,list_of_idx) -> tuple[list[list[dict]], list[str]]:
    global answer_contributions
    extracted_answers = []
    list_of_messages_to_keep = []
    list_of_idx_to_keep = []
    for idx,messages in zip(list_of_idx,list_of_messages):
        answers = extract_boxed_texts(messages[-1]['content'])
        if answers:
            extracted_answers.extend(answers)
            for answer in answers:
                answer_contributions[answer].append(idx)
        else:
            list_of_messages_to_keep.append(messages)
            list_of_idx_to_keep.append(idx)
    return list_of_messages_to_keep, extracted_answers, list_of_idx_to_keep


#execute the codes in the responses
def batch_message_execute(list_of_messages,round_idx) -> list[list[dict]]:
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'],round_idx)
        python_code = process_python_code(python_code)
        try:
            success, output = PythonREPL()(python_code)
        except Exception as e:
            output = str(e)
        messages.append({'role': 'user', 'content': output})
        print(messages[-1])
    return list_of_messages

#execute the code and generate the answer from responses
def batch_message_execute_and_get_answer(list_of_messages,round_idx) -> tuple[list[list[dict]], list[int]]:
    ans = []
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'])
        python_code = process_python_code(python_code)
        try:
            success, output = PythonREPL()(python_code)
            if success:
                patten = r'(\d+)'
                matches = re.findall(patten, output)
                if matches:
                    for match in matches:
                        ans.append(int(match)%1000)
                        ans.append(int(match)%1000) #代码权重高于自然语言，所以添加两次 
        except Exception as e:
            output = str(e)
        print(f'python code output: {output}')
    return ans

#execute code and generate answer for all elements in batch
def batch_message_list_execute_and_get_answer(list_of_messages,round_idx) -> tuple[list[list[dict]], list[int]]:
    ans = []
    for messages in list_of_messages:
        python_code_list = extract_python_code_list(messages[-1]['content'])
        for python_code in python_code_list:
            python_code = process_python_code(python_code)
            print("\npython code: \n", python_code)
            try:
                success, output = PythonREPL()(python_code)
                if success:
                    patten = r'(\d+)'
                    matches = re.findall(patten, output)
                    if matches:
                        for match in matches:
                            ans.append(int(match)%1000)
                            ans.append(int(match)%1000) 
            except Exception as e:
                output = str(e)
            print(f'\npython code output: {output}\n')
    return ans


import os

import pandas as pd
import polars as pl

#API for competition submission
import kaggle_evaluation.aimo_2_inference_server


#correct answers for reference problems
def get_correct_answer(question):
    if 'Three airline' in question: return 79
    if 'Fred and George' in question: return 250
    if 'Triangle $ABC$' in question: return 180
    if 'Find the three' in question: return 143
    if 'We call a' in question: return 3
    if 'Let $ABC$ be' in question: return 751
    if 'For a positive' in question: return 891
    if 'For positive integers' in question: return 810
    if 'The Fibonacci numbers' in question: return 201
    if 'Alice writes all' in question: return 902
    return 0


#predict function to solve single problem

from collections import Counter, defaultdict
g_score = 0
g_count = 0
prompt_score = Counter()
answer_contributions = defaultdict(list)
def predict_for_question(question: str) -> int:
    global g_score
    global g_count
    global prompt_score
    global answer_contributions
    question += "\nIf the final answer is a number larger than 1000, take modulo 1000. "
    if time.time() > cutoff_time: 
        return 210
    print(question)
    
    list_of_messages = [
        [
            {"role": "system", "content": thoughts[k]},
            {"role": "user", "content": question}
        ] for k in range(6)
    ]

    all_extracted_answers = []
    list_of_idx = list(range(len(list_of_messages)))
    max_round = 1
    for round_idx in range(max_round):
        print(f"round {round_idx+1}")
        list_of_messages = batch_message_generate(list_of_messages)
        #extracted_python_answer = batch_message_execute_and_get_answer(list_of_messages,round_idx)
        extracted_python_answer = batch_message_list_execute_and_get_answer(list_of_messages,round_idx)
        list_of_messages, extracted_answers, list_of_idx  = batch_message_filter(list_of_messages, list_of_idx)
        all_extracted_answers.extend(extracted_python_answer)
        all_extracted_answers.extend(extracted_answers)
        print("extracted boxed answers:",extracted_answers)
        print("extracted python answers:",extracted_python_answer)
        print("all extracted answers:",all_extracted_answers)
        if not list_of_messages:
            break
        #list_of_messages = batch_message_execute(list_of_messages,round_idx)
    answer = select_answer(all_extracted_answers)
    print("answer:",answer)
    correct_answer = get_correct_answer(question)
    print("correct answer:",correct_answer)
    g_count += 1
    if str(answer) == str(correct_answer):
        g_score += 1

    print(f"score: {g_score}/{g_count}")
    print("\n\n")
    return answer


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    print("------")
    print(id_)
    
    question = question.item(0)
    answer = predict_for_question(question)
    print(question)
    print("------\n\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            'reference.csv',
            # '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
        )
    )

