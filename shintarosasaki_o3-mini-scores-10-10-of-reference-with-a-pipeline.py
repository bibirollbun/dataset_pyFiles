!pip install --upgrade openai  


# Please install OpenAI SDK first: `pip3 install openai`
# https://platform.deepseek.com/usage
# https://api-docs.deepseek.com/guides/reasoning_model
# https://api-docs.deepseek.com/guides/multi_round_chat
# https://zenn.dev/atoka/articles/ac48928977ce93
import os
from repls import PythonREPL

import json

import re
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)
import nest_asyncio
nest_asyncio.apply()
from stopwatch import Stopwatch,printJST
import asyncio
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

printJST()
#MAX_NUM_SEQS = 16
MAX_NUM_SEQS = 6

# warnings.simplefilter('ignore')

OVERALL_WATCH=Stopwatch()
OVERALL_TOKENS=0
def default_answer()->int:
    # random.randint(0, 999)
    return 999


!pip show openai


#############################
## Async Openai API
#############################
from openai import OpenAI
import openai
# 
from stopwatch import Stopwatch
from typing import List, Dict, TypeAlias
Conversations:TypeAlias = list[list[dict]]
from tqdm.asyncio import tqdm
API_KEY = UserSecretsClient().get_secret("OpenAI")
url="https://api.openai.com/v1"

MAX_MODEL_LEN = 20000
params={
    # "model":"o1" ,
    "model":"o3-mini" ,        
    # "model":"o1-preview" ,
    # "model":"gpt-4.5-preview" ,
    # "model":"o1-mini" ,        
    "max_completion_tokens": MAX_MODEL_LEN,
    "temperature": 1,
    # "temperature": 0.0,
    "temperature": 1.0,
    # "reasoning_effort": "medium",
    # "reasoning_effort": "high",
    "n": 1,
}
print(params["model"])
print("\033[1;31m",params["model"],"\033[0m")
async def generate_text_async(client,messages,delay):

    await asyncio.sleep(delay)  
    # print("models:",client.models.list())

    # response = await client.chat.completions.create(
    #     messages=messages,
    #     stream=False,
    #     **PARAMS,
    # )
    # # return response.choices[0].message.content
    # return response
    retry_attempts = 5 
    for attempt in range(retry_attempts):
        try:
            # print("params:\n",PARAMS)
            response = await client.chat.completions.create(
                messages=messages,
                stream=False,
                # temperature=0.0,
                **params,
            )
            return response
        except openai.RateLimitError as e:
            wait_time = random.uniform(1.5, 5)  # 1.5〜5秒ランダムに待つ
            print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
            await asyncio.sleep(wait_time)  # 待ってからリトライ
        except Exception as e:
            print(f"Unexpected error: {e}")
            break  # 予期しないエラーの場合はループを抜ける
    return None  # 失敗した場合は None を返す

# https://platform.openai.com/settings/organization/usage
async def main(convos:Conversations):
    async with openai.AsyncClient(api_key=API_KEY, base_url=url) as client:
        tasks = [generate_text_async(client,prompt,i * 30) for i, prompt in enumerate(convos)]
        #tasks = [generate_text_async2(client,prompt) for i, prompt in enumerate(convos)]
        # results = await asyncio.gather(*tasks)
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            result = await future
            results.append(result)    
    return results

# # Test
# convos=[
#     [{"role": "user", "content": "最新のAI技術について教えて"}],

#     # [   {"role": "user", "content": "You are a helpful assistant"},
#     #     {"role": "user", "content": "Hello"},
#     # ],
# ]
# sw_gen=Stopwatch()
# responses =asyncio.run(main(convos))    
# sw_gen.print_time()
# print(responses)




#############################
## Pipeline
#############################
def generate_to_flagged(convos:Conversations, response_flags)->Conversations:
    """generate respons to flagged convo only"""
    print("generate_to_flagged start")
    max_tokens = MAX_MODEL_LEN
    # if time.time() > cutoff_times[-1]:
    #     print("Speedrun")
    #     max_tokens = 2 * MAX_MODEL_LEN // 3


    # print("response_flags",response_flags)
    filtered_convos = [convo for convo, flag in zip(convos, response_flags) if flag]
    if not filtered_convos:return convos
    # print("filtered_convos in generate_to_flagged:",filtered_convos)

    sw_gen=Stopwatch()
    responses =asyncio.run(main(convos))    
    sw_gen.print_time()
    # Add responses to the conversations.
    response_idx = 0
    total_tokens = 0
    for convo, flag in zip(convos, response_flags):
        if not flag:continue
        # content=responses[response_idx].outputs[0].text
        try:
            content=responses[response_idx].choices[0].message.content
            total_tokens += responses[response_idx].usage.total_tokens
            convo.append({"role": "assistant", "content": content})
        except Exception as e:
            print("****error occured****",e)
            global DEBUG_OBJ
            DEBUG_OBJ=responses
        response_idx += 1
    global OVERALL_TOKENS
    OVERALL_TOKENS+=total_tokens
    print("Total tokens : ",total_tokens)
    return convos


def exec_pyCode_and_generate(convos: Conversations) -> Conversations:
    """ If last message in each convo contains Python code, extract and execute it.  
    \n If execution completes successfully, request inference to LLM.  
    \n  In an error, regenerate and execute the code up to a predefined number of times. """

    def extract_pyCode(text)->str:
        pattern = r'```python\s*(.*?)\s*```'
        # pattern = r"""```python\s*(.*?)\s*(?:```|''')"""

        matches = re.findall(pattern, text, re.DOTALL)
        code = matches[-1] if matches else ""
        return code

    infer_msg=  ':\n The results are as shown above. Based on these,'
    infer_msg+= ' infer the answer and output the result in LaTeX mathematical format. '
    infer_msg+= ' Ensure the output is in the form of $ \boxed{} $. '    
    infer_msg+= ' Never generate Python code.'
    infer_msg+= ' For any unconfirmed content, make a reasonable guess'
    infer_msg+= ' based on the results.'

    # Loop multiple times to run the regenerated code in error cases.
    response_flags=[False]*len(convos)
    print("in exec_pyCode_and_generate",response_flags)
    for _ in range(2): 
        for idx, convo in enumerate(convos):
     #       print(f"Index {idx}: Last convo type: {type(convo[-1])}, \nValue: {convo[-1]}")
     #       print(f"Content Type: {type(convo[-1].get('content', None))}, Value: {convo[-1].get('content', None)}")
     #       python_code = extract_pyCode(convo[-1].get('content', ""))
            
            # Extract py-code from the last message.
            python_code = extract_pyCode(convo[-1]['content'])
            
            if python_code:
                print("\033[1;31m","code extracted.","\033[0m")
                # Run the code and add the output to the conversation.
                response_flags[idx] = True
                success, output = PythonREPL(timeout=60)(python_code)
                # print("output:\n",output)
                if success: output += infer_msg 
                convo.append({"role": "user", "content": output})
            else:
                print("\033[1;31m","code nothing.","\033[0m")
                # no code, no generate.
                response_flags[idx] = False

        # On success, execute the inference.
        # On error, pass the error details and regenerate the code.
        convos = generate_to_flagged(convos,response_flags)

    return convos



def extract_answers(convos:Conversations) -> list[str]:
    '''extract answer from the responses'''

    def extract_boxed_text(text)->str:
        # pattern = r'boxed{(.*?)}'
        pattern = r'{(.*?)}'
        matches = re.findall(pattern, text)
        default_value="1000"
        if not matches:
            return default_value
        for match in matches[::-1]: #pick the last one
            if match != "":
                return match
        return default_value

    extracted_answers = []
    for convo in convos:
        answer = extract_boxed_text(convo[-1]['content'])
        if answer:extracted_answers.append(answer)

    return extracted_answers

from collections import Counter
import random
def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                if 0 <= int(answer) <= 999 and int(answer) % 1000 > 0:
               #     counter[int(answer)] += 1 + random.random() / 1_000
                    counter[int(answer)] += 1
        except:
            pass
    if not counter:
        return default_answer()
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000



#############################
## Prompt
#############################
def create_starter_messages(question, index)->Conversations:
    options = []
    modulo = r' The final answer I want is that value modulo 1000.'
    def combine(message): return question + modulo + message

    # msg1 = r':\n You are the smartest math expert in the world, please spike this problem and put the answer in \\boxed{}.' 
    # # msg1 +=r' Initiate your response with "<think>\n" at the beginning of every output.'
    # msg1 +=r' Use the Chinese remainder theorem if you need. '
    # for _ in range(0):options.append( [{"role": "user", "content": combine(msg1)}])

    msg2= r""":
  Step 1 **Problem Formulation**:
    Formulate this problem.
    There is no need to solve the problem.
    Change this problem to a mathematical statement.
    Never use any assumptions not stated in the problem statement.
    Do not use any wording from the original problem statement.

  Step 2 **Solution Code**:
    Based on the formulation in Step 1, write a Python code to calculate and solve this problem,
    and put the code between ```python and ```　.
    Do not assume any additional structure or properties that are not explicitly stated in the problem statement.
    In the case of combinatorial problems, prioritize Exhaustive Search over computational efficiency.
    Never use any assumptions not stated in the problem statement.
    Keep all values in fractional form, and Never use float type.
    Make the process of processing and calculations visible.

    """
    # The execution time of the code may be long(e.g., brute force), but ensure the calculation is accurate.
    # For geometric problems, it is assumed that point coordinates and line equations are determined by calculation.
    # Use the Chinese remainder theorem if you need. 
    # You are the smartest math expert in the world.
    # Reflect and verify while reasoning.

    
  #   msg2_1= r""":
  # Step 1 **Problem Formulation**:
  #   Formulate this problem.
  #   There is no need to solve the problem.
  #   Change this problem to a mathematical statement.
  #   Never use any assumptions not stated in the problem statement.
  #   Do not use any wording from the original problem statement.

  # Step 2 **Solution Code**:
  #   You are the smartest math expert in the world.
  #   Based on the formulation in Step 1, write a Python code to calculate and solve this problem,
  #   and put the code between ```python and ```　.
  #   Reflect and verify while reasoning.
  #   Never use any assumptions not stated in the problem statement.
  #   Keep all values in fractional form, and Never use float type.
  #   For geometric problems, it is assumed that point coordinates and line equations are determined by calculation.
  #   Use the Chinese remainder theorem if you need. 
  #   The total execution time should be at most a few minutes.
  #   Make the process of processing and calculations visible.

  #   """
    # Note that for python ranges, stop is exclusive.
    # No need to do the exact calculations, just give me the python code.
    for _ in range(4):options.append( [{"role": "user", "content": combine(msg2)}])

#     msg3= r""":
# Step 1: Mathematical Formulation
# - Convert the problem statement into a mathematical description by deriving the necessary equations, inequalities, and definitions.
# - Explain the formulation using entirely new mathematical expressions, avoiding any phrasing from the original problem.

# Step 2: Algorithm Design and Verification
# - Describe the algorithm based on the mathematical formulation from Step 1.
# - Detail any intermediate calculations, verification methods, or checks that will be used to ensure the correctness of the solution.

# Step 3: Python Code Development
# - Write the Python code based on the algorithm designed in Step 2, placing the code between ```python and ```.
# - Use the Fraction class or similar to maintain all values in fractional form instead of using floats.
# - Include thorough comments within the code to explain the purpose of each section and the algorithms used.
# - Display the intermediate steps of the computation to make the process transparent.
# - No explanation or calculation is needed.

#     """
#     for _ in range(0):options.append( [{"role": "user", "content": combine(msg3)}])


    
    
    return options[index%len(options)]



#############################
## Inference
#############################
def get_label_and_answer(question):
    '''correct answers for reference problems'''
    if 'Three airline' in question: return 'Problem_01',79
    if 'Fred and George' in question: return 'Problem_08',250
    if 'Triangle $ABC$' in question: return 'Problem_03',180
    if 'Find the three' in question: return 'Problem_04',143
    if 'We call a' in question: return 'Problem_07',3
    if 'Let $ABC$ be' in question: return 'Problem_02',751
    if 'For a positive' in question: return 'Problem_09',891
    if 'For positive integers' in question: return 'Problem_06',810
    if 'The Fibonacci numbers' in question: return 'Problem_10',201
    if 'Alice writes all' in question: return 'Problem_05',902
    return '',0

import random
g_score = 0
g_count = 0
def predict_for_question(question: str) -> int:
    print("\n\n------")    

    global g_score
    global g_count
    label,correct_answer = get_label_and_answer(question)

    print(label)
    print(question)
    # Skip probrems to save GPU execution time before submission
    selected_questions_only = True
    selected_questions_only = False
    if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        match label:
            # case 'Problem 3'|'Problem 4'|'Problem 5'|'Problem 7'|'Problem 9'|'Problem 10':
            case 'Problem_02':
                pass
            case _ :
                return default_answer()

    # # Abort inference by total time
    # if time.time() > cutoff_time: return random.randint(0, 999)

    # # Cut off prompts by allotted time per a problem
    num_seqs = MAX_NUM_SEQS
    # if time.time() > cutoff_times[-1]: num_seqs = 2 * MAX_NUM_SEQS // 3

    # Inference
    
    conversations = [create_starter_messages(question, index) for index in range(num_seqs)]
    response_flags = [True] * len(conversations)  # Response for all convos
    # print('*************start-regular process')
    conversations = generate_to_flagged(conversations,response_flags)
    # print('*************start-py process')
    conversations = exec_pyCode_and_generate(conversations)
    extracted_answers = extract_answers(conversations)
    answer = select_answer(extracted_answers)
  
    # output result
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        with open(label.replace(" ", "_")+".json", "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=4)

        print("extracted_answers : ",extracted_answers)
        print("answer : ", answer)
        print("correct answer:",correct_answer)
        g_count += 1
        if str(answer) == str(correct_answer): g_score += 1
        print(f"score: {g_score}/{g_count}")

    print("------\n\n")

    # print("\n\n")
    # cutoff_times.pop()
    return answer

# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
import pandas as pd
import polars as pl
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    question = question.item(0)
    answer = predict_for_question(question)
    return pl.DataFrame({'id': id_, 'answer': answer})



pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


# import kaggle_evaluation.aimo_2_inference_server
# inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
# if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     inference_server.serve()
# else:
#     inference_server.run_local_gateway(
#         (
#             'reference.csv',
#         )
#     )

df = pd.read_csv("reference.csv")
for i in range(len(df)):
    predict_for_question(df.loc[i, "problem"])

print("\n\n------------")
print("OVERALL_TOKEN:","{:,}".format(OVERALL_TOKENS))
OVERALL_WATCH.print_time("Overall ")
printJST()



# # ファイルを読み込む
# with open('/kaggle/working/Problem_02.json', 'r') as file:
#     data = json.load(file)




# idx=0
# print("len:",len(data[idx]))
# print(data[idx][1]['content'])







