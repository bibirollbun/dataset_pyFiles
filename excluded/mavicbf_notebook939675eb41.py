import os
# https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"



import os
import gc
import time
import warnings

import pandas as pd
import polars as pl
import numpy as np

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 53) * 60



# from vllm import LLM, SamplingParams

# warnings.simplefilter('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# llm_model_pth = '/kaggle/input/m/shelterw/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b-awq/1'

# MAX_NUM_SEQS = 16
# MAX_MODEL_LEN = 4096 * 3

# llm = LLM(
#     llm_model_pth,
#     # dtype="half",                # The data type for the model weights and activations
#     max_num_seqs=MAX_NUM_SEQS,   # Maximum number of sequences per iteration. Default is 256
#     max_model_len=MAX_MODEL_LEN, # Model context length
#     trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
#     tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
#     gpu_memory_utilization=0.98, # The ratio (between 0 and 1) of GPU memory to reserve for the model
#     seed=391,
#     enable_prefix_caching=True,
# )


# import vllm
# print(vllm.__version__)


# tokenizer = llm.get_tokenizer()


# import re
# import keyword


# def extract_boxed_text(text):
#     pattern = r'oxed{(.*?)}'
#     matches = re.findall(pattern, text)
#     if not matches:
#         return ""
#     for match in matches[::-1]:
#         if match != "":
#             return match
#     return ""


# from collections import Counter
# import random
# def select_answer(answers):
#     counter = Counter()
#     for answer in answers:
#         try:
#             if int(answer) == float(answer):
#                 counter[int(answer)] += 1 + random.random() / 1_000
#         except:
#             pass
#     if not counter:
#         return 3
#     _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
#     return answer%1000


# def create_starter_messages(question, index):
#     options = []
#     for _ in range(3):
#         options.append(
#             [
#                 {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
#                 {"role": "user", "content": question + ' Return final answer within \boxed{}, after taking modulo 1000.'},
#             ]
#         )
#     for _ in range(2):
#         options.append(
#             [
#                 {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
#                 {"role": "user", "content": question},
#             ]
#         )
#     for _ in range(1):    
#         options.append(
#             [
#                 {"role": "system", "content": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
#                 {"role": "user", "content": question},
#             ],
#         )

#     options.append(
#         [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a mathematical genius. "
#                     "Explain each step clearly and verify for consistency. "
#                     "Give your final solution in \\boxed{} with the result mod 1000."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": question
#             }
#         ]
#     )

#     # 11
#     options.append(
#         [
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a highly accurate math assistant. "
#                     "Pay special attention to possible pitfalls. "
#                     "Your final answer must be in \\boxed{}, and taken modulo 1000."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": question
#             }
#         ]
#     )
#     return options[index%len(options)]


from math import ceil

def predict_for_question(question: str) -> int:
    return 0
    # question_start_time = time.time()
    # if time.time() > cutoff_time:
    #     return 3
    # selected_questions_only = True
    # if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    #     # if "Triangle" not in question:
    #     return 210
    # print(question)
    
    # if time.time() > cutoff_times[-1]:
    #     print(time.time(), cutoff_times[-1])
    #     print("SPEEDUP!!!")
    #     num_of_iters = 2
    #     num_of_seqs=6
    # else:
    #     num_of_iters = 3
    #     num_of_seqs=6


    # messages = [create_starter_messages(question, index) for index in range(num_of_seqs)]
    # list_of_texts = [
    #     tokenizer.apply_chat_template(
    #         conversation=message,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
    #     for message in messages
    # ]
    # old = list_of_texts
    # new = list_of_texts
    # res = []

    # for k in range(num_of_iters):
    #     # if k == 1:
    #     sampling_params = SamplingParams(
    #         temperature=0.7,              # randomness of the sampling
    #         min_p=0.05,
    #         top_p=0.9,
    #         skip_special_tokens=True,     # Whether to skip special tokens in the output
    #         max_tokens=8192//2 ,
        #     stop=["</think>"]
        # )
        # outputs = llm.generate(old,  sampling_params=sampling_params,)
        # new = []

        # new_old = []
        # for i in range(len(old)):
        #     if extract_boxed_text(outputs[i].outputs[0].text):
        #         res.append(old[i] + outputs[i].outputs[0].text)
        #     else:
        #         new_old.append(old[i] + outputs[i].outputs[0].text)       

        # if num_of_iters == 3 and k == 1 and len(new_old) >= 7:
        #     curr_num = min(7, len(new_old))
            
        #     new_old = random.sample(new_old, curr_num)
        # print(list(extract_boxed_text(x) for x in res))
        # if k == 0:
        #     repeats = 2
        # elif k == 1:
        #     repeats = 1
        # for i in range(len(new_old)):
        #     for _ in range(repeats):
        #         new.append(new_old[i])
    #     old = new

    #     print(list(extract_boxed_text(x) for x in res))

    #     if len(res) >= 8:
    #         numbers = []
    #         for r in res:
    #             try:
    #                 num = int(extract_boxed_text(r))
    #                 numbers.append(num)
    #             except ValueError:
    #                 pass
    #         # Если удалось собрать 7+ числовых ответов
    #         if len(numbers) >= 8:
    #             counter = Counter(numbers)
    #             total = len(numbers)
                
    #             # Проверяем, не встречается ли какое-то число
    #             # в более чем 80% случаев (c округлением вверх)
    #             for num, cnt in counter.items():
    #                 if cnt >= ceil(total * 0.8):
    #                     cutoff_times.pop()
    #                     question_end_time = time.time()
    #                     print(num)
    #                     print("TIME FOR QUESTION: ", question_end_time - question_start_time)
    #                     print("PRE RESULT!!!")
    #                     return num
            
    # answer = select_answer(list(extract_boxed_text(x) for x in res))
    # print(answer)
    # cutoff_times.pop()
    # question_end_time = time.time()
    # print("TIME FOR QUESTION: ", question_end_time - question_start_time)
    # return answer


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


# predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?")


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


cutoff_times = [int(x) for x in np.linspace(cutoff_time, time.time() + 500, 50)]


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
#             '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            'reference.csv',
        )
    )




