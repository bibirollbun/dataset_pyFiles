import time
import os

notebook_start_time = time.time()


!pip install triton --no-deps --no-index --find-links file:///kaggle/input/sglang-download/packages

!pip install "sglang[all]" --no-index --find-links file:///kaggle/input/sglang-download/packages -U

!pip install lmdeploy --no-index --find-links file:///kaggle/input/lmdeploy-download/packages



print(f'time install packages {time.time() - notebook_start_time}')


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
# os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# # Force FlashInfer
# os.environ["VLLM_ATTENTION_BACKEND"]='FLASHINFER'
# os.environ["VLLM_USE_FLASHINFER_SAMPLER"]='1'
# os.environ["VLLM_FLASHINFER_FORCE_TENSOR_CORES"]='1'

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    problem_num = 50
else:
    problem_num = 10


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import re
import time
import random
import warnings
from collections import Counter
import numpy as np, pandas as pd, polars as pl

# import torch
# import vllm
# from vllm import LLM, SamplingParams

import kaggle_evaluation.aimo_2_inference_server

warnings.simplefilter('ignore')
# print('PyTorch version:', torch.__version__)
# print('vLLM:', vllm.__version__)


# def seed_everything(seed):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = True
# seed_everything(seed=0)

# notebook_start_time = time.time()
# cutoff_time = notebook_start_time + (4 * 60 + 45) * 60
# cutoff_times = [int(x) for x in np.linspace(cutoff_time, notebook_start_time + 60 * 60, 50 + 1)]


if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    cutoff_time = notebook_start_time + (4 * 60 + 55) * 60
else:
    cutoff_time = notebook_start_time + (55) * 60




# if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     llm_model_pth = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1'
# else:
#     llm_model_pth = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'


# llm_model = '/kaggle/input/deepseek-r1/transformers/casperhansen-distill-qwen-14b-awq/1'
llm_model = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b-awq-v2/1'
# llm_model = '/kaggle/input/deepseek-r1/transformers/14b-awq-v2-g128-s64/1'

MAX_NUM_SEQS = 10
# MAX_MODEL_LEN = 8192 * 3 // 2
MAX_MODEL_LEN = 13300




infer_bs = MAX_NUM_SEQS
seq_len = (32768, MAX_MODEL_LEN)
# temperature = 0.6
# top_p = 0.95
# min_p = 0.0
hard_problems_only = False       # for testing worst case throughput
quick_commit_run = False         # for saving gpu quota when commit
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    hard_problems_only = False
    quick_commit_run = False
# max_num_batched_tokens = 32768
# max_num_seqs = 256
# no_system_prompt = False
# prompt_weight = (12, 3)
quick_run_thres = 280
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    quick_run_thres = 20
quick_run_seqlen = 12000
quick_run_bs = 6
early_stop_min_tokens_num = 50000


# llm = LLM(
#     llm_model_pth,
# #    dtype="half",                 # The data type for the model weights and activations
#     max_num_seqs=MAX_NUM_SEQS,    # Maximum number of sequences per iteration. Default is 256
#     max_model_len=MAX_MODEL_LEN,  # Model context length
#     trust_remote_code=True,       # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
#     tensor_parallel_size=4,       # The number of GPUs to use for distributed execution with tensor parallelism
#     gpu_memory_utilization=0.85,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
#     seed=2024,
# )

# tokenizer = llm.get_tokenizer()


# from vllm import EngineArgs
# engine_args = EngineArgs(
#     model=llm_model,
#     max_model_len=seq_len[0],
#     tokenizer=llm_model,
#     trust_remote_code=True,
#     tensor_parallel_size=4,
#     gpu_memory_utilization=0.90,
#     # disable_custom_all_reduce=True,
#     # enable_chunked_prefill = True,
#     # max_num_batched_tokens=max_num_batched_tokens,
#     enable_prefix_caching = True,
#     # # enforce_eager = True,
#     # disable_log_stats=False,
#     seed=2024,
# )

# from vllm.v1.engine.llm_engine import LLMEngine
# engine = LLMEngine.from_engine_args(engine_args)

# tokenizer = engine.tokenizer.get_lora_tokenizer()


from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

backend_config = TurbomindEngineConfig(tp=4, cache_max_entry_count=0.9)

pipe = pipeline(llm_model, backend_config=backend_config)


print(f'time init {time.time() - notebook_start_time}')


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""

def batch_message_filter(list_of_messages) -> tuple[list[list[dict]], list[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers

def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000

# def batch_message_generate(list_of_messages) -> list[list[dict]]:
#     max_tokens = MAX_MODEL_LEN
#     if time.time() > cutoff_times[-1]:
#         print("Speedrun")
#         max_tokens = 2 * MAX_MODEL_LEN // 3

#     sampling_params = SamplingParams(
#         temperature=1.0,               # Randomness of the sampling
#         top_p=0.90,                    # Cumulative probability of the top tokens to consider
#         min_p=0.05,                    # Minimum probability for a token to be considered
#         skip_special_tokens=True,      # Whether to skip special tokens in the output
#         max_tokens=max_tokens,         # Maximum number of tokens to generate
#         stop=["</think>"],             # List of strings that stop the generation
#         seed=777,
#     )
    
#     list_of_texts = [
#         tokenizer.apply_chat_template(
#             conversation=messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         for messages in list_of_messages
#     ]

#     request_output = llm.generate(
#         prompts=list_of_texts,
#         sampling_params=sampling_params,
#     )
#     print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

#     sort_keys_and_list_of_messages = []
#     for messages, single_request_output in zip(list_of_messages, request_output):
#         #print()
#         #print(single_request_output.outputs[0].text)
#         #print()
#         messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

#         sort_keys_and_list_of_messages.append(
#             (
#                 len(single_request_output.outputs[0].token_ids),
#                 messages
#             )
#         )
#     print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
#     sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])
#     print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    
#     list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]
#     return list_of_messages


def create_starter_messages(question, index):
    
    options = []
    for _ in range(5):
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": question},
            ]
        )
    for _ in range(1):    
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
                {"role": "user", "content": question},
            ],
        )
    return options[index%len(options)]



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
    return -1

def check_early_stop(all_extracted_answers, question_bs):
    if not all_extracted_answers:
        return False
    all_extracted_answers = [int(answer) % 1000 for answer in all_extracted_answers]
    
    can_determine_answer = False
    # if question_bs == 3 and temperature == 0.0:
    #     print(f'receive first answer, early stopping')
    #     can_determine_answer = True

    counter = Counter(all_extracted_answers)
    cnts = list(counter.values())
    cnts.sort(reverse=True)

    # exclude 0
    if cnts[0] == counter[0]:
        return False
    
    if question_bs == 5:
        if cnts[0] - (len(all_extracted_answers) - cnts[0]) >= 2:
            print(f'can determine answer now, early stopping')
            can_determine_answer = True
    
    if len(all_extracted_answers) >= 3:
        if question_bs <= 5:
            if cnts[0] >= (question_bs + 1) // 2:
                print(f'can determine answer now, early stopping')
                can_determine_answer = True
        elif cnts[0] >= len(all_extracted_answers) * min(1, (0.8 - (len(all_extracted_answers) - 5) / (infer_bs - 5) * 0.3)):
            print(f'can determine answer now, early stopping')
            can_determine_answer = True
    return can_determine_answer


# from vllm import SamplingParams
from collections import Counter

question_cnt = 0
correct_cnt = 0
total_reqs = 0

def predict_for_question(question: str) -> int:
    # selected_questions_only = True
    # #selected_questions_only = False
    # if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    #     #if "Triangle" not in question:
    #     #    return 210
    #     if "Triangle" not in question and "delightful" not in question and "George" not in question:
    #         return 210

    if hard_problems_only:
        if get_correct_answer(question) not in (3, 79, 751):
            print(f'hard question only, skipping current question')
            return 210

    if quick_commit_run:
        if get_correct_answer(question) not in (250, 180, 79):
            print(f'quick commit run, skipping current question')
            return 210
    
    if time.time() > cutoff_time:
        return 210
    
    # print(question)

    # num_seqs = MAX_NUM_SEQS
    # if time.time() > cutoff_times[-1]:
    #     num_seqs = 2 * MAX_NUM_SEQS // 3
    
    global question_cnt, correct_cnt, total_reqs
    # question += "\nIf the final answer is a number larger than 1000, take modulo 1000. "
    print(question)
    question_cnt += 1
    question_start_time = time.time()
    remain_avg_time = (cutoff_time - question_start_time) / (problem_num - question_cnt + 1)
    print(f'remain avg time {remain_avg_time}')
    question_cutoff_time = min(question_start_time + 600, cutoff_time)
    
    max_tokens = seq_len[1]
    # max_tokens = 100
    question_bs = infer_bs
    if remain_avg_time < quick_run_thres:
        print(f'quick run, using seqlen {quick_run_seqlen}, bs {quick_run_bs}')
        max_tokens = quick_run_seqlen
        question_bs = quick_run_bs


    
    # list_of_messages = [create_starter_messages(question, index) for index in range(question_bs)]
    # all_input_texts = [
    #     tokenizer.apply_chat_template(
    #         conversation=message,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     ) for message in list_of_messages
    # ]
    
    all_extracted_answers = []


    gen_config = GenerationConfig(top_p=0.90,
                                  # top_k=30,
                                  min_p=0.05,
                                  temperature=0.9,
                                  max_new_tokens=max_tokens,
                                  # max_new_tokens=100,
                                  do_sample=True,
                                  stop_words=['</think>'],
                                 )
    
    prompts = [create_starter_messages(question, i) for i in range(question_bs)]
    # for item in pipe.stream_infer(prompts, gen_config=gen_config):
    #     print(item)
    # print(f'prompts {prompts}')
    
    responses = pipe(prompts, gen_config=gen_config)
    # print(f'responses {responses}')
    
    output_text = [response.text for response in responses]
    output_tokens_num = [len(response.token_ids) for response in responses]

    all_extracted_answers = [extract_boxed_text(text) for text in output_text]


    
    # for _ in range(1):
    #     list_of_messages = batch_message_generate(list_of_messages)
        
    #     if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    #         df = pd.DataFrame(
    #             {
    #                 "question": [question] * len(list_of_messages),
    #                 "message": [messages[-1]["content"] for messages in list_of_messages],
    #             }
    #         )
    #         df.to_csv(f"{str(int(time.time() - notebook_start_time)).zfill(5)}.csv", index=False)
        
    #     list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
    #     all_extracted_answers.extend(extracted_answers)

    # get answers from vllm engine

    
    # all_results_generators = []
    # running_reqs = []
    
    # for req_id in range(total_reqs, total_reqs + question_bs):
    #     engine.add_request(
    #         str(req_id),
    #         all_input_texts[(req_id - total_reqs) % (len(all_input_texts))],
    #         SamplingParams(
    #             temperature=0.6,               # Randomness of the sampling
    #             top_p=0.95,                    # Cumulative probability of the top tokens to consider
    #             min_p=0.00,                    # Minimum probability for a token to be considered
    #             skip_special_tokens=True,      # Whether to skip special tokens in the output
    #             max_tokens=max_tokens,         # Maximum number of tokens to generate
    #             stop=["</think>"],             # List of strings that stop the generation
    #             seed=777,
    #         )
    #     )
    #     running_reqs.append(req_id)

    # total_reqs += question_bs


    # output_text = [''] * question_bs
    # output_tokens_num = [0] * question_bs

    # last_step_call_time = 0
    # cur_step = 0
    # can_determine_answer = False
    # early_stop_first_check_finished = False
    
    # # scheduler loop
    # while True:
    #     cur_step += 1
    #     # if cur_step % 100 == 0:
    #     #     engine.do_log_stats()

    #     # early stop check
    #     if not early_stop_first_check_finished and max(output_tokens_num) >= early_stop_min_tokens_num:
    #         early_stop_first_check_finished = True
    #         if all_extracted_answers and check_early_stop(all_extracted_answers, question_bs):
    #             can_determine_answer = True

        
    #     # check timeout / can determine answer
    #     if time.time() > question_cutoff_time or can_determine_answer:
    #         before_abort_time = time.time()
    #         if can_determine_answer:
    #             for req_id in running_reqs:
    #                 print(f'can determine answer, aborting req {req_id}')
    #                 engine.abort_request(str(req_id))
    #         else:
    #             for req_id in running_reqs:
    #                 print(f'timeout, aborting req {req_id}')
    #                 engine.abort_request(str(req_id))
    #         print(f'time abort requests: {time.time() - before_abort_time}')
    #         break

    #     # step
    #     # print(f'time between step calls: {time.time() - last_step_call_time}')
    #     request_outputs = engine.step()
    #     # last_step_call_time = time.time()

    #     # update text
    #     for request_output in request_outputs:
    #         req_id = int(request_output.request_id)
    #         # print(f'req_id {req_id}, request_output {request_output}')
    #         text = request_output.outputs[0].text
    #         output_text[req_id - total_reqs + question_bs] = text
    #         tokens_num = len(request_output.outputs[0].token_ids)
    #         output_tokens_num[req_id - total_reqs + question_bs] = tokens_num
    #         if request_output.finished:
    #             engine.abort_request(str(req_id))
    #             answer = extract_boxed_text(text)
    #             # if answer and int(answer) == float(answer):
    #             skip_check = True
    #             try:
    #                 if int(answer) == float(answer):
    #                     skip_check = False
    #             except:
    #                 pass
    #             if not skip_check:
    #                 all_extracted_answers.append(answer)
    #                 print(f'receive answer {answer} from request {req_id} ({req_id - total_reqs + question_bs})')
    #                 # check if early stop
    #                 if max(output_tokens_num) >= early_stop_min_tokens_num and check_early_stop(all_extracted_answers, question_bs):
    #                     can_determine_answer = True
    #                 # if len(all_extracted_answers) >= 3:
    #                 #     cnts = list(Counter(all_extracted_answers).values())
    #                 #     cnts.sort(reverse=True)
    #                 #     if cnts[0] >= len(all_extracted_answers) * (0.8 - (len(all_extracted_answers) - 5) / (infer_bs - 5) * 0.3):
    #                 #         print(f'can determine answer now, early stopping')
    #                 #         can_determine_answer = True
    #             if req_id in running_reqs:
    #                 running_reqs.remove(req_id)

    #     # check answer obtained
    #     if cur_step % 10 == 0:
    #         for req_id in running_reqs:
    #             # text = output_text[req_id - total_reqs + question_bs]
    #             answer = extract_boxed_text(text)
    #             # if answer and int(answer) == float(answer):
    #             skip_check = True
    #             try:
    #                 if int(answer) == float(answer):
    #                     skip_check = False
    #             except:
    #                 pass
    #             if not skip_check:
    #                 all_extracted_answers.append(answer)
    #                 print(f'receive answer {answer} from request {req_id} ({req_id - total_reqs + question_bs})')
    #                 engine.abort_request(str(req_id))
    #                 if req_id in running_reqs:
    #                     running_reqs.remove(req_id)
    #                 # check if early stop
    #                 if max(output_tokens_num) >= early_stop_min_tokens_num and check_early_stop(all_extracted_answers, question_bs):
    #                     can_determine_answer = True
                
                    
    #     if not running_reqs or not engine.has_unfinished_requests():
    #         break

    # # for text in output_text:
    # #     print(text)

    print(f'output num tokens {output_tokens_num}')
    print(f'sorted {sorted(output_tokens_num)}')

    question_time = time.time() - question_start_time
    print(f'question time: {question_time}s')
    print(f'avg tokens/s {sum(output_tokens_num) / question_time}')

    # save texts
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        import datetime
        qid = get_correct_answer(question)
        for i, text in enumerate(output_text):
            text_filename = f'qid_{qid}_req_id_{i}_model_{llm_model.split("/")[-2]}_' + datetime.datetime.now().strftime(f'%Y-%m-%d_%H-%M')
            text_filename += '.txt'
            os.system(f'touch {text_filename}')
            with open(text_filename, 'w') as f:
                print(prompts[i], file=f)
                print('===', file=f)
                print(text, file=f)
                print('===', file=f)
                print(output_tokens_num[i], file=f)

    # print(all_extracted_answers)
    # answer = select_answer(all_extracted_answers)
    # print(answer)

    print(f'all extracted_answers: {all_extracted_answers}')
    
    answer = select_answer(all_extracted_answers)
    print(f'answer: {answer}')
    print(f'correct answer: {get_correct_answer(question)}')
    if answer == get_correct_answer(question):
        correct_cnt += 1
    print(f'current correct: {correct_cnt} / {question_cnt}')

    
    print("\n\n")
    # cutoff_times.pop()
    return answer


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_ = id_.item(0)
    print("------")
    print(id_)
    question = question.item(0)
    answer = predict_for_question(question)
    print(question)
    print("------\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})


#predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?")


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index=False)


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
#            '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            'reference.csv',
        )
    )




