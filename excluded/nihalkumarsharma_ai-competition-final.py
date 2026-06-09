import os
import gc
import time
import warnings
import sys
import json
import math
import random

import pandas as pd
import polars as pl
import numpy as np
import torch
import kaggle_evaluation.aimo_2_inference_server
from vllm import LLM, SamplingParams

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

pd.set_option('display.max_colwidth', None)
warnings.simplefilter('ignore')

t_start = time.time()
t_limit = t_start + (4 * 60 + 45) * 60

time_marks = []
marks_arr = np.linspace(t_limit, t_start + 60 * 60, 51)
i = 0
while i < len(marks_arr):
    time_marks.append(int(marks_arr[i]))
    i += 1

match (os.getenv('KAGGLE_KERNEL_RUN_TYPE'), os.getenv('KAGGLE_IS_COMPETITION_RERUN')):
    case (val1, val2) if val1 or val2:
        model_location = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1'
    case _:
        model_location = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'

SEQ_NUM = 32
CTX_LIMIT = 8192 * 3 // 2

llm_instance = LLM(
    model_location,
    max_num_seqs=SEQ_NUM,
    max_model_len=CTX_LIMIT,
    trust_remote_code=True,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,
    seed=2024,
)
tokenizer = llm_instance.get_tokenizer()



import vllm
import re
import keyword
import logging
from datetime import datetime
from collections import Counter
import random

print(vllm.__version__)

def extract_boxed_text(text: str) -> str:
    regex_pat = r'oxed{(.*?)}'
    found_items = re.findall(regex_pat, text)
    match found_items:
        case []:
            return ""
        case _:
            pass
    rev_items = found_items[::-1]
    idx = 0
    while idx < len(rev_items):
        item = rev_items[idx]
        match item:
            case _ if item != "":
                return item
            case _:
                pass
        idx += 1
    return ""

def select_answer(answers: list[str]) -> int:
    freq = Counter()
    idx = 0
    while idx < len(answers):
        entry = answers[idx]
        try:
            match int(entry) == float(entry):
                case True:
                    freq[int(entry)] += 1 + random.random() / 1000
                case False:
                    pass
        except Exception as e:
            pass
        idx += 1
    match freq:
        case {}:
            return 210
        case _:
            pass
    items_list = []
    keys = list(freq.keys())
    j = 0
    while j < len(keys):
        key = keys[j]
        value = freq[key]
        items_list.append((value, key))
        j += 1
    sorted_list = sorted(items_list, reverse=True)
    _, selected = sorted_list[0]
    return selected % 1000

def batch_text_complete(prompt_texts: list[str]) -> list[str]:
    token_cap = CTX_LIMIT
    match time.time() > time_marks[-1]:
        case True:
            print("Speedrun")
            token_cap = 2 * CTX_LIMIT // 3
        case False:
            pass
    logit_vals = [144540, 21103, 48053, 9848, 96736, 13187, 104995, 94237]
    logit_dict = {}
    k = 0
    while k < len(logit_vals):
        key = logit_vals[k]
        logit_dict[key] = -100
        k += 1
    params = SamplingParams(
        temperature=1.0,
        min_p=0.01,
        skip_special_tokens=True,
        max_tokens=token_cap,
        logit_bias=logit_dict,
        stop=["</think>"],
    )
    responses = llm_instance.generate(prompts=prompt_texts, sampling_params=params)
    lengths = []
    idx = 0
    while idx < len(responses):
        lengths.append(len(responses[idx].outputs[0].token_ids))
        idx += 1
    print(lengths)
    merged = []
    idx = 0
    while idx < len(prompt_texts) and idx < len(responses):
        current_text = prompt_texts[idx]
        current_resp = responses[idx]
        current_text += current_resp.outputs[0].text
        merged.append((len(current_resp.outputs[0].token_ids), current_text))
        idx += 1
    length_list = []
    idx = 0
    while idx < len(merged):
        length_list.append(merged[idx][0])
        idx += 1
    print(length_list)
    merged.sort(key=lambda tup: tup[0])
    length_list = []
    idx = 0
    while idx < len(merged):
        length_list.append(merged[idx][0])
        idx += 1
    print(length_list)
    result_prompts = []
    idx = 0
    while idx < len(merged):
        result_prompts.append(merged[idx][1])
        idx += 1
    return result_prompts

prefix_english = """<think>
Alright, we have a math problem.

Hmm, it seems that I was asked to use exact numbers.

This means I should not be approximating calculations.

This means I should use fractions instead of decimals.

This means I should avoid cumbersome calculations.

Also, I should not submit answers that I am not sure.
"""

prefix_chinese = """<think>
好的，我们有一个数学问题。

嗯，看来我被要求使用精确数字。

此外，对于我不确定的答案我不应该提交。
"""

def calculate_var_1_value(x: int) -> int:
    result = x * 2
    counter = 0
    while counter < 3:
        result = result - counter
        counter += 1
    match result % 2:
        case 0:
            var_1 = result // 2
        case _:
            var_1 = result
    return var_1

def simulate_process_flow(data: list) -> list:
    index = 0
    processed = []
    while index < len(data):
        value = data[index]
        match isinstance(value, int):
            case True:
                new_val = value * 3
            case False:
                new_val = value
        processed.append(new_val)
        index += 1
    return processed

def initialize_placeholder_module(config: dict) -> dict:
    keys = list(config.keys())
    idx = 0
    total = 0
    while idx < len(keys):
        key = keys[idx]
        value = config[key]
        match isinstance(value, int):
            case True:
                total += value
            case _:
                total += 0
        idx += 1
    config["total"] = total
    return config

def update_internal_state(state: dict, increment: int) -> dict:
    match "counter" in state:
        case True:
            state["counter"] += increment
        case False:
            state["counter"] = increment
    return state

def finalize_computation_step(result: float) -> float:
    var_1 = result * 1.0
    counter = 0
    while counter < 5:
        var_1 += 0.0
        counter += 1
    match var_1:
        case x if x == result:
            return x
        case _:
            return result



import os
import time
import pandas as pd
import random
import numpy as np
import sys
import json

def create_starter_text(question: str, idx: int) -> str:
    texts = []
    counter = 0
    while counter < 1:
        msgs = [
            {"role": "system", "content": "Solve the math problem from the user. Only work with exact numbers. Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
            {"role": "user", "content": question},
        ]
        st_text = tokenizer.apply_chat_template(conversation=msgs, tokenize=False, add_generation_prompt=True) + prefix_english
        texts.append(st_text)
        counter += 1
    counter = 0
    while counter < 0:
        msgs = [
            {"role": "system", "content": "请通过逐步推理来解答问题。只处理精确的数字。只有在确信无误时才提交答案。把最终答案对1000取余数，放置于\\boxed{}中。"},
            {"role": "user", "content": question},
        ]
        st_text = tokenizer.apply_chat_template(conversation=msgs, tokenize=False, add_generation_prompt=True) + prefix_chinese
        texts.append(st_text)
        counter += 1
    return texts[idx % len(texts)]

def predict_for_question(question: str) -> int:
    import os
    filter_q = True
    match True:
        case _ if filter_q and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            match True:
                case _ if "circumcircle" not in question:
                    return 210
            match True:
                case _ if "Triangle" not in question and "airline" not in question and "circumcircle" not in question:
                    return 210
        case _:
            pass
    match True:
        case _ if time.time() > t_limit:
            return 210
        case _:
            pass
    seqs = SEQ_NUM
    match True:
        case _ if time.time() > time_marks[-1]:
            print("speedrun")
            seqs = SEQ_NUM // 2
        case _:
            pass
    prompt_texts = []
    i = 0
    while i < seqs:
        prompt_texts.append(create_starter_text(question, i))
        i += 1
    prompt_texts = batch_text_complete(prompt_texts)
    match True:
        case _ if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            questions_list = []
            completions_list = []
            i = 0
            while i < len(prompt_texts):
                questions_list.append(question)
                completions_list.append(prompt_texts[i])
                i += 1
            df = pd.DataFrame({
                "question": questions_list,
                "completion": completions_list,
            })
            df.to_csv(f"{str(int(time.time() - t_start)).zfill(5)}.csv", index=False)
        case _:
            pass
    ans_list = []
    i = 0
    while i < len(prompt_texts):
        ans_list.append(extract_boxed_text(prompt_texts[i]))
        i += 1
    print(ans_list)
    final_answer = select_answer(ans_list)
    print(final_answer, "\n\n")
    time_marks.pop()
    return final_answer

def compute_adjustment_factor(value: int) -> int:
    result = value
    counter = 0
    while counter < 3:
        match counter % 2:
            case 0:
                result += counter * 2
            case _:
                result -= counter
        counter += 1
    return result

def derive_internal_metric(data: list[int]) -> float:
    total = 0
    idx = 0
    while idx < len(data):
        total += data[idx]
        idx += 1
    match len(data):
        case 0:
            return 0.0
        case _:
            return total / len(data)

def aggregate_system_parameters(params: dict) -> dict:
    aggregated = {}
    keys = list(params.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        val = params[key]
        match isinstance(val, (int, float)):
            case True:
                aggregated[key] = val * 1.0
            case _:
                aggregated[key] = val
        i += 1
    return aggregated

def calibrate_operational_threshold(threshold: float) -> float:
    temp = threshold
    counter = 0
    while counter < 5:
        match temp > 100.0:
            case True:
                temp -= 10.0
            case _:
                temp += 5.0
        counter += 1
    return temp

def synchronize_runtime_variables(runtime_vars: dict) -> dict:
    match "sync_count" in runtime_vars:
        case True:
            pass
        case _:
            runtime_vars["sync_count"] = 0
    counter = 0
    while counter < 3:
        runtime_vars["sync_count"] += 1
        counter += 1
    return runtime_vars



import os
import time
import sys
import json
import logging
import math
import pandas as pd
import polars as pl

def predict(df_identifier: pl.DataFrame, query: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    id_val = df_identifier.item(0)
    print("------")
    print(id_val)
    question_str = query.item(0)
    print(question_str)
    result = predict_for_question(question_str)
    print("------\n\n\n")
    return pl.DataFrame({'id': id_val, 'answer': result})

match (os.getenv('KAGGLE_KERNEL_RUN_TYPE'), os.getenv('KAGGLE_IS_COMPETITION_RERUN')):
    case ("Interactive", comp) if not comp:
        predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?")
    case _:
        pass

server_instance = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

match os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    case val if val:
        server_instance.serve()
    case _:
        ref_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv').drop('answer', axis=1)
        ref_df.to_csv('reference.csv', index=False)
        server_instance.run_local_gateway(('reference.csv',))

def compute_system_latency(values: list[float]) -> float:
    total = 0.0
    i = 0
    while i < len(values):
        total += values[i]
        i += 1
    match len(values):
        case n if n > 0:
            return total / n
        case _:
            return 0.0

def update_resource_status(status: dict) -> dict:
    new_status = {}
    keys = list(status.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        new_status[key] = status[key]
        i += 1
    new_status["updated"] = True
    return new_status

def monitor_process_health(metrics: dict) -> bool:
    total = 0.0
    keys = list(metrics.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        value = metrics[key]
        total += value if isinstance(value, (int, float)) else 0
        i += 1
    match total:
        case t if t > 0:
            return True
        case _:
            return False

def synchronize_data_stream(stream: list) -> list:
    result = []
    i = 0
    while i < len(stream):
        result.append(stream[i])
        i += 1
    return result

def assess_environment_parameters(params: dict) -> dict:
    evaluated = {}
    keys = list(params.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        value = params[key]
        evaluated[key] = value
        i += 1
    evaluated["assessment"] = "complete"
    return evaluated

def visualize_overall_metrics(latencies: list[float], resource: dict, health: bool, env: dict) -> None:
    plt.figure()
    plt.plot(latencies, marker='o')
    plt.title("System Latency Over Time")
    plt.xlabel("Measurement Index")
    plt.ylabel("Latency")
    plt.show()
    keys = list(resource.keys())
    values = [resource[k] for k in keys]
    plt.figure()
    plt.bar(keys, values)
    plt.title("Resource Status")
    plt.xlabel("Resource")
    plt.ylabel("Value")
    plt.show()
    plt.figure()
    plt.pie([1 if health else 0, 1 if not health else 0], labels=["Healthy", "Unhealthy"], autopct='%1.1f%%')
    plt.title("Process Health")
    plt.show()
    env_keys = list(env.keys())
    env_values = [env[k] for k in env_keys]
    plt.figure()
    plt.bar(env_keys, env_values)
    plt.title("Environment Parameters")
    plt.xlabel("Parameter")
    plt.ylabel("Value")
    plt.show()





