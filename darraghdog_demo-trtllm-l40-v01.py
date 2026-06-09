%%capture
# To do - can we move the pip wheels to a /kaggle/working/wheels/ directory, instead of /kaggle/working/
!pip install --no-index --target=/kaggle/working  --find-links file:///kaggle/input/nemo-skills-wheel/wheels/ openai evalplus latex2sympy2_extended math_verify nemo_run hydra-core
!pip install --force-reinstall --upgrade --no-deps --no-index --target=/kaggle/working --find-links file:///kaggle/input/nemo-skills-wheel/wheels/ nemo_skills


import os
import copy
import time
import atexit
import requests
import signal
import subprocess
from typing import Generator, Union
from pathlib import Path
from IPython.display import FileLink, display
from concurrent.futures import ThreadPoolExecutor


import transformers
from openai import Stream
from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.inference.model import get_code_execution_model, get_model
from nemo_skills.prompt.utils import get_prompt, CodeTags
from nemo_skills.inference.model.code_execution import CodeExecutionWrapper


MODEL_DIR = "/kaggle/input/openmath-nemotron-trtllm/tensorrtllm/"
MODEL_DIR_BF16 = f"{MODEL_DIR}/7b-tp4-bf16/1/"
MODEL_DIR_FP8 = f"{MODEL_DIR}/7b-tp4-fp8/1/"
MODEL_DIR_FP8_DRAFT = f"{MODEL_DIR}/7b-tp4-fp8-draft/3/"

host = "127.0.0.1"
port = 5000
kv_cache_free_gpu_memory_fraction = 0.92
max_batch_size = 12
tp_size = 4


def get_cmd(MODEL_DIR):
    return (
    f'python -m tensorrt_llm.commands.serve serve {MODEL_DIR} '
    f'    --tp_size {tp_size} '
    f'    --kv_cache_free_gpu_memory_fraction {kv_cache_free_gpu_memory_fraction} '
    f'    --max_batch_size {max_batch_size} '
    f'    --host {host} '
    f'    --port {port} '
)
    
def wait_for_server(url=f"http://{host}:{port}", timeout=300, interval=1):
    start_time = time.time()
    while True:
        try:
            response = requests.put(url)
            if response.status_code != 403:  # Check if server responds
                return True
        except requests.RequestException:
            if time.time() - start_time > timeout:
                raise TimeoutError("Server did not respond within timeout period")
            time.sleep(interval)


tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR_FP8)


cmd = get_cmd(MODEL_DIR_FP8)
print(cmd)
log_path = Path(f"trtllm_server_FP8.log").resolve()
log_file = open(log_path, "w", buffering=1)
proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
wait_for_server()
print(f"Server is ready!")


# TODO : Add number of tokens, code rounds and time for each question, similar to https://www.kaggle.com/code/darraghdog/nemo-skills-qwen2-5-14b-mix48-trtllm-redrafter?scriptVersionId=228756964
# TODO : make a non code version of the notebook, a lot of people would want to use without code. 
# TODO : Use a subset of questions from the attached /kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv

def consume_stream(stream: Union[Stream, Generator], thread_id=None):
    """Process a single stream and return concatenated text with timing."""
    start_time = time.time()
    result = ""
    try:
        for chunk in stream:
            if chunk['generation'] is not None:
                result += chunk['generation']
    except Exception as e:
        pass    
        
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        'result': result,
        'duration': duration,
        'thread_id': thread_id
    }

def stream_generate(
    code_exec_model: CodeExecutionWrapper,
    prompts: list[str | dict],
    code_begin: str | list[str],
    code_end: str | list[str],
    code_output_begin: str | list[str],
    code_output_end: str | list[str],
    code_output_format: str | list[str],
    tokens_to_generate: int | list[int] = 512,
    temperature: float | list[float] = 0.0,
    top_p: float | list[float] = 0.95,
    top_k: int | list[int] = 0,
    min_p: float | list[float] = 0.0,
    repetition_penalty: float | list[float] = 1.0,
    random_seed: int | list[int] = 0,
    stop_phrases: list[str] | list[list[str]] | None = None,
    remove_stop_phrases: bool = True,
    timeout: int | list[int] | None = None,
    max_code_executions: int | list[int] | None = None,
    stop_after_n_completed=None, 
    stop_after_n_seconds=None,
    stop_after_n_same_answer=None,
    ) -> list[dict]:
    """Process multiple streams concurrently and return results with durations."""

    streams = code_exec_model.generate(
        prompts=prompts,
        code_begin=code_begin,
        code_end=code_end,
        code_output_begin=code_output_begin,
        code_output_end=code_output_end,
        code_output_format=code_output_format,
        tokens_to_generate=tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        random_seed=random_seed,
        stop_phrases=stop_phrases,
        remove_stop_phrases=remove_stop_phrases,
        timeout=timeout,
        max_code_executions=max_code_executions,
        stream=True,
        )

    with ThreadPoolExecutor() as executor:
        # Submit all streams to thread pool with thread IDs
        futures = [(i, executor.submit(consume_stream, stream, i)) for i, stream in enumerate(streams)]
        
        if stop_after_n_completed is not None:
            stop_after_n_completed = min(stop_after_n_completed, len(streams))
        
        start_time = time.time()
        current_answers = set() # set of answers that have been completed
        completed_futures = []  # list of tuples (thread_id, result_data)
        
        print('Finish times:', end = " ")
        while futures:
            
            if stop_after_n_completed is not None and len(completed_futures) >= stop_after_n_completed:
                print(f"\nStopping after {stop_after_n_completed} completed...")
                # This will asynchronously cancel all generations
                # We don't break here because we want to collect results up to now
                code_exec_model.model.cancel_all_generations()
                print("Cancel threads @ seconds... ", end = " ")
                
            elif stop_after_n_seconds is not None and time.time() - start_time >= stop_after_n_seconds:
                print(f"\nStopping after {stop_after_n_seconds} seconds...")
                code_exec_model.model.cancel_all_generations()
                print("Cancel threads @ seconds... ", end = " ")
            
            elif stop_after_n_same_answer is not None:
                # Check whether at least n elements in current_answers are the same
                if len(completed_futures) - len(current_answers) >= stop_after_n_same_answer-1:
                    print(f"\nStopping after {stop_after_n_same_answer} same answers...")
                    code_exec_model.model.cancel_all_generations()
                    print("Cancel threads @ seconds... ", end = " ")
            
            # TODO we can add other stopping conditions here 
            
            time.sleep(0.1)
            
            completed_in_this_iteration = []
            for idx, future in futures:
                if future.done():
                    result_data = future.result()
                    duration = result_data['duration']
                    print(f"{duration:.1f}", end = " ", flush = True)
                    completed_futures.append((idx, result_data))
                    completed_in_this_iteration.append((idx, future))

            
            for item in completed_in_this_iteration:
                futures.remove(item)
    
    # Sort by original index and return results with durations
    completed_futures.sort(key=lambda x: x[0])
    return [result_data for _, result_data in completed_futures]


sandbox = get_sandbox()  # localhost by default
llm = get_code_execution_model(server_type="trtllm-serve", sandbox=sandbox)


# Initialize the prompt template
prompt_template = get_prompt('generic/math', 'qwen-instruct')

# Set the code tags directly on the config's code_tags attribute
prompt_template.config.code_tags = CodeTags(
    code_begin="<tool_call>\n",
    code_end="</tool_call>\n",
)


# TODO - At the end make tokens_to_generate and time much longer, it will be needed for the harder questions. 
sampling_params = {
    "tokens_to_generate": 8000,
    "temperature": 0.,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
    # "stream": True,
}

request = copy.deepcopy(sampling_params)
list_of_texts = [
    prompt_template.fill({'problem': 'This is a very simple question, no tricks, just testing how quickly you answer. What is 1+1?'}),
    prompt_template.fill({'problem': 'What number comes after 1? Answer without thinking, in one word.'}),
    prompt_template.fill({'problem': 'What is the sum of all prime numbers less than 10 million?'}),
    ]
request["prompts"] = list_of_texts


res = stream_generate(
    llm,
    **request,
    **prompt_template.get_code_execution_args(),
    stop_after_n_seconds=60,
    stop_after_n_completed=2,
    stop_after_n_same_answer=None
    )


# res


os.killpg(proc.pid, signal.SIGTERM)    # or SIGKILL
proc.wait()
log_file.close() 
time.sleep(10)
print(f'Server closed... see output logs in {log_path}')


cmd = get_cmd(MODEL_DIR_BF16)
print(cmd)
log_path = Path(f"trtllm_server_BF16.log").resolve()
log_file = open(log_path, "w", buffering=1)
proc = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
wait_for_server()
print(f"Server is ready!")


res = stream_generate(
    llm,
    **request,
    **prompt_template.get_code_execution_args(),
    stop_after_n_seconds=60,
    stop_after_n_completed=2,
    stop_after_n_same_answer=None
    )


os.killpg(proc.pid, signal.SIGTERM)    # or SIGKILL
proc.wait()
log_file.close() 
time.sleep(10)
print(f'Server closed... see output logs in {log_path}')


# Look at one output
res[0]

