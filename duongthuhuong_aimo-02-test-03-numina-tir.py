import os
import gc
import time
import warnings
#Data
import pandas as pd
import polars as pl
from datasets import load_dataset, Dataset, concatenate_datasets
#Processing
import re
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from collections import Counter
#Model
import torch
from transformers import set_seed
from tqdm import tqdm
from vllm import LLM, SamplingParams
#Evaluation
import kaggle_evaluation.aimo_2_inference_server


warnings.simplefilter('ignore')
pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 45) * 60

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
set_seed(42)

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


@dataclass 
class Config:
    model_id: str
    #LLM Parameters
    max_model_len: int
    num_gpu: int
    #Decoding Parameters
    num_samples: int         # Number of candidates to generate (width) 
    num_generations: int     # Number of steps to generate per candidate (depth) 
    restart_on_fail: bool    # Regenerate a step if it fails to generate Python codeblocks (True/False)
    #Sampling Parameters 
    temperature: float       # Creativity score
    min_p: float


config = Config(
    model_id        = '/kaggle/input/numinamath-7b-cot/transformers/default/1',
    max_model_len   = 2048,
    num_gpu         = 4,
    
    num_samples     = 5,
    num_generations = 3,
    restart_on_fail = True,
    
    temperature     = 0.8,
    min_p           = 0.01
)


llm = LLM(
    model                  = config.model_id,
    max_model_len          = config.max_model_len,
    trust_remote_code      = True,
    tensor_parallel_size   = config.num_gpu,
    gpu_memory_utilization = 0.96,
)
tokenizer = llm.get_tokenizer()


def extract_boxed_answer (text):
    def last_boxed_only_string(text):
        idx = text.rfind("\\boxed")
        if idx < 0:
            idx = text.rfind("\\fbox")
            if idx < 0:
                return  None
        i = idx
        
        right_brace_idx = None 
        num_left_braces_open = 0 
        while i < len(text):
            if text[i] == "{" :
                num_left_braces_open += 1 
            if text[i] == "}" :
                num_left_braces_open -= 1 
                if num_left_braces_open == 0 :
                    right_brace_idx = i
                    break 
            i += 1
        
        if right_brace_idx is None :
             return None 
        return text[idx:right_brace_idx + 1]

    def remove_boxed(boxed):
        left = "\\boxed{" 
        try:
            assert boxed[:len(left)] == left
            assert boxed[-1] == "}" 
            length = len (left)
            return boxed[length:-1]
        except Exception:
            return None

    boxed = last_boxed_only_string(text)
    if boxed is None:
         return None
    answer = remove_boxed(boxed)
    return answer


def normalize_answer(answer):
    #Exchange
    subs = [("an ",""), ("a ",""), (".$","$"), ("\\$",""), (r"\ "," "), (" ",""), ("mbox","text"), (",\\text{and}",","), ("\\text{and}",","), ("\\text{m}","\\text{}"), ("\\le","<")]
    for before, after in subs:
        answer = answer.replace(before, after)
    #Remove
    remove = ["square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units", "\\ldots", " sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots", "\n", "\r", "\f", "\%"]
    for expr in remove:
        answer = answer.replace(expr, "")
    #Keep inner content
    sub_patterns = [ r"(\\text\{)(.*?)(\})" , r"(\\textbf\{)(.*?)(\})" , r"(\\overline\ {)(.*?)(\})" , r"(\\boxed\{)(.*)(\})" ]
    for pattern in sub_patterns:
        answer = re.sub(pattern, "\\2" , answer)
    #Get answer
    split_patterns = [ r"finalansweris(.*)" , r"answer?is:?(.*)" , r"oxed\{(.*?)\}" , r"\$(.*?)\$ " ]
    for pattern in split_patterns:
        if len(re.findall(pattern, answer)) > 0 :
            answer = re.findall(pattern, answer)[- 1 ]
    answer = answer.strip()
    
    if "rac" in answer and "\\frac" not in answer:
        answer = answer.replace("rac", "\\frac")
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    
    if answer.replace(",", "").isdigit():
        answer = answer.replace(",", "")
    return answer


def filter_answers(answers):
    valid_answers = []
    for answer in answers:
        try:
            if int(answer) == float(answer):
                valid_answers.append(int(answer))
        except:
            pass
    if not valid_answers:
        return [210]

    return [answer%1000 for answer in valid_answers if answer > 0]


def answer_majority_vote(answers):
    if not len(answers):
        return 0
    c = Counter(answers)
    value, _ = c.most_common()[0]
    return value


sampling_params = SamplingParams(
    temperature = config.temperature,
    max_tokens  = config.max_model_len,
    min_p       = config.min_p,
    skip_special_tokens = True,
    include_stop_str_in_output = True,
    stop = ["```output\n"],
)


system = "You are a helpful and reflective maths assistant, please use chained deep reasoning to put the final integer answer in \\boxed{}."

def apply_template(problem, tokenizer, prompt):
    messages = [
        {"role": "system", "content": system},
        {"role": "user"  , "content": prompt.format(problem["prompt"], "{}")}
    ]
    text = tokenizer.apply_chat_template(
        conversation = messages,
        tokenize = False,
        add_generation_prompt = True
    )
    problem["text"] = text
    return problem


def generate_batched(samples, llm, sampling_params ):
    outputs = llm.generate(
        samples["gen_texts"],
        sampling_params = sampling_params,
    )
    samples["gen_texts"] = [o.prompt + o.outputs[0].text for o in outputs]
    
    return samples


class PythonREPL:
    def __init__ (self, timeout = 60):
        self.timeout = timeout

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(*_):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield 
        finally:
            signal.alarm(0)

    def __call__(self, query):
        query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
        query = query.strip().split("\n")
        if "print(" not in query[-1]:
            if "#" in query[-1]:
                query[-1] = query[-1].split("#")[0]
            query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open (temp_file_path, "w" , encoding = "utf-8") as f:
                f.write(query)
            with self.time_limit(self.timeout):
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output = True,
                    check   = False,
                    text    = True,
                    timeout = self.timeout,
                )
                if result.returncode == 0 :
                    output = result.stdout
                    return True, output.strip()
                error_msg = result.stderr.strip()
                msgs = error_msg.split( "\n" )
                new_msgs = []
                want_next = False 
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif temp_file_path in m:
                        st  = m.index('"/') + 1 if '"/' in m else 0 
                        ed  = m.index(temp_file_path) + 1 if temp_file_path in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m   = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True 
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False 
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()


def execute_completion(executor, completion, return_status, last_code_block):
    executions = re.findall(r"```python(.*?)```", completion, re.DOTALL)
    if len(executions) == 0:
        return completion, False if return_status else completion
    if last_code_block:
        executions = [executions[-1]]
    
    outputs   = []
    successes = []
    for code in executions:
        success = False 
        for lib in ("subprocess", "venv"):
            if lib in code:
                output = f"{lib} is not allowed"
                outputs.append(output)
                successes.append(success)
                continue
        try:
            success, output = executor(code)
        except TimeoutError as e:
            print("Code timed out")
            output = e
        if not success and not return_status:
            output = ""
        outputs.append(output)
        successes.append(success)
    output  = str(outputs[-1]).strip()
    success = successes[-1]
    if return_status:
        return output, success
    return output

def postprocess_completion(text, return_status, last_code_block):
    executor = PythonREPL()
    result = execute_completion(
        executor, text,
        return_status   = return_status,
        last_code_block = last_code_block
    )
    del executor
    return result


def process_LLM_output(sample, restart_on_fail, last_step, check_last_n_chars = 100):
    gen_text = sample["gen_texts"]
    num_python_blocks = len(re.findall(r"```python(.*?)```", gen_text, re.DOTALL))
    region_to_check   = gen_text[-check_last_n_chars:]
    
    # if num_python_blocks == 0:
    #     if restart_on_fail:
    #         print("no code has ever been generated, RESTARTING\n")
    #         sample["gen_texts"] = sample["text"]
    #     else:
    #         print ("no code has ever been generated, STOP")
    #         sample["should_prune"] = True
    #         sample["has_code"]     = False
    #     return sample
    
    if (not gen_text.endswith("```output\n")) and ("answer is" in region_to_check or "\\boxed" in region_to_check):
        # if "answer is" in region_to_check or "\\boxed" in region_to_check:
        #     num_output_blocks = len(re.findall(r"```output(.*?)```", gen_text, re.DOTALL))
        # if num_output_blocks == 0:
        #     print("HALLUCINATED")
        #     sample["should_prune"] = True 
        #     return sample
        if "boxed" in region_to_check:
            try:
                answer = normalize_answer(extract_boxed_answer(region_to_check))
                print("\nANSWER: ", answer)
            except Exception:
                answer = "-1" 
        else:
            answer = normalize_answer(region_to_check)
        
        sample["model_answers"] = answer
        return sample
     
    if last_step:
        return sample
    
    if not gen_text.endswith( "```output\n" ):
        print ("\nWarning: output block not found: " , gen_text[-40:])
        if restart_on_fail:
            sample["gen_texts"] = sample["text"]
        else:
            sample["should_prune"] = True 
        return sample
        
    code_result, _ = postprocess_completion(gen_text, return_status = True, last_code_block = True)
    truncation_limit = 200
    if len(code_result) > truncation_limit:
        code_result = code_result[:truncation_limit] + " ... (output truncated)" 
    sample["gen_texts"] = gen_text + "\\boxed{" + f"{code_result}" + "}"
    
    return sample


def answer_prediction(question: str) -> int:
    print(question)

    if time.time() > cutoff_time: 
        return 210
    
    problem = apply_template(
        problem   = {"prompt" : question},
        tokenizer = tokenizer,
        prompt    = "{}"
    )
    
    samples = Dataset.from_list([{
        "text"         : problem["text"],
        "gen_texts"    : problem["text"],
        "should_prune" : False,
        "model_answers": "-1",
        "has_code"     : True,
    } for _ in range(config.num_samples)
    ])
    
    completed = []
    for step in range(config.num_generations):
        ##LLM Generating
        samples = samples.map(
            generate_batched,
            batch_size = 128,
            batched    = True ,
            fn_kwargs  = {
                "llm": llm,
                "sampling_params": sampling_params
            },
            load_from_cache_file = False ,
        )
        ##Code Executation
        samples = samples.map(
            process_LLM_output,
            num_proc = 4,
            load_from_cache_file = False,
            fn_kwargs = {
                "restart_on_fail": config.restart_on_fail,
                "last_step"      : step == (config.num_generations - 1)
            },
        )
        
        done = samples.filter(
            lambda x: x["should_prune"] is True,
            load_from_cache_file = False
        )
        if len(done):
            completed.append(done)
        
        samples = samples.filter(
            lambda x: x["should_prune"] is False ,
            load_from_cache_file = False
        )
    
    completed.append(samples)
    samples = concatenate_datasets(completed)
    candidates = samples["model_answers"]
    #Logging
    print(f"=== CANDIDATE ANSWERS ({len(candidates)}) ===\n {candidates} \n")
    filtered = filter_answers(candidates)
    print(f"=== FILTERED ANSWERS ({len(filtered)})===\n {filtered} \n")
    majority = answer_majority_vote(filtered)
    print(f"=== MAJORITY ANSWER (mod 1000) ===\n {majority} \n")
    return majority


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    #Unpack values
    id_ = id_.item(0)
    question = question.item(0)
    #Make a prediction
    prediction = answer_prediction(question)
    print("---------------------------------------------\n\n\n")
    return pl.DataFrame({'id': id_, 'answer': prediction})


pd.read_csv(
    '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
).drop('answer', axis=1).to_csv('reference.csv', index = False)


inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

start_time = time.time()
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            'reference.csv',
        )
    )
end_time = time.time()
inference_time = end_time - start_time
print(f"\n\n Inference time: {inference_time:.4f} seconds")

