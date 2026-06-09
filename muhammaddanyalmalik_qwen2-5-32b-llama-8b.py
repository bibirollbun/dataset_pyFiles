# !pip install vllm
# !pip install logits-processor-zoo==0.1.10
# !pip install triton==3.2.0


! mkdir -p /tmp/src



%%writefile /tmp/src/infer_qwen.py

import os
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

MODEL_NAME = "/kaggle/input/qwen2-5-32b-instruct-gptq-int4"
LORA_PATH = "/kaggle/input/qwen2-5-32b-gptq-int4-batch4-full"
if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"

    llm = vllm.LLM(
        MODEL_NAME,
        # quantization='awq',
        quantization='gptq',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
    )
    tokenizer = llm.get_tokenizer()
    SYS_PROMPT = """
    You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
    """
    
    prompts = []
    for i, row in df.iterrows():
        text = f"""
    r/{row.subreddit}
    Rule: {row.rule}
    
    1) {row.positive_example_1}
    Violation: Yes
    
    2) {row.positive_example_2}
    Violation: Yes
    
    3) {row.negative_example_1}
    Violation: No
    
    4) {row.negative_example_2}
    Violation: No
    
    5) {row.body}
    """
        
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": text}
        ]
    
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        ) + "Answer:"
        prompts.append(prompt)
    
    df["prompt"] = prompts
    
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=['Yes','No'])
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, LORA_PATH)
    )
    logprobs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    logit_matrix = pd.DataFrame(logprobs)[['Yes','No']]
    df = pd.concat([df, logit_matrix], axis=1)
    
    df[['Yes',"No"]] = df[['Yes',"No"]].apply(lambda x: softmax(x.values), axis=1, result_type="expand")
    df["pred"] = df["Yes"]
    df['rule_violation'] = df["pred"]
    df[['row_id', 'rule_violation']].to_csv("submission_qwen.csv",index=False)
    pd.read_csv('submission_qwen.csv')


%cd /tmp
!python src/infer_qwen.py


import os, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


import pandas as pd
import numpy as np

test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv', index_col='row_id')
sub

import vllm

llm = vllm.LLM(
    "/kaggle/input/jigsaw-llama3-1-8b-instruct-training-one-epoch/llama-8b-instruct-jigsaw",
    tensor_parallel_size=2, 
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=2048,
    # disable_log_stats=True,
    # enable_prefix_caching=True,
    
)
tokenizer = llm.get_tokenizer()


from typing import Any, Dict, List
from transformers import LogitsProcessor
import torch

choices = ["No", "Yes"]

KEEP = []
for x in choices:
    c = tokenizer.encode(x,add_special_tokens=False)[0]
    KEEP.append(c)
print(f"Force predictions to be tokens {KEEP} which are {choices}.")

class DigitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.allowed_ids = KEEP
        
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        scores[self.allowed_ids] += 100
        return scores


sys_prompt = '''You are given a comment on reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No.'''


def formatting(dataset):
    texts = []
    for i in range(len(dataset)):
        texts.append(tokenizer.apply_chat_template(dataset[i], tokenize=False, add_generation_prompt=False))
    return texts


template = """
Subreddit: r/{subreddit}
Rule: {rule}
Examples:
1) {positive_example_1}
Violation: Yes

2) {negative_example_1}
Violation: No

3) {negative_example_2}
Violation: No

4) {positive_example_2}
Violation: Yes
Comment:
{body}
Violation: """


dataset = []
for index,row in test.iterrows():
    
    formatted_sample = [
        {
        "role": "system",
        "content": sys_prompt
    },
       {
           "role": "user",
           "content": template.format(
               rule = row.rule,
               subreddit = row.subreddit,
               body = row.body,
               positive_example_1 = row.positive_example_1,
               negative_example_1 = row.negative_example_1,
               positive_example_2 = row.positive_example_2,
               negative_example_2 = row.negative_example_2
           )
       }]
    
    dataset.append( formatted_sample )


all_prompts = formatting(dataset)


logits_processors = [DigitLogitsProcessor(tokenizer)]
responses = llm.generate(
    all_prompts,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=True,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=logits_processors,
        logprobs = 2
    ),
    use_tqdm = True
)


results = []
errors = 0

for i,response in enumerate(responses):
    try:
        x = response.outputs[0].logprobs[0]
        logprobs = []
        for k in KEEP:
            if k in x:
                logprobs.append( math.exp(x[k].logprob) )
            else:
                logprobs.append( 0 )
                print(f"bad logits {i}")
        logprobs = np.array( logprobs )
        logprobs /= logprobs.sum()
        results.append( logprobs )
    except:
        #print(f"error {i}")
        results.append( np.array([1/2., 1/2.]) )
        errors += 1
        
print(f"There were {errors} inference errors out of {i+1} inferences")
results = np.vstack(results)


probs = [x[1] for x in results]
sub['rule_violation'] = probs
sub.to_csv('submission_llama.csv')


q = pd.read_csv('submission_qwen.csv')
l = pd.read_csv('submission_llama.csv')

rq = q['rule_violation'].rank(method='average') / (len(q)+1)
rl = l['rule_violation'].rank(method='average') / (len(l)+1)

blend = 0.7*rq + 0.3*rl   # or tune the rank-weights with a tiny grid using OOF
q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)




