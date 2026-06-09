MODEL_NAME = "/kaggle/input/qwen2.5/transformers/7b-instruct-gptq-int4/1"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" works with internet
LORA_PATH = "/kaggle/input/qwen-full-final/trained_model"


import os
os.environ["VLLM_USE_V1"] = "0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import torch
import vllm
import numpy as np
from vllm.lora.request import LoRARequest
import argparse
from scipy.special import softmax
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"CUDA Compute Capability: {torch.cuda.get_device_capability()}")


llm = vllm.LLM(
    MODEL_NAME,
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
Given a reddit comment and a rule, your task is to classify whether the comment violates the given rule. Only respond with Yes/No.
"""

prompts = []
for i, row in df.iterrows():
    text = f"""
r/{row.subreddit}
Rule: {row.rule}

1) {row.positive_example_1}
Violation: Yes

2) {row.negative_example_1}
Violation: No

3) {row.negative_example_2}
Violation: No

4) {row.positive_example_2}
Violation: Yes

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
df[['row_id', 'rule_violation']].to_csv("submission.csv",index=False)
df[['row_id', 'rule_violation']].head()

