# !pip install vllm==0.10.0
# !pip install logits-processor-zoo==0.1.10
# !pip install triton==3.2.0
# !pip install clean-text
# !pip install -U --no-deps bitsandbytes peft accelerate datasets


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

MODEL_NAME = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
LORA_PATH = "/kaggle/input/qwen2-5-32b-awq"
if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"

    llm = vllm.LLM(
        MODEL_NAME,
        # quantization='awq',
        quantization='awq',
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
You are an expert subreddit moderation bot assistant. You will be given: the subreddit name, the exact rule text, any labeled examples (violations and non-violations), and one comment to evaluate. Use only the provided rule, examples, and comment; do not assume, invent, or add facts. Decide whether the comment violates the given rule. Your final output must be exactly one token: Yes or No. Write "Yes" only if the comment clearly and unambiguously meets the rule's violation criteria. Write "No" if it does not, or if the information is ambiguous or insufficient to determine a clear violation. Produce no explanations, no punctuation, no extra characters or whitespace, and no chain-of-thought. Be deterministic and concise.
"""
    
    prompts = []
    for i, row in df.iterrows():
        text = f"""
        Subreddit: r/{row.subreddit}
        Rule: {row.rule}
        
        Examples of rule violations and non-violations:
        
        Violation Example 1:
        - Comment: {row.positive_example_1}
        - Violation: Yes
        
        Non-Violation Example 1:
        - Comment: {row.negative_example_1}
        - Violation: No
        
        Violation Example 2:
        - Comment: {row.positive_example_2}
        - Violation: Yes
        
        Non-Violation Example 2:
        - Comment: {row.negative_example_2}
        - Violation: No
        
        ---
        
        **Analyze the following comment:**
        
        Comment: {row.body}
        Violation: 
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
    df[['row_id', 'rule_violation']].to_csv("/kaggle/working/submission_qwen.csv",index=False)
    pd.read_csv('/kaggle/working/submission_qwen.csv')


%cd /tmp
!python src/infer_qwen.py


%%writefile constants.py
import os
import pandas as pd
EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
MODEL_OUTPUT_PATH = '/kaggle/input/qwen3-0-6b-embedding-finetune-epoch2'
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = """
Instruction: Convert the following user search into a single-line, concise retrieval phrase optimized for semantic embedding. Preserve important named entities, numbers, dates, and locations; remove filler words and punctuation; normalize to lowercase. Output exactly one line with 3–12 meaningful words that capture the user’s intent. Do not include explanations, tokens, or metadata.
Query:
"""


CLEAN_TEXT = True
TOP_K = 2000
BATCH_SIZE = 128


%%writefile utils.py
import pandas as pd
import torch.distributed as dist

from datasets import Dataset
from cleantext import clean
from tqdm.auto import tqdm

from constants import CLEAN_TEXT


def build_prompt(row):
    return f"""r/{row["subreddit"]}\nComment: {row["body"]}"""


def cleaner(text):
    return clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        lang="en",
    )



def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []
    flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
            sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0)    
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    return dataframe


def prepare_dataframe(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    
    if CLEAN_TEXT:
        tqdm.pandas(desc="cleaner")
        dataframe["prompt"] = dataframe["prompt"].progress_apply(cleaner)

    if "rule_violation" in dataframe.columns:
        dataframe["rule_violation"] = dataframe["rule_violation"].map(
            {
                1: 1,
                0: -1,
            }
        )

    return dataframe


%%writefile semantic.py
import pandas as pd
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from tqdm.auto import tqdm
from peft import PeftModel, PeftConfig


from utils import get_dataframe_to_train, prepare_dataframe
from constants import DATA_PATH, EMBDEDDING_MODEL_PATH, EMBEDDING_MODEL_QUERY, TOP_K, BATCH_SIZE, MODEL_OUTPUT_PATH



def get_scores(test_dataframe):
    corpus_dataframe = get_dataframe_to_train(DATA_PATH)
    corpus_dataframe = prepare_dataframe(corpus_dataframe)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(EMBDEDDING_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(EMBDEDDING_MODEL_PATH)
    
    # Load adapter configuration and model
    adapter_config = PeftConfig.from_pretrained(MODEL_OUTPUT_PATH)
    lora_model = PeftModel.from_pretrained(model, MODEL_OUTPUT_PATH, config=adapter_config)
    merged_model = lora_model.merge_and_unload()
    tokenizer.save_pretrained("Qwen3Emb_Finetuned")
    merged_model.save_pretrained("Qwen3Emb_Finetuned")

    # 4. Tạo lại SentenceTransformer từ encoder đã merge
    embedding_model = SentenceTransformer(model_name_or_path="Qwen3Emb_Finetuned", device="cuda")

    print('Done loading model!')

    result = []
    for rule in tqdm(test_dataframe["rule"].unique(), desc=f"Generate scores for each rule"):
        test_dataframe_part = test_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe_part.reset_index(names="row_id")
        
        query_embeddings = embedding_model.encode(
            sentences=test_dataframe_part["prompt"].tolist(),
            prompt=EMBEDDING_MODEL_QUERY,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        document_embeddings = embedding_model.encode(
            sentences=corpus_dataframe_part["prompt"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        test_dataframe_part["semantic"] = semantic_search(
            query_embeddings,
            document_embeddings,
            top_k=TOP_K,
            score_function=dot_score,
        )
        def get_score(semantic):
            semantic = pd.DataFrame(semantic)
            semantic = semantic.merge(
                corpus_dataframe_part[["row_id", "rule_violation"]],
                how="left",
                left_on="corpus_id",
                right_on="row_id",
            )
            semantic["score"] = semantic["score"]*semantic["rule_violation"]
            return semantic["score"].sum()
            
        tqdm.pandas(desc=f"Add label for {rule=}")
        test_dataframe_part["rule_violation"] = test_dataframe_part["semantic"].progress_apply(get_score)
        result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())
        
    submission = pd.concat(result, axis=0)
    return submission


def generate_submission():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataframe = prepare_dataframe(test_dataframe)
    
    submission = get_scores(test_dataframe)
    submission = test_dataframe[["row_id"]].merge(submission, on="row_id", how="left")
    submission.to_csv("/kaggle/working/submission_qwen3.csv", index=False)


if __name__ == "__main__":
    generate_submission()


from semantic import generate_submission
generate_submission()


import os, math, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


import pandas as pd
import numpy as np

test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv', index_col='row_id')


import vllm

llm = vllm.LLM(
    "/kaggle/input/jigsaw-llama3-1-8b-instruct-training-one-epoch/llama-8b-instruct-jigsaw",
    tensor_parallel_size=2, 
    gpu_memory_utilization=0.95, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=2048,    
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


SYS_PROMPT = """
    You are an expert moderator bot for Reddit. Your task is to accurately determine whether a given Reddit comment violates a specific subreddit rule. You will be provided with the subreddit name, the rule, and several examples of both rule violations and non-violations. Analyze the comment carefully in the context of the provided rule and examples. Your final response MUST be either 'Yes' or 'No'. A 'Yes' indicates a clear violation, and a 'No' indicates no violation.    
    """


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
        "content": SYS_PROMPT
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
sub.to_csv('/kaggle/working/submission_llama3-1.csv')


import pandas as pd
import numpy as np

q = pd.read_csv('/kaggle/working/submission_qwen.csv')
l = pd.read_csv('/kaggle/working/submission_qwen3.csv')
ll = pd.read_csv('/kaggle/working/submission_llama3-1.csv')

rq = q['rule_violation'].rank(method='average') / (len(q)+1)
rl = l['rule_violation'].rank(method='average') / (len(l)+1)
rll = ll['rule_violation'].rank(method='average') / (len(ll)+1)

blend = 0.5*rq + 0.4*rl + 0.1*rll   # or tune the rank-weights with a tiny grid using OOF
q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)

