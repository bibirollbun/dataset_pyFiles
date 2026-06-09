!bash /kaggle/input/dependence-script/install_requirements.sh


%%writefile constants.py
BASE_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = '''You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No.'''


%%writefile utils.py
import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT
import random, numpy as np
random.seed(42)
np.random.seed(42)


def build_prompt(row):
    return f"""
{BASE_PROMPT}

Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{COMPLETE_PHRASE} Yes

2) {row["negative_example"]}
{COMPLETE_PHRASE} No

---
Comment: {row["body"]}
{COMPLETE_PHRASE}"""


def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv").sample(frac=0.5, random_state=42).reset_index(drop=True)

    flatten = []

    # ---------- 处理训练集 ----------
    train_df = train_dataset[["body", "rule", "subreddit", "rule_violation",
                              "positive_example_1","positive_example_2",
                              "negative_example_1","negative_example_2"]].copy()

    # 随机选 positive_example 和 negative_example
    train_df["positive_example"] = np.where(
        np.random.rand(len(train_df)) < 0.5,
        train_df["positive_example_1"],
        train_df["positive_example_2"]
    )
    train_df["negative_example"] = np.where(
        np.random.rand(len(train_df)) < 0.5,
        train_df["negative_example_1"],
        train_df["negative_example_2"]
    )

    # 删除原来的候选列
    train_df.drop(columns=["positive_example_1","positive_example_2",
                           "negative_example_1","negative_example_2"], inplace=True)

    flatten.append(train_df)

    # ---------- 处理测试集 ----------
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[["rule","subreddit",
                                        "positive_example_1","positive_example_2",
                                        "negative_example_1","negative_example_2"]].copy()

            if violation_type == "positive":
                # body 用当前 positive_example
                body_col = f"positive_example_{i}"
                other_positive_col = f"positive_example_{3-i}"  # 另一个 positive
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["positive_example"] = sub_dataset[other_positive_col]
                # negative_example 随机选
                sub_dataset["negative_example"] = np.where(
                    np.random.rand(len(sub_dataset)) < 0.5,
                    sub_dataset["negative_example_1"],
                    sub_dataset["negative_example_2"]
                )
                sub_dataset["rule_violation"] = 1

            else:  # violation_type == "negative"
                body_col = f"negative_example_{i}"
                other_negative_col = f"negative_example_{3-i}"
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["negative_example"] = sub_dataset[other_negative_col]
                sub_dataset["positive_example"] = np.where(
                    np.random.rand(len(sub_dataset)) < 0.5,
                    sub_dataset["positive_example_1"],
                    sub_dataset["positive_example_2"]
                )
                sub_dataset["rule_violation"] = 0

            # 删除原来的候选列
            sub_dataset.drop(columns=["positive_example_1","positive_example_2",
                                      "negative_example_1","negative_example_2"], inplace=True)

            flatten.append(sub_dataset)

    # 合并所有 DataFrame
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(ignore_index=True)

    return dataframe



def build_dataset(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    columns = ["prompt"]
    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: POSITIVE_ANSWER,
                0: NEGATIVE_ANSWER,
            }
        )
        columns.append("completion")

    dataframe = dataframe[columns]
    dataset = Dataset.from_pandas(dataframe)
    dataset.to_pandas().to_csv("/kaggle/working/dataset.csv", index=False)
    return dataset


%%writefile train.py
import pandas as pd

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers.utils import is_torch_bf16_gpu_available
from utils import build_dataset, get_dataframe_to_train
from constants import DATA_PATH, BASE_MODEL_PATH, LORA_PATH


def main():
    dataframe = get_dataframe_to_train(DATA_PATH)
    train_dataset = build_dataset(dataframe)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    training_args = SFTConfig(
        num_train_epochs=1,
        
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        
        optim="paged_adamw_8bit",
        learning_rate=1e-4, #keep high, lora usually likes high. 
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    
        save_strategy="no",
        report_to="none",
    
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
    )
    
    trainer = SFTTrainer(
        BASE_MODEL_PATH,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    
    trainer.train()
    trainer.save_model(LORA_PATH)


if __name__ == "__main__":
    main()


%%writefile inference.py
import os
os.environ["VLLM_USE_V1"] = "0"

import vllm
import torch
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from vllm.lora.request import LoRARequest
from utils import build_dataset
from constants import BASE_MODEL_PATH, LORA_PATH, DATA_PATH, POSITIVE_ANSWER, NEGATIVE_ANSWER
import random
import multiprocessing as mp


def run_inference_on_device(df_slice):
    """在当前进程可见的 GPU 上跑 vLLM 推理"""
    llm = vllm.LLM(
        BASE_MODEL_PATH,
        quantization="gptq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2836,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )

    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE_ANSWER, NEGATIVE_ANSWER])

    test_dataset = build_dataset(df_slice)
    texts = test_dataset["prompt"]

    outputs = llm.generate(
        texts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
        lora_request=LoRARequest("default", 1, LORA_PATH)
    )

    log_probs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    predictions = pd.DataFrame(log_probs)[[POSITIVE_ANSWER, NEGATIVE_ANSWER]]
    predictions["row_id"] = df_slice["row_id"].values
    return predictions


def worker(device_id, df_slice, return_dict):
    # 限制该进程只看到一张 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[Worker {device_id}] Running on GPU {device_id}, data size={len(df_slice)}")

    preds = run_inference_on_device(df_slice)
    return_dict[device_id] = preds


def main():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")

    # 随机选择例子
    test_dataframe["positive_example"] = test_dataframe.apply(
        lambda row: random.choice([row["positive_example_1"], row["positive_example_2"]]),
        axis=1
    )
    test_dataframe["negative_example"] = test_dataframe.apply(
        lambda row: random.choice([row["negative_example_1"], row["negative_example_2"]]),
        axis=1
    )
    test_dataframe = test_dataframe.drop(
        columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"],
        errors="ignore"
    )

    # 切分数据
    mid = len(test_dataframe) // 2
    df0 = test_dataframe.iloc[:mid].reset_index(drop=True)
    df1 = test_dataframe.iloc[mid:].reset_index(drop=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    # 两个进程并行
    p0 = mp.Process(target=worker, args=(0, df0, return_dict))
    p1 = mp.Process(target=worker, args=(1, df1, return_dict))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    # 合并结果
    predictions = pd.concat([return_dict[0], return_dict[1]], ignore_index=True)

    # 构建 submission
    submission = predictions[["row_id", POSITIVE_ANSWER]].rename(columns={POSITIVE_ANSWER: "rule_violation"})
    rq = submission['rule_violation'].rank(method='average') / (len(submission) + 1)
    submission['rule_violation'] = rq

    submission.to_csv("submission_qwen.csv", index=False)
    print("✅ Saved submission_qwen.csv")


if __name__ == "__main__":
    main()


%%writefile accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  train_batch_size: 64
  train_micro_batch_size_per_gpu: 4
  
  zero_stage: 2
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  
  stage3_gather_16bit_weights_on_model_save: false
  stage3_max_live_parameters: 1e8
  stage3_max_reuse_distance: 1e8
  stage3_prefetch_bucket_size: 5e7
  stage3_param_persistence_threshold: 1e5
  
  zero_allow_untested_optimizer: true
  zero_force_ds_cpu_optimizer: false
  
  fp16:
    enabled: true
    loss_scale: 0
    initial_scale_power: 16
    loss_scale_window: 1000
    hysteresis: 2
    min_loss_scale: 1
  
distributed_type: DEEPSPEED
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: INDUCTOR
  dynamo_use_fullgraph: false
  dynamo_use_dynamic: false
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false


!accelerate launch --config_file accelerate_config.yaml train.py


!python inference.py


!head submission_qwen.csv


%%writefile infer_qwen.py

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

MODEL_NAME = "/kaggle/input/qwen2.5/transformers/14b-instruct-gptq-int4/1"
LORA_PATH = "/kaggle/input/lora_14b_gptq_1epoch_r32/keras/default/1"
if __name__=='__main__':
    os.environ["VLLM_USE_V1"] = "0"

    llm = vllm.LLM(
        MODEL_NAME,
        # quantization='awq',
        quantization='gptq',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.98,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=2836,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=32
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
    df[['row_id', 'rule_violation']].to_csv("submission_qwen14b.csv",index=False)
    pd.read_csv('submission_qwen14b.csv')


!python infer_qwen.py


!head submission_qwen14b.csv


import os
import pandas as pd



%%writefile constants.py
EMBDEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
# MODEL_OUTPUT_PATH = '/kaggle/input/qwen3-8b-embedding'
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules"

# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json
EMBEDDING_MODEL_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

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
    test_dataset = pd.read_csv(f"{data_path}/test.csv").sample(frac=0.6, random_state=42).reset_index(drop=True)

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
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, dot_score
from tqdm.auto import tqdm

from utils import get_dataframe_to_train, prepare_dataframe
from constants import DATA_PATH, EMBDEDDING_MODEL_PATH, EMBEDDING_MODEL_QUERY, TOP_K, BATCH_SIZE


def get_scores(test_dataframe):
    corpus_dataframe = get_dataframe_to_train(DATA_PATH)
    corpus_dataframe = prepare_dataframe(corpus_dataframe)
    
    # ✅ 直接加载 Qwen3-0.6B Embedding 模型，不加载 LoRA
    embedding_model = SentenceTransformer(model_name_or_path=EMBDEDDING_MODEL_PATH, device="cuda")
    print('✅ Done loading Qwen3-0.6B Embedding model!')

    result = []
    for rule in tqdm(test_dataframe["rule"].unique(), desc=f"Generate scores for each rule"):
        test_dataframe_part = test_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe.query("rule == @rule").reset_index(drop=True)
        corpus_dataframe_part = corpus_dataframe_part.reset_index(names="row_id")
        
        # Encode queries (test set)
        query_embeddings = embedding_model.encode(
            sentences=test_dataframe_part["prompt"].tolist(),
            prompt=EMBEDDING_MODEL_QUERY,  # 使用 HF 配置中指定的 query prompt
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        
        # Encode documents (train set)
        document_embeddings = embedding_model.encode(
            sentences=corpus_dataframe_part["prompt"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_tensor=True,
            device="cuda",
            normalize_embeddings=True,
        )
        
        # Semantic search (cosine/dot similarity)
        search_results = semantic_search(
            query_embeddings,
            document_embeddings,
            top_k=TOP_K,
            score_function=dot_score,  # cosine 相似度等价于 normalize 后的 dot_score
        )
        
        # Attach search results
        test_dataframe_part["semantic"] = search_results

        # Scoring function: weighted sum by similarity * label
        def get_score(semantic_list):
            if not semantic_list:
                return 0.0
            semantic_df = pd.DataFrame(semantic_list)
            semantic_df = semantic_df.merge(
                corpus_dataframe_part[["row_id", "rule_violation"]],
                how="left",
                left_on="corpus_id",
                right_on="row_id",
            )
            semantic_df["score"] = semantic_df["score"] * semantic_df["rule_violation"]
            return semantic_df["score"].sum()
            
        tqdm.pandas(desc=f"Scoring for rule: {rule}")
        test_dataframe_part["rule_violation"] = test_dataframe_part["semantic"].progress_apply(get_score)
        result.append(test_dataframe_part[["row_id", "rule_violation"]].copy())
        
    submission = pd.concat(result, axis=0)
    return submission


def generate_submission():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataframe = prepare_dataframe(test_dataframe)
    
    submission = get_scores(test_dataframe)
    # Merge to ensure all row_id are present
    submission = test_dataframe[["row_id"]].merge(submission, on="row_id", how="left")
    # Fill NaN with 0 or median if needed
    submission["rule_violation"] = submission["rule_violation"].fillna(0)
    submission.to_csv("submission_qwen3.csv", index=False)
    print("✅ Submission saved to submission_qwen3.csv")


if __name__ == "__main__":
    generate_submission()


!python semantic.py


!head submission_qwen3.csv


import pandas as pd
import numpy as np

q = pd.read_csv('submission_qwen.csv')
l = pd.read_csv('submission_qwen3.csv')
m = pd.read_csv('submission_qwen14b.csv')


rq = q['rule_violation'].rank(method='average') / (len(q)+1)
rl = l['rule_violation'].rank(method='average') / (len(l)+1)
rm = m['rule_violation'].rank(method='average') / (len(m)+1)


blend = 0.5*rq + 0.3*rl + 0.2*rm   # or tune the rank-weights with a tiny grid using OOF
q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)


import pandas as pd
pd.read_csv('/kaggle/working/submission.csv')




