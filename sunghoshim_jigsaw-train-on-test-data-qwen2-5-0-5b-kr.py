!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'trl==0.21.0' 'optimum==1.27.0' 'auto-gptq==0.7.1' 'bitsandbytes==0.46.1' 'deepspeed==0.17.4' 'logits-processor-zoo==0.2.1' 'vllm==0.10.0'
!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'triton==3.2.0'
!uv pip install --system --no-index --find-links='/kaggle/input/jigsaw-packages2/whls/' 'clean-text'
!uv pip install --system --no-index -U --no-deps --find-links='/kaggle/input/jigsaw-packages2/whls/' 'peft' 'accelerate' 'datasets'


!pip list | grep -E "torch|transformers|datasets|peft|accelerate|unsloth|trl|optimum|gptq|bitsandbytes|deepspeed|logits|vllm|triton"


%%writefile constants.py
BASE_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
# BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."
BASE_PROMPT = '''You are given a comment from reddit and a rule. Your task is to classify whether the comment violates the rule. Only respond Yes/No.'''


%%writefile utils.py
import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT
import random, numpy as np
random.seed(42)
np.random.seed(42)


# def build_prompt(row):
#     return f"""
# {BASE_PROMPT}

# r/{row["subreddit"]} rule: {row["rule"]}

# Comment: {row["body"]}
# ---
# {COMPLETE_PHRASE}"""
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

    # =========== train.csv ë¶€ë¶„ ==============
    train_df = train_dataset[["body", "rule", "subreddit", "rule_violation",
                              "positive_example_1","positive_example_2",
                              "negative_example_1","negative_example_2"]].copy()

    # ìƒ˜í”Œ ë‘� ê°œì¤‘ í•˜ë‚˜ ê³ ë¦„
    # np.random.rand() : 0~1 ì‚¬ì�´ uniform distribution ìœ¼ë¡œ ì›�í•˜ëŠ” shape ë§Œë“¤ì–´ ì¤Œ
    # np.where() : ì¡°ê±´ì—� ë”°ë�¼ ì„ íƒ�
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

    train_df.drop(columns=["positive_example_1","positive_example_2",
                           "negative_example_1","negative_example_2"], inplace=True)

    flatten.append(train_df)

    # =========== test.csv ë¶€ë¶„ ==============
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[["rule","subreddit",
                                        "positive_example_1","positive_example_2",
                                        "negative_example_1","negative_example_2"]].copy()

            if violation_type == "positive":
                body_col = f"positive_example_{i}"  # ìƒ˜í”Œ í•˜ë‚˜ ì„ íƒ�í•´ì„œ body ë¡œ
                other_positive_col = f"positive_example_{3-i}"  # ë‹¤ë¥¸ ìƒ˜í”Œì�€ example ë¡œ
                sub_dataset["body"] = sub_dataset[body_col]
                sub_dataset["positive_example"] = sub_dataset[other_positive_col]
                # negative_example éš�æœºé€‰
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

            sub_dataset.drop(columns=["positive_example_1","positive_example_2",
                          "negative_example_1","negative_example_2"], inplace=True)

            flatten.append(sub_dataset)

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
    print(f"dataframe.shape: {dataframe.shape}")
    print(f"dataframe.columns: {dataframe.columns}")
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


!ls -al


!ls -al output


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
    """åœ¨å½“å‰�è¿›ç¨‹å�¯è§�çš„ GPU ä¸Šè·‘ vLLM æ�¨ç�†"""
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
    # é™�åˆ¶è¯¥è¿›ç¨‹å�ªçœ‹åˆ°ä¸€å¼  GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"[Worker {device_id}] Running on GPU {device_id}, data size={len(df_slice)}")

    preds = run_inference_on_device(df_slice)
    return_dict[device_id] = preds


def main():
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")

    # ë‘˜ ì¤‘ í•˜ë‚˜ ì„ íƒ�
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

    # åˆ‡åˆ†æ•°æ�®
    mid = len(test_dataframe) // 2
    df0 = test_dataframe.iloc[:mid].reset_index(drop=True)
    df1 = test_dataframe.iloc[mid:].reset_index(drop=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    # ä¸¤ä¸ªè¿›ç¨‹å¹¶è¡Œ
    p0 = mp.Process(target=worker, args=(0, df0, return_dict))
    p1 = mp.Process(target=worker, args=(1, df1, return_dict))
    p0.start()
    p1.start()
    p0.join()
    p1.join()

    # å�ˆå¹¶ç»“æ�œ
    predictions = pd.concat([return_dict[0], return_dict[1]], ignore_index=True)

    # æ�„å»º submission
    submission = predictions[["row_id", POSITIVE_ANSWER]].rename(columns={POSITIVE_ANSWER: "rule_violation"})
    rq = submission['rule_violation'].rank(method='average') / (len(submission) + 1)
    submission['rule_violation'] = rq

    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("âœ… Saved submission.csv using Qwen 2.5 0.5B model only")


if __name__ == "__main__":
    main()


!python inference.py


!head /kaggle/working/submission.csv

