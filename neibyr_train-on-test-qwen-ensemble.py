%%writefile constants.py
BASE_INPUT_PATH = "/kaggle/input"
LORA_PATH = "lora"
PRE_TRAIN_LORA_PATH = f"{BASE_INPUT_PATH}/lora-5-models/kaggle/working/lora/"
DATA_PATH = f"{BASE_INPUT_PATH}/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."


%%writefile utils.py
import pandas as pd
from datasets import Dataset
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT


def build_prompt(row):
    return f"""
{BASE_PROMPT}

r/{row["subreddit"]} rule: {row["rule"]}

Comment: {row["body"]}
---
{COMPLETE_PHRASE}"""


def get_dataframe_to_train(data_path, seed=42):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []
    # flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
            sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_dataset)

    dataframe = pd.concat(flatten, axis=0).sample(frac=1, random_state=seed)
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    return dataframe


def build_dataset(dataframe):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    columns = ["prompt", "rule"]
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
    return dataset


%%writefile train.py
import argparse
import gc
import pandas as pd
import time
import torch
import torch.distributed as dist

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.utils import is_torch_bf16_gpu_available

from utils import build_dataset, get_dataframe_to_train
from constants import DATA_PATH, PRE_TRAIN_LORA_PATH, BASE_INPUT_PATH, LORA_PATH


class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_time_seconds):
        self.max_time_seconds = max_time_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time_seconds:
            control.should_training_stop = True
            print(f"Training stopped: time limit of {self.max_time_seconds} seconds exceeded.")


def cleanup_resources():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_model(model_path, batch_size=32, per_device_train_batch_size=8, train_time=3600, seed=42):
    dataframe = get_dataframe_to_train(DATA_PATH, seed)
    dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)
    train_dataset = build_dataset(dataframe)

    adapter_path = f"{PRE_TRAIN_LORA_PATH}/{model_path}"
    
    training_args = SFTConfig(
        num_train_epochs=1,
        
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=batch_size // per_device_train_batch_size,
        
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
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

    base_model = AutoModelForCausalLM.from_pretrained(
        f"{BASE_INPUT_PATH}/{model_path}",
        dtype=torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(f"{BASE_INPUT_PATH}/{model_path}", use_fast=False)
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    peft_config = None

    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     bias="none",
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     task_type="CAUSAL_LM",
    # )

    model.print_trainable_parameters()
    
    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        callbacks=[TimeLimitCallback(max_time_seconds=train_time)]
    )
    
    trainer.train()
    trainer.save_model(f"{LORA_PATH}/{model_path}")

    del trainer
    cleanup_resources()
    
    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--per_device_train_batch_size", type=int, required=False, default=8)
    parser.add_argument("--train_time", type=int, required=False, default=3600)
    parser.add_argument("--seed", type=int, required=False, default=42)
    args = parser.parse_args()
    
    train_model(
        args.model_path,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        train_time=args.train_time,
        seed=args.seed,
    )



%%writefile inference.py
import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import vllm
import torch
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from datasets import Dataset
from vllm.lora.request import LoRARequest

from train import cleanup_resources
from utils import build_dataset
from constants import BASE_INPUT_PATH, LORA_PATH, DATA_PATH, POSITIVE_ANSWER, NEGATIVE_ANSWER


def inference_model(model_path, quantization="gptq"):
    llm = vllm.LLM(
        f"{BASE_INPUT_PATH}/{model_path}",
        quantization=quantization,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=1024,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=64,
    )
    
    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE_ANSWER, NEGATIVE_ANSWER])
    
    test_dataframe = pd.read_csv(f"{DATA_PATH}/test.csv")
    test_dataset = build_dataset(test_dataframe)

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
        lora_request=LoRARequest("default", 1, f"{LORA_PATH}/{model_path}")
    )
    
    log_probs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]

    predictions = pd.DataFrame(log_probs)[[POSITIVE_ANSWER, NEGATIVE_ANSWER]]
    predictions["row_id"] = test_dataframe["row_id"]
    submission = predictions[["row_id", POSITIVE_ANSWER]].rename(columns={POSITIVE_ANSWER: "rule_violation"})
    
    del llm
    cleanup_resources()
    
    return submission


def main(model_path, quantization="gptq", submission_file_name="submission.csv"):
    submission = inference_model(model_path, quantization=quantization)
    submission.to_csv(submission_file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--quantization", type=str, required=False, default=None)
    parser.add_argument("--submission_file_name", type=str, required=False, default="submission.csv")
    args = parser.parse_args()
    
    main(args.model_path, args.quantization, args.submission_file_name)


%%writefile accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
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


import subprocess
import pandas as pd

from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class Config:
    model_path: str
    submission_path: str
    train_time: int = 7200
    batch_size: int = 32
    per_device_train_batch_size: int = 8
    quantization: str | None = "gptq"
    submission_weight: float | int = 1
    seed: int = 42


submission = True
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
if len(test) == 10:
    submission = False


models = [
    Config(
        model_path="qwen-3/transformers/1.7b-gptq-int8/1",
        submission_path="submission_qwen_3_1.7b.csv",
        train_time=6250,
        submission_weight=0.17,
    ),
    Config(
        model_path="qwen2.5/transformers/3b-instruct-gptq-int8/1",
        submission_path="42_submission_qwen_2.5_3b.csv",
        train_time=7500,
        submission_weight=0.2,
        seed=42,
    ),
    Config(
        model_path="qwen2.5/transformers/7b-instruct-gptq-int8/1",
        submission_path="43_submission_qwen_2.5_7b.csv",
        train_time=7500,
        per_device_train_batch_size=4,
        submission_weight=0.3,
        seed=43,
    ),
    Config(
        model_path="qwen2.5/transformers/7b-instruct-gptq-int8/1",
        submission_path="44_submission_qwen_2.5_7b.csv",
        train_time=7500,
        per_device_train_batch_size=4,
        submission_weight=0.3,
        seed=44,
    ),
    Config(
        model_path="qwen2.5/transformers/7b-instruct-gptq-int8/1",
        submission_path="45_submission_qwen_2.5_7b.csv",
        train_time=7500,
        per_device_train_batch_size=4,
        submission_weight=0.3,
        seed=45,
    ),
]


for config in tqdm(models, desc="Train each model"):
    train_model_cmd = [
        "accelerate", "launch",
        "--config_file", "accelerate_config.yaml",
        "train.py",
        "--model_path", str(config.model_path),
        "--batch_size", str(config.batch_size),
        "--per_device_train_batch_size", str(config.per_device_train_batch_size),
        "--train_time", str(config.train_time),
        "--seed", str(config.seed),
    ]
    if submission:
        print("="*50)
        print(config.model_path)
        subprocess.run(train_model_cmd, check=True)
        print("="*50)


for config in tqdm(models, desc="Inference each model"):
    inference_model_cmd = [
        "python", "inference.py",
        "--model_path", str(config.model_path),
        "--quantization", str(config.quantization),
        "--submission_file_name", str(config.submission_path),
    ]
    if submission:
        subprocess.run(inference_model_cmd, check=True)


blend = 0

if submission:
    for config in tqdm(models, desc="Blend solutions"):
        sub = pd.read_csv(config.submission_path).sort_values("row_id", ignore_index=True)
        sub["rule_violation"] = sub["rule_violation"].rank(method='average') / (len(sub) + 1)
        sub["rule_violation"] = config.submission_weight * sub["rule_violation"]
        blend += sub["rule_violation"]
else:
    sub = test

submission = pd.DataFrame()
submission["row_id"] = sub["row_id"]
submission["rule_violation"] = blend

submission.to_csv('submission.csv', index=False)


submission

