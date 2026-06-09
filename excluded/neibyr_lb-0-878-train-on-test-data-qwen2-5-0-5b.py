%%writefile constants.py
BASE_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/0.5b-instruct-gptq-int4/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

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
        
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        
        optim="paged_adamw_8bit",
        learning_rate=5e-5,
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
from datasets import Dataset
from vllm.lora.request import LoRARequest
from utils import build_dataset
from constants import BASE_MODEL_PATH, LORA_PATH, DATA_PATH, POSITIVE_ANSWER, NEGATIVE_ANSWER


def main():
    llm = vllm.LLM(
        BASE_MODEL_PATH,
        quantization="gptq",
        tensor_parallel_size=1,#torch.cuda.device_count(), see issue https://github.com/vllm-project/vllm/issues/2699?ysclid=me2q9w8mby837333215
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        max_model_len=4096,
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
        lora_request=LoRARequest("default", 1, LORA_PATH)
    )
    
    log_probs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]
    
    predictions = pd.DataFrame(log_probs)[[POSITIVE_ANSWER, NEGATIVE_ANSWER]]
    predictions["row_id"] = test_dataframe["row_id"]
    submission = predictions[["row_id", POSITIVE_ANSWER]].rename(columns={POSITIVE_ANSWER: "rule_violation"})
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()


%%writefile accelerate_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  train_batch_size: 64
  train_micro_batch_size_per_gpu: 16
  
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


!head submission.csv

