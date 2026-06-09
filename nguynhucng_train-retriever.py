!pip uninstall -y tensorflow && pip install tensorflow-cpu
!pip install datasets typer omegaconf peft sentence-transformers wandb


%%writefile /kaggle/working/cfg.yaml
input_dir: "/kaggle/input/data-eedi/exp_output"
save_dir: "/kaggle/working/"
best_model_dir: "/kaggle/working/best_models"
train_biencoder:
  model_name: "dunzhang/stella_en_1.5B_v5"
  input_name: "data_kd.csv"
  output_dir: "output_bi_1.5B"
  is_lora: true
  load_in_4bit: false
  mini_batch_size: 1
  seed: 42
  lora_config:
    r: 48
    lora_alpha: 96
  hard_negative_params:
    range_min: 512
    num_negatives: 2
    batch_size: 32
  train_args:
    num_train_epochs: 1.0
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 4
    learning_rate: 0.001
    warmup_steps: 0
    metric_for_best_model: val_cosine_recall@100
    greater_is_better: true
    save_strategy: "no"
    save_steps: 2
    lr_scheduler_type: "cosine"
    save_total_limit: 1
    logging_steps: 1
    report_to: wandb
    bf16: true
    local_rank: 2
    eval_strategy: 'no'
    do_eval: false
    do_train: true


!accelerate config default


%%writefile /root/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_use_orig_params: true
machine_rank: 0
gpu_ids: all
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false


%%writefile /kaggle/working/train_retriever.py
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Set

import pandas as pd
import typer
from datasets import Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    models
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

from transformers import set_seed
from transformers.training_args import ParallelMode

import torch
import torch.distributed as dist

import json
import wandb
import typer
# Template for formatting the prompt
PROMPT_FORMAT: str = """Subject: {SubjectName}
Construct: {ConstructName}
Question: {QuestionText}
CorrectAnswer: {Correct}
IncorrectAnswer: {Answer}
IncorrectReason: {kd}"""


def create_val(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Create validation dataset by merging dataframe with mapping and adding labels.

    Args:
        df: Input DataFrame containing the base data
        mapping: DataFrame containing misconception mapping information

    Returns:
        DataFrame with processed validation data
    """
    df = df.merge(mapping, how="cross")
    df["label"] = 0
    df.loc[df["MisconceptionId_x"] == df["MisconceptionId_y"], "label"] = 1
    target_cols = ["prompt", "MisconceptionName_y", "label"]
    df = df[target_cols].rename(columns={"MisconceptionName_y": "MisconceptionName"})
    return df


# def create_evaluator(df: pd.DataFrame, name: str = "train") -> InformationRetrievalEvaluator:
#     """
#     Create an evaluator for information retrieval tasks.

#     Args:
#         df: DataFrame containing prompts, misconception names, and labels
#         name: Name identifier for the evaluator

#     Returns:
#         Configured InformationRetrievalEvaluator object
#     """
#     relevant_docs: DefaultDict[str, Set[str]] = defaultdict(set)
#     queries: Dict[str, str] = {str(k): v for k, v in enumerate(df["prompt"].unique())}
#     corpus: Dict[str, str] = {str(k): v for k, v in enumerate(df["MisconceptionName"].unique())}

#     # Create reverse mappings for efficient lookup
#     qid_dict: Dict[str, str] = {v: k for k, v in queries.items()}
#     cid_dict: Dict[str, str] = {v: k for k, v in corpus.items()}

#     # Build relevant documents mapping
#     for prompt, g in df.groupby("prompt"):
#         for mis_name, label in g[["MisconceptionName", "label"]].values:
#             if label == 1:
#                 qid = qid_dict[str(prompt)]
#                 cid = cid_dict[mis_name]
#                 relevant_docs[qid].add(cid)

#     return InformationRetrievalEvaluator(
#         queries=queries,
#         corpus=corpus,
#         relevant_docs=relevant_docs,
#         name=name,
#         map_at_k=[25],
#         mrr_at_k=[25],
#         precision_recall_at_k=[50, 100, 150, 200],
#         ndcg_at_k=[25],
#         accuracy_at_k=[25],
#     )


def main(
    fold: int = typer.Option(..., help="Fold number for cross-validation"),
    config= "/kaggle/working/cfg.yaml",
) -> None:
    """
    Main training function for the bi-encoder model.

    Args:
        fold: Cross-validation fold number
        config: Path to configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.train_biencoder
    set_seed(params.seed)

    # Load and prepare data
    df = pd.read_csv(Path(cfg.input_dir) / params.input_name)
    mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
    df["prompt"] = df.apply(lambda x: PROMPT_FORMAT.format(**x), axis=1)

    # Split data into train and validation sets
    train_df = df.loc[df.fold != fold].copy()
    val_df = df.loc[(df.fold == fold) & (df.original)].copy()
    # val_df = create_val(val_df, mapping)

    # Create dataset for training
    train_dset = Dataset.from_dict(
        {
            "anchor": train_df["prompt"].tolist(),
            "positive": train_df["MisconceptionName"].tolist(),
        }
    )

    # Setup paths and wandb
    name = f"fold_{fold}"
    output_dir = str(Path(cfg.save_dir) / params.output_dir / name)
    best_model_path = str(Path(cfg.best_model_dir) / params.output_dir / name)
    wandb.login(key="ccd07261eef86e04beb9d6f9e459d8995bdc4b16")
    wandb.init(project="eedi-biencoder", name=f"{name}_{params.model_name.split('/')[-1]}")

    # Initialize model
    model = SentenceTransformer(
        params.model_name,
        trust_remote_code=True,
        model_kwargs={"load_in_4bit": params.load_in_4bit},
    )
    # Perform hard negative mining
    train_dset = Dataset.load_from_disk("/kaggle/working/train_dset_hard_negatives")
    # Add LoRA adapter if specified
    if params.is_lora:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            target_modules="all-linear",
            lora_dropout=0.01,
            **params["lora_config"],
        )
        # Apply LoRA BEFORE dispatching
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
        model[0].auto_model.print_trainable_parameters()
    
        # # Dispatch model AFTER applying LoRA
        # try:
        #     device_map_path = "/kaggle/working/device_map.json"
        #     with open(device_map_path, "r") as f:
        #         device_map = json.load(f)
        #         device_map = {k: str(v) for k, v in device_map.items()}
        #     model[0].auto_model = dispatch_model(model[0].auto_model, device_map=device_map)
        #     print("✅ LoRA model dispatched to multiple GPUs.")
        # except Exception as e:
        #     print("⚠️ Failed to dispatch LoRA model:", e)
    # Setup loss function and evaluator
    loss = losses.CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=params.mini_batch_size, show_progress_bar=True
    )
    # val_evaluator = create_evaluator(val_df, name="val")
    
    # Configure training arguments
    args = SentenceTransformerTrainingArguments(
        **params.train_args,
        output_dir=output_dir,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        fsdp=["full_shard", "auto_wrap"],
        fsdp_config= "/root/.cache/huggingface/accelerate/default_config.yaml",
    )
    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_dset,
        loss=loss,
        # evaluator=val_evaluator,
    )
    trainer.train()
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(best_model_path)

if __name__ == "__main__":
    typer.run(main)


!CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu /kaggle/working/train_retriever.py --fold 0


from accelerate import Accelerator

accelerator = Accelerator()

print(f"Thiết bị hiện tại: {accelerator.device}")
print(f"Process index: {accelerator.process_index}")
print(f"Có phải main process không? {accelerator.is_main_process}")

