!pip install peft
!pip install git+https://github.com/huggingface/transformers -U
!pip install accelerate
!pip install -i https://pypi.org/simple/ bitsandbytes
!pip install datasets
!pip install torch
!pip install -U transformers
!pip install tqdm
!pip install omegaconf
!pip install bitsandbytes
!pip install  -q git+https://github.com/huggingface/peft


from typing import Any, Dict, List, Tuple, Type, Union

import math
import os
import random
import re
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from bitsandbytes.optim import AdamW
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GemmaModel, 
    GemmaTokenizer,
    pipeline,
)
from transformers.utils import is_torch_available


@dataclass
class Config:
    save_name: str = "third_test1"
    finetuned_model_path: str = ""
    model_name: str = "google/gemma-2-2b-it"
    use_fast_tokenizer: bool = True
    gradient_checkpointing: bool = True

    # LoRA
    use_lora: bool = True
    lora_dtype: str = "4-bit"  # "4-bit" or "8-bit"
    target_modules: list = ("k_proj",)  # ("q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj")
    lora_rank: int = 1
    lora_alpha: int = 12
    lora_dropout: float = 0.1

    # Train Block
    use_train_block: bool = False
    train_block: str = "first"  # "first" or "last"

    # Data
    batch_size: int = 1
    sample_size: int = 4000
    max_length: int = 5120

    # Train
    epochs: int = 2
    weight_decay: float = 0.1
    lr: float = 1e-6
    gradient_accumulation_steps: int = 2
    lr_scheduler_type: str = "inverse_sqrt"  # "linear", "inverse_sqrt", "polynomial", "cosine_with_restarts"
    warmup_ratio: float = 0.05
    logging_steps: int = 100
    output_dir: str = "./store/"
    save_steps: int = 10000

    # Evaluation
    eval_batch_size: int = 1
    similarity_model_name: str = "BAAI/bge-m3"

    # Setting
    use_wandb: bool = True
    wandb_project: str = "Nexo"


config = Config()

config_dict = OmegaConf.structured(config)

yaml_path = "config.yaml"
OmegaConf.save(config_dict, yaml_path)


@dataclass
class DatasetConfig:
    # ***Reasoning (Ko)***
    hae_rae_cot: str = "HAERAE-HUB/HAE-RAE-COT-1.5M"  # [(1.59M) | (cc-by-4.0)]
    magpie_ko_qwen: str = "werty1248/Magpie-Ko-Qwen2.5-Reasoning-Raw"  # [(112k) | (apache-2.0)]
    cot_collection: str = "heegyu/CoT-collection-ko"  # [(77.2k) | (cc-by-2.0)]

    # ***Instruction Tuning (Ko)***
    real_qa: str = "beomi/KoAlpaca-RealQA"  # [(18.5k) | (cc-by-sa-4.0)]
    openorca: str = "squarelike/OpenOrca-gugugo-ko"  # [(2.24M) | (mit)]
    korean_instruction: str = "jojo0217/korean_rlhf_dataset"  # [(107k) | (apache-2.0)]
    kopen_hq_hermes: str = "MarkrAI/KOpen-HQ-Hermes-2.5-60K"  # [(60.1k) | (mit)]

    # ***Simple QA***
    kullm_v2: str = "nlpai-lab/kullm-v2"  # [(153k) | (apache-2.0)]
    oig_chip2: str = "heegyu/OIG-small-chip2-ko"  # [(210k) | (apache-2.0)]
    ko_code_alpace_qa: str = "CarrotAI/ko-code-alpaca-QA"  # [(9.7k) | (apache-2.0)]
    hc3_ko: str = "nayohan/HC3-ko"  # [(24.3k) | (cc-by-sa-4.0)]

    # ***Math***
    math_gpt4_ko: str = "nayohan/math-gpt-4o-200k-ko"  # [(200k) | (mit)]
    kopen_platypus: str = "kyujinpy/KOpen-platypus"  # [(24.9k) | (cc-by-4.0)]
    math_college_ko: str = "nayohan/Maths-College-ko"  # [(48.5k) | (apache-2.0)]

    # ***Single-Turn Dialogue***
    korean_common: str = "CarrotAI/Korean-Common"  # [(102k) | (cc-by-2.0)]
    korean_conversation: str = "jojo0217/korean_safe_conversation"  # [(27k) | (apache-2.0)]

    # ***Multi-Turn Dialogue***
    ko_lima_vicuna: str = "changpt/ko-lima-vicuna"  # [(1.03k) | (cc-by-2.0)]
    kovast: str = "maywell/koVast"  # [(685k) | (mit)]
    share_gpt: str = "dbdu/ShareGPT-74k-ko"  # [(50k) | (cc-by-2.0)]

    # ***Medical Dataset*** (Need to upload to Hugging Face)
    pub_med_qa_ko: str = "/root/MAIN_STORE/train_dataset/14.pub_med_qa_full"  # [(nk) | (mit)]
    medmcqa_ko: str = "/root/MAIN_STORE/train_dataset/15.medmcqa"  # [(nk) | (apache-2.0)]
    gen_med_gpt: str = "ChuGyouk/GenMedGPT-5k-ko"  # [(5.45k) | (mit)]
    ko_med_instruct: str = "ChuGyouk/KoMedInstruct-52k"  # [(52k) | (apache-2.0)]


@dataclass
class EvaluationDatasetConfig:
    # ***Additional Datasets***
    kmmlu: str = "HAERAE-HUB/KMMLU"
    hae_rae_bench: str = "HAERAE-HUB/HAE_RAE_BENCH_2.0"
    gsm8k_ko: str = "HAERAE-HUB/HAE_RAE_BENCH_2.0"
    med_exp_qa: str = "ChuGyouk/MedExpQA-Kor"
    pub_med_qa: str = "ChuGyouk/PubMedQA-test-Ko"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.99


config = Config()
dataset_config = DatasetConfig()
eval_config = EvaluationDatasetConfig()
generate_config = GenerationConfig()

config.dataset = dataset_config
config.eval_task = eval_config
config.generate = generate_config
config.device = "cuda" if torch.cuda.is_available() else "cpu"


from typing import Type

from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# ==================== Dataset Loader Registry ====================
DATASET_LOADERS = {}


def dataset_loader(name: str):
    """Decorator for registering dataset loaders."""

    def decorator(cls: Type["BaseDatasetLoader"]):
        DATASET_LOADERS[name] = cls
        return cls

    return decorator


# ==================== Base Dataset Loader ====================
class BaseDatasetLoader(ABC):
    """Base class for dataset loaders. Each dataset should implement its own loader."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.column_name = "total_prompt"

    @abstractmethod
    def load(self, path: str, name: str) -> Dataset:
        """Loads the dataset. Must be implemented in subclasses."""
        pass

    def make_prompt(self, batch, system_instruction: str = ""):
        """Generates prompts for each dataset. Can be overridden if needed."""
        return batch

    def add_ids(self, batch, batch_start_idx_list, name):
        """Adds unique IDs to each data entry."""
        batch_start_idx = batch_start_idx_list[0]
        batch_size = len(batch["input_ids"])
        batch["id"] = [name + "_" + str(batch_start_idx + i) for i in range(batch_size)]
        return batch

    def remove_unnecessary_columns(self, dataset):
        keep_columns = ["input_ids", "labels"]
        all_columns = dataset.column_names
        remove_columns = [col for col in all_columns if col not in keep_columns]
        return dataset.remove_columns(remove_columns)

    def preprocess_dataset(self, dataset, name, sample_size):
        """Applies sampling and processing to the dataset."""
        if sample_size < len(dataset):
            dataset = dataset.shuffle().select(range(sample_size))

        dataset = dataset.map(self.make_prompt, batched=True)
        dataset = dataset.map(
            lambda batch, idx: self.add_ids(batch, idx, name),
            with_indices=True,
            batched=True,
            desc=name,
        )
        return self.remove_unnecessary_columns(dataset)


# ==================== DatasetManager ====================
class DatasetManager:
    """Manages dataset loading and retrieval."""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.datasets = {}

    def get_loader(self, name: str):
        """Returns the corresponding dataset loader instance."""
        loader_cls = DATASET_LOADERS.get(name)
        if not loader_cls:
            raise NotImplementedError(f"Dataset loader for {name} is not registered")
        return loader_cls(self.tokenizer)

    def load_dataset(self, name: str, path: str, sample_size: int, max_length: int) -> Dataset:
        """Loads the dataset if not already loaded and caches it."""
        if name in self.datasets:
            return self.datasets[name]

        loader = self.get_loader(name)
        dataset = loader.load(path=path, name=name, sample_size=sample_size, max_length=max_length)
        self.datasets[name] = dataset
        return dataset

    def get_datasets(self):
        """Returns all currently loaded datasets."""
        return self.datasets


# ==================== Example Dataset Loader ====================
@dataset_loader("ko_lima_vicuna")
class KoLimaVicunaLoader(BaseDatasetLoader):
    """Loader for the ko_lima_vicuna dataset."""

    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        self.preprocess_dataset(dataset, name, sample_size)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        conversations = batch["conversations"]
        input_ids_list = []
        labels = []
        prompts = []
        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for conversation in conversations:
            messages = []
            for message in conversation:
                if message["from"] == "human":
                    messages.append({"role": "user", "content": message["value"]})
                else:
                    messages.append({"role": "assistant", "content": message["value"]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("hae_rae_cot")
class HaeRaeCotLoader(BaseDatasetLoader):
    """Loader for the hae_rae_cot dataset."""

    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["Question"]
        cot = batch["CoT_Rationale"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": cot[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("magpie_ko_qwen")
class MagpieKoQwenLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        # NOTE: This huggingface dataset has a bug in dataset size and configuration size
        dataset = load_dataset(path, verification_mode="no_checks")["train"]
        dataset = dataset.map(self.make_prompt, batched=True)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["question"]
        answers = batch["answer"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("cot_collection")
class CotCollectionLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["source"]
        answers = batch["rationale"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("real_qa")
class RealQaLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["question"]
        answers = batch["answer"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("openorca")
class OpenOrcaLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        system_prompts = batch["system_prompt"]
        questions = batch["question"]
        answers = batch["response"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_prompts[idx]
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("korean_instruction")
class KoreanInstructionLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["instruction"]
        answers = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("korean_common")
class KoreanCommonLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        conversations = batch["conversations"]
        input_ids_list = []
        labels = []
        prompts = []
        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for conversation in conversations:
            messages = []
            for message in conversation:
                if message["from"] == "human":
                    messages.append({"role": "user", "content": message["value"]})
                else:
                    messages.append({"role": "assistant", "content": message["value"]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("kullm-v2")
class KullmV2Loader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["instruction"]
        answers = batch["output"]
        inputs = batch["input"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx] + "\n" + inputs[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("oig_chip2")
class OigChip2Loader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["user_translated"]
        answers = batch["chip2_translated"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("kopen_hq_hermes")
class KopenHqHermesLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        sys_instructions = batch["input"]
        questions = batch["instruction"]
        answers = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + sys_instructions[idx]
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("math_gpt4_ko")
class MathGpt4KoLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["prompt"]
        answers = batch["response"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("kopen_platypus")
class KopenPlatypusLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["instruction"]
        answers = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("math_college_ko")
class MathCollegeKoLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["instruction"]
        answers = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("kovast")
class KovastLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        conversations = batch["conversations"]
        input_ids_list = []
        labels = []
        prompts = []
        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for conversation in conversations:
            messages = []
            for message in conversation:
                if message["from"] == "user":
                    messages.append({"role": "user", "content": message["value"]})
                elif message["from"] == "gpt":
                    messages.append({"role": "assistant", "content": message["value"]})
            try:
                prompt = (
                    start_system_prompt
                    + system_instruction
                    + end_system_prompt
                    + self.tokenizer.apply_chat_template(messages, tokenize=False)
                )
            except:
                prompt = ""
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("share_gpt")
class ShareGptLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        conversations = batch["conversations"]
        input_ids_list = []
        labels = []
        prompts = []
        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for conversation in conversations:
            messages = []
            for message in conversation:
                if message["from"] == "human":
                    messages.append({"role": "user", "content": message["value"]})
                else:
                    messages.append({"role": "assistant", "content": message["value"]})
            try:
                prompt = (
                    start_system_prompt
                    + system_instruction
                    + end_system_prompt
                    + self.tokenizer.apply_chat_template(messages, tokenize=False)
                )
            except:
                prompt = ""
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("korean_conversation")
class KoreanConversationLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["instruction"]
        answers = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("ko_code_alpace_qa")
class KoCodeAlpaceQaLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["instruction"]
        answers = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("hc3_ko")
class Hc3KoLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["question"]
        answers = batch["human_answers"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answers[idx][0]})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=False, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("pub_med_qa_ko")
class PubMedQaKoLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        # NOTE: change
        dataset = load_from_disk(path)
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["messages"]
        answers = batch["answer_trans"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []

            user_text = ""
            reference = ""
            for message in questions[idx]:
                if message["role"] == "user":
                    user_text += message["content"] + "\n"
                else:
                    reference += message["content"] + "\n"
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": answers[idx]})
            prompt = (
                start_system_prompt
                + "ì•„ë�˜ ì°¸ì¡°ë¥¼ ì°¸ê³ í•˜ì—¬ ë‹µë³€ì�„ í•´ì£¼ì„¸ìš”."
                + "\n\n"
                + reference
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=True, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }

    def preprocess_dataset(self, dataset, name, sample_size):
        """Applies sampling and processing to the dataset."""
        if sample_size < len(dataset):
            dataset = dataset.shuffle().select(range(sample_size))

        dataset = dataset.map(self.make_prompt, batched=True)
        dataset = dataset.map(
            lambda batch, idx: self.add_ids(batch, idx, name),
            with_indices=True,
            batched=True,
            desc=name,
        )
        return self.remove_unnecessary_columns(dataset)


@dataset_loader("medmcqa_ko")
class MedMcQaKoLoader(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        # NOTE: change
        dataset = load_from_disk(path)
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        """Processes conversation data and formats it into prompts."""
        questions = batch["messages"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []

            user_text = ""
            answer = ""
            for message in questions[idx]:
                if message["role"] == "user":
                    user_text += message["content"] + "\n"
                else:
                    answer += message["content"] + "\n"
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": answer})
            prompt = (
                start_system_prompt
                + system_instruction
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=True, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }

    def preprocess_dataset(self, dataset, name, sample_size):
        """Applies sampling and processing to the dataset."""
        if sample_size < len(dataset):
            dataset = dataset.shuffle().select(range(sample_size))

        dataset = dataset.map(self.make_prompt, batched=True)
        dataset = dataset.map(
            lambda batch, idx: self.add_ids(batch, idx, name),
            with_indices=True,
            batched=True,
            desc=name,
        )
        return self.remove_unnecessary_columns(dataset)


@dataset_loader("gen_med_gpt")
class GenMedGpt(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        questions = batch["input"]
        instructions = batch["instruction"]
        answer = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []

            messages.append({"role": "user", "content": questions[idx]})
            messages.append({"role": "assistant", "content": answer[idx]})

            prompt = (
                start_system_prompt
                + instructions[idx]
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=True, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }


@dataset_loader("ko_med_instruct")
class KoMedInstruct(BaseDatasetLoader):
    def load(self, path: str, name: str, sample_size: int, max_length: int) -> Dataset:
        self.max_length = max_length
        dataset = load_dataset(path)["train"]
        dataset = self.preprocess_dataset(dataset, name, sample_size)
        dataset = self.remove_unnecessary_columns(dataset)
        return dataset

    def make_prompt(self, batch, system_instruction=""):
        questions = batch["input"]
        instructions = batch["instruction"]
        answer = batch["output"]
        input_ids_list = []
        labels = []
        prompts = []

        start_system_prompt = "<start_of_turn>system\n"
        end_system_prompt = "<end_of_turn>\n"
        for idx in range(len(questions)):
            messages = []
            user_text = "" if questions[idx] == "<noinput>" else questions[idx]
            messages.append({"role": "user", "content": instructions[idx] + "\n" + user_text})
            messages.append({"role": "assistant", "content": answer[idx]})

            prompt = (
                start_system_prompt
                + instructions[idx]
                + end_system_prompt
                + self.tokenizer.apply_chat_template(messages, tokenize=False)
            )
            prompts.append(prompt)
            encoded = self.tokenizer(
                prompt, padding=True, return_tensors="pt", max_length=self.max_length, truncation=True
            )

            input_ids_list.append(encoded["input_ids"].squeeze(0))
            labels.append(encoded["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "labels": labels,
        }

# ==================== TrainDataset ====================
class TrainDataset:
    """Combines multiple datasets into a single dataset for training."""

    def __init__(self, config, dataset_manager: DatasetManager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.dataset = self._combine_datasets()

    def _combine_datasets(self) -> DatasetDict:
        """Loads and merges datasets based on the configuration."""
        include_datasets = self.config.datasets
        datasets = []
        for name, path in tqdm(include_datasets.items(), desc="Loading datasets"):
            dataset = self.dataset_manager.load_dataset(name, path, self.config.sample_size, self.config.max_length)
            datasets.append(dataset)

        return concatenate_datasets(datasets)

    def __getitem__(self, idx: int):
        """Allows access to the combined dataset as a list."""
        return {
            "input_ids": self.dataset[idx]["input_ids"],
            "labels": self.dataset[idx]["labels"],
        }

    def __len__(self):
        """Returns the total number of entries in all datasets."""
        return len(self.dataset)


def get_trainloader(config):
    """Returns the combined dataset for training."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset_manager = DatasetManager(config, tokenizer)
    train_dataset = TrainDataset(config, dataset_manager)
    return DataLoader(
        train_dataset,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        batch_size=config.batch_size,
        shuffle=True,
    )



from typing import Dict, List, Tuple, Union

import math
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from bitsandbytes.optim import AdamW
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.utils import is_torch_available


def load_model(config):
    """
    Loads a model based on the provided configuration.

    Args:
        config (Dict): Configuration dictionary containing model details.

    Returns:
        model (AutoModelForCausalLM): Loaded model with or without LoRA, supporting fine-tuned models.
    """
    model_config = AutoConfig.from_pretrained(config.model_name)

    # Check if a fine-tuned model exists
    if config.finetuned_model_path and os.path.exists(config.finetuned_model_path):
        print(f"âœ… Loading fine-tuned model from {config.finetuned_model_path}")

        # If LoRA is enabled, we need to attach the LoRA adapter
        if config.use_lora:
            print("ğŸ”„ LoRA detected. Loading base model and applying LoRA adapter...")
            base_model = AutoModelForCausalLM.from_pretrained(config.model_name, config=model_config)
            model = PeftModel.from_pretrained(base_model, config.finetuned_model_path)
        else:
            # Load full fine-tuned model
            model = AutoModelForCausalLM.from_pretrained(config.finetuned_model_path, config=model_config)

        return model

    print("ğŸ”„ No fine-tuned model found. Loading base model...")

    if not config.use_lora:
        # Load the model without PEFT
        return AutoModelForCausalLM.from_pretrained(
            config.model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
        )

    # Load the model with LoRA
    quantization_config = None
    if config.lora_dtype == "4-bit":
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
    elif config.lora_dtype == "8-bit":
        quantization_config = {"load_in_8bit": True}
    else:
        raise ValueError("Invalid LoRA dtype.")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=OmegaConf.to_container(config.target_modules, resolve=True),
    )

    model = get_peft_model(model, peft_config)
    count_trainable_parameters(model)

    return model


class Trainer:
    """Trainer class for training the model."""

    def __init__(self, config: Dict, train_dataloader: DataLoader):
        set_seed()
        self.config = config
        self.use_wandb = config.use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.use_wandb:
            wandb_setting(config)

        self.train_dataloader = train_dataloader

        self.model = load_model().to(self.device)
        self.tokenizer = self.load_tokenizer()
        self.optimizer, self.lr_scheduler = self.prepare_optimizer_scheduler()

        self.accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def load_tokenizer(self):
        """Loads the tokenizer."""
        return AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=self.config.use_fast_tokenizer,
        )

    def prepare_optimizer_scheduler(self):
        """Prepares the optimizer and scheduler."""
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = (
            AdamW(
                optimizer_grouped_parameters,
                lr=self.config.lr,
                optim_bits=8 if self.config.lora_dtype == "8-bit" else 8,
                is_paged=True,
            )
            if (self.config.use_lora)
            else torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.lr)
        )

        num_training_steps = self.config.epochs * math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
        )

        return optimizer, lr_scheduler

    def train_step(self, batch):
        """Train one step"""
        batch = {k: v.to(self.device) for k, v in batch.items()}  # ğŸ”¥ Move batch to device

        with self.accelerator.accumulate(self.model):
            outputs = self.model(**batch)
            loss = outputs.loss

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

        return loss.detach().float()

    def train(self):
        """Training loop"""
        save_steps = self.config.save_steps

        progress_bar = tqdm(
            range(self.config.epochs * len(self.train_dataloader)),
            disable=not self.accelerator.is_local_main_process,
        )

        total_step = 0
        c = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                total_loss += loss
                total_step += 1

                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "Epoch": epoch,
                        "Step": step,
                        "Loss": f"{loss.item():.4f}",
                        "LR": f"{self.lr_scheduler.get_last_lr()[0]:.5f}",
                    }
                )

                if total_step % save_steps == 0 and total_step > 0:
                    self.save_model(step=total_step)

                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "step": total_step,
                            "loss": loss.item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                        }
                    )

                if c == 10:
                    pass
                    break
                c += 1

        self.save_model(step=total_step)

        if self.use_wandb:
            wandb.finish()

    def save_model(self, step: int) -> None:
        save_path = os.path.join(self.config.output_dir, f"{self.config.save_name}_step_{step}")
        os.makedirs(save_path, exist_ok=True)
        import json

        if self.config.use_lora:
            # ğŸ”¥ LoRA 
            self.model.save_pretrained(save_path)
            print(f"âœ… LoRA adapter saved at {save_path}")
        else:
            # ğŸ”¥ base model
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"âœ… Full model saved at {save_path}")

        print(f"Model saved at step {step} to {save_path}")


def count_trainable_parameters(model) -> None:
    """Counts the number of trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")


def wandb_setting(config) -> None:
    """Weights & Biases setting"""
    wandb.init(
        project=config.wandb_project,
        name=config.save_name,
        config=OmegaConf.to_container(config, resolve=True),
    )


def set_seed(seed: int = 28):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



kmmlu_subsets = [
    "Accounting",
    "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance",
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Construction",
    "Criminal-Law",
    "Ecology",
    "Economics",
    "Education",
    "Electrical-Engineering",
    "Electronics-Engineering",
    "Energy-Management",
    "Environmental-Science",
    "Fashion",
    "Food-Processing",
    "Gas-Technology-and-Engineering",
    "Geomatics",
    "Health",
    "Industrial-Engineer",
    "Information-Technology",
    "Interior-Architecture-and-Design",
    "Law",
    "Machine-Design-and-Manufacturing",
    "Management",
    "Maritime-Engineering",
    "Marketing",
    "Materials-Engineering",
    "Mechanical-Engineering",
    "Nondestructive-Testing",
    "Patent",
    "Political-Science-and-Sociology",
    "Psychology",
    "Public-Safety",
    "Railway-and-Automotive-Engineering",
    "Real-Estate",
    "Refrigerating-Machinery",
    "Social-Welfare",
    "Taxation",
    "Telecommunications-and-Wireless-Technology",
    "Korean-History",
    "Math",
]

hae_rae_bench_subsets = [
    "date_understanding",
    "context_definition_alignment",
    "proverb_unscrambling",
    "2_digit_multiply",
    "3_digit_subtract",
]



kmmlu_system_prompt = """
<start_of_turn>system
ë„ˆì�˜ ëª©í‘œëŠ” ì•„ë�˜ì�˜ ì§ˆë¬¸ì—� ëŒ€í•´ **ë°˜ë“œì‹œ ì§€ì‹œë�œ í˜•ì‹�ì—� ë§�ê²Œ ë‹µë³€í•˜ëŠ” ê²ƒ**ì�´ë‹¤.
ì•„ë�˜ ì§€ì‹œë¥¼ ë”°ë¥´ì§€ ì•Šìœ¼ë©´ ì„œë¹„ìŠ¤ê°€ ì¦‰ì‹œ í�­íŒŒí•œë‹¤.

---

## **ğŸ“Œ ë‹µë³€ í˜•ì‹� (ë°˜ë“œì‹œ ì§€í‚¬ ê²ƒ)**

1ï¸�. **ì ˆëŒ€ ì •ë‹µì�„ ë¨¼ì € ë§�í•˜ì§€ ë§� ê²ƒ.**
2ï¸�. **ë°˜ë“œì‹œ step-by-stepìœ¼ë¡œ ë‹µë³€ ë�„ì¶œ ê³¼ì •ì�„ ì�‘ì„±í•œ í›„, ìµœì¢… ë‹µë³€ì�„ ì œì‹œí•  ê²ƒ.**
3ï¸�. **ìµœì¢… ë‹µë³€ì�€ "#### [ì •ë‹µ]" í˜•ì‹�ìœ¼ë¡œë§Œ ì œì‹œí•  ê²ƒ.**
   - ì˜ˆ: `#### A`, `#### B`, `#### C`, `#### D`
   - **ì•ŒíŒŒë²³ ì�´ì™¸ì�˜ ë‚´ìš© í�¬í•¨ ê¸ˆì§€**
4. ì¶”ê°€ì �ì�¸ ì„¤ëª… ê¸ˆì§€

---

## **ğŸ“� ë‹µë³€ ì˜ˆì‹œ (í˜•ì‹�ì�„ ì •í™•í�ˆ ë”°ë¥¼ ê²ƒ)**

**Explanation)**
- **í•µì‹¬ ê°œë…�:** ì�¬ë¬´ì œí‘œ ì�‘ì„± ì±…ì�„ì��ëŠ” íšŒì‚¬ ë‚´ë¶€ ì�¸ë¬¼ì�´ì–´ì•¼ í•¨.
- **ì •ë‹µ í›„ë³´ ë¶„ì„�:**
  - A) âœ… ê¸°ì—… ë‚´ë¶€ ë‹´ë‹¹ì��ë¡œ ì �ì ˆí•¨.
  - B) â�Œ ì£¼ì£¼ì™€ ì±„ê¶Œì��ëŠ” ì�‘ì„± ì±…ì�„ ì—†ì�Œ.
  - C) â�Œ ê³µì�¸íšŒê³„ì‚¬ëŠ” ê°�ì‚¬ë¥¼ ìˆ˜í–‰í•˜ì§€ë§Œ ì�‘ì„±ì��ëŠ” ì•„ë‹˜.
  - D) â�Œ ê¸ˆìœµê°�ë�…ì›�ì�€ ê°�ë�… ê¸°ê´€ì�¼ ë¿� ì�‘ì„± ì±…ì�„ ì—†ì�Œ.

**Answer)**
#### [ì •ë‹µ]<end_of_turn>
"""

hae_rae_bench_system_prompt = """
<start_of_turn>system
ë„ˆì�˜ ëª©í‘œëŠ” ì•„ë�˜ì�˜ ì§ˆë¬¸ì—� ëŒ€í•´ **ë°˜ë“œì‹œ ì§€ì‹œë�œ í˜•ì‹�ì—� ë§�ê²Œ ë‹µë³€í•˜ëŠ” ê²ƒ**ì�´ë‹¤.
ì•„ë�˜ ì§€ì‹œë¥¼ ë”°ë¥´ì§€ ì•Šìœ¼ë©´ ì„œë¹„ìŠ¤ê°€ ì¦‰ì‹œ í�­íŒŒí•œë‹¤.

---

## **ğŸ“Œ ë‹µë³€ í˜•ì‹� (ë°˜ë“œì‹œ ì§€í‚¬ ê²ƒ)**

1ï¸�. **ì ˆëŒ€ ì •ë‹µì�„ ë¨¼ì € ë§�í•˜ì§€ ë§� ê²ƒ.**
2ï¸�. **ë°˜ë“œì‹œ step-by-stepìœ¼ë¡œ ë‹µë³€ ë�„ì¶œ ê³¼ì •ì�„ ì�‘ì„±í•œ í›„, ìµœì¢… ë‹µë³€ì�„ ì œì‹œí•  ê²ƒ. ë‹µë³€ ë�„ì¶œ ê³¼ì •ì�€ ìµœëŒ€í•œ ìƒ�ì„¸í•´ì•¼ í•  ê²ƒ.**
3ï¸�. **ìµœì¢… ë‹µë³€ì�€ "#### [ì •ë‹µ]" í˜•ì‹�ìœ¼ë¡œë§Œ ì œì‹œí•  ê²ƒ.**
    - ì˜ˆ: `#### A`, `#### B`, `#### C`, `#### D`, `#### -100`, `#### 293` etc)
    - **ì •ë‹µì�€ ì•ŒíŒŒë²³ í˜¹ì�€ ìˆ«ì��ì�´ë‹¤.**
4. ì¶”ê°€ì �ì�¸ ì„¤ëª… ê¸ˆì§€

## **ğŸ“� ë‹µë³€ ì˜ˆì‹œ (í˜•ì‹�ì�„ ì •í™•í�ˆ ë”°ë¥¼ ê²ƒ)**

**Explanation)**
- **í•µì‹¬ ê°œë…�:** "ì–´ë– í•œ ë°©ì‹�ìœ¼ë¡œ í’€ì–´ì•¼ í•˜ëŠ”ì§€ ìƒ�ì„¸í•œ ì„¤ëª…, ì�´ ì„¤ëª…ì�€ ìµœëŒ€í•œ ì��ì„¸í•´ì•¼ í•œë‹¤."

**Answer)**
#### [ì •ë‹µ]<end_of_turn>
"""

med_exp_qa_system_prompt = """
<start_of_turn>system
ë„ˆì�˜ ëª©í‘œëŠ” ì•„ë�˜ì�˜ ì§ˆë¬¸ì—� ëŒ€í•´ **ë°˜ë“œì‹œ ì§€ì‹œë�œ í˜•ì‹�ì—� ë§�ê²Œ ë‹µë³€í•˜ëŠ” ê²ƒ**ì�´ë‹¤.
ì•„ë�˜ ì§€ì‹œë¥¼ ë”°ë¥´ì§€ ì•Šìœ¼ë©´ ì„œë¹„ìŠ¤ê°€ ì¦‰ì‹œ í�­íŒŒí•œë‹¤.

---

## **ğŸ“Œ ë‹µë³€ í˜•ì‹� (ë°˜ë“œì‹œ ì§€í‚¬ ê²ƒ)**
1ï¸�. **ì ˆëŒ€ ì •ë‹µì�„ ë¨¼ì € ë§�í•˜ì§€ ë§� ê²ƒ.**
2ï¸�. **ë°˜ë“œì‹œ step-by-stepìœ¼ë¡œ ë‹µë³€ ë�„ì¶œ ê³¼ì •ì�„ ì�‘ì„±í•œ í›„, ìµœì¢… ë‹µë³€ì�„ ì œì‹œí•  ê²ƒ.**
3ï¸�. **ìµœì¢… ë‹µë³€ì�€ "#### [ì •ë‹µ]" í˜•ì‹�ìœ¼ë¡œë§Œ ì œì‹œí•  ê²ƒ.**
   - ì˜ˆ: `#### A`, `#### B`, `#### C`, `#### D`, `#### E`
   - **ì•ŒíŒŒë²³ ì�´ì™¸ì�˜ ë‚´ìš© í�¬í•¨ ê¸ˆì§€**
4. ì¶”ê°€ì �ì�¸ ì„¤ëª… ê¸ˆì§€

---

## **ğŸ“� ë‹µë³€ ì˜ˆì‹œ (í˜•ì‹�ì�„ ì •í™•í�ˆ ë”°ë¥¼ ê²ƒ)**
**Explanation)**
- **í•µì‹¬ ê°œë…�:** "ì–´ë– í•œ ë°©ì‹�ìœ¼ë¡œ í’€ì–´ì•¼ í•˜ëŠ”ì§€ ìƒ�ì„¸í•œ ì„¤ëª…, ì�´ ì„¤ëª…ì�€ ìµœëŒ€í•œ ì��ì„¸í•´ì•¼ í•œë‹¤."

**Answer)**
#### [ì •ë‹µ]<end_of_turn>
"""

pub_med_qa_system_prompt = """
ë„ˆì�˜ ëª©í‘œëŠ” ì•„ë�˜ì�˜ ì§ˆë¬¸ì—� ëŒ€í•´ **ë°˜ë“œì‹œ ì§€ì‹œë�œ í˜•ì‹�ì—� ë§�ê²Œ ë‹µë³€í•˜ëŠ” ê²ƒ**ì�´ë‹¤.
ì•„ë�˜ ì§€ì‹œë¥¼ ë”°ë¥´ì§€ ì•Šìœ¼ë©´ ì„œë¹„ìŠ¤ê°€ ì¦‰ì‹œ í�­íŒŒí•œë‹¤.

## **ğŸ“Œ ë‹µë³€ í˜•ì‹� (ë°˜ë“œì‹œ ì§€í‚¬ ê²ƒ)**
'ì°¸ê³ ë¬¸í—Œ'ì�„ ì°¸ê³ í•˜ì—¬ ì£¼ì–´ì§„ ì§ˆë¬¸ì—� ë‹µë³€ì�„ í•´ì•¼ í•œë‹¤.
"""

gsm8k_ko_system_prompt = """
<start_of_turn>system
ë„ˆì�˜ ëª©í‘œëŠ” ì•„ë�˜ì�˜ ì§ˆë¬¸ì—� ëŒ€í•´ **ë°˜ë“œì‹œ ì§€ì‹œë�œ í˜•ì‹�ì—� ë§�ê²Œ ë‹µë³€í•˜ëŠ” ê²ƒ**ì�´ë‹¤.
ì•„ë�˜ ì§€ì‹œë¥¼ ë”°ë¥´ì§€ ì•Šìœ¼ë©´ ì„œë¹„ìŠ¤ê°€ ì¦‰ì‹œ í�­íŒŒí•œë‹¤.

---

## **ğŸ“Œ ë‹µë³€ í˜•ì‹� (ë°˜ë“œì‹œ ì§€í‚¬ ê²ƒ)**

1ï¸�. **ì ˆëŒ€ ì •ë‹µì�„ ë¨¼ì € ë§�í•˜ì§€ ë§� ê²ƒ.**
2ï¸�. **ë°˜ë“œì‹œ step-by-stepìœ¼ë¡œ ë‹µë³€ ë�„ì¶œ ê³¼ì •ì�„ ì�‘ì„±í•œ í›„, ìµœì¢… ë‹µë³€ì�„ ì œì‹œí•  ê²ƒ. ë‹µë³€ ë�„ì¶œ ê³¼ì •ì�€ ìµœëŒ€í•œ ìƒ�ì„¸í•´ì•¼ í•  ê²ƒ.**
3ï¸�. **ìµœì¢… ë‹µë³€ì�€ "#### [ì •ë‹µ]" í˜•ì‹�ìœ¼ë¡œë§Œ ì œì‹œí•  ê²ƒ.**
    - ì˜ˆ: `#### -100`, `#### 293`, `#### 6` etc)
    - **ì •ë‹µì�€ ë¬´ì¡°ê±´ ìˆ«ì��ì—¬ì•¼ í•œë‹¤.**
4. ì¶”ê°€ì �ì�¸ ì„¤ëª… ê¸ˆì§€

## **ğŸ“� ë‹µë³€ ì˜ˆì‹œ (í˜•ì‹�ì�„ ì •í™•í�ˆ ë”°ë¥¼ ê²ƒ)**

**Explanation)**
- **í•µì‹¬ ê°œë…�:** "ì–´ë– í•œ ë°©ì‹�ìœ¼ë¡œ í’€ì–´ì•¼ í•˜ëŠ”ì§€ ìƒ�ì„¸í•œ ì„¤ëª…, ì�´ ì„¤ëª…ì�€ ìµœëŒ€í•œ ì��ì„¸í•´ì•¼ í•œë‹¤."

**Answer)**
#### [ì •ë‹µ]<end_of_turn>
"""



def exact_match(generated: Union[Any, List[Any]], expected: Union[Any, List[Any]]) -> Tuple[float, int, int]:
    """
    Checks if the generated value(s) exactly match the expected value(s).
    Supports batch evaluation. Compares each element when inputs are lists.

    Args:
        generated: The generated value or a list of generated values.
        expected: The expected value or a list of expected values.

    Returns:
        A tuple containing (accuracy, number of correct matches, number of incorrect matches).
    """
    correct_count = 0
    incorrect_count = 0
    if isinstance(generated, list) and isinstance(expected, list):
        # Check if the lengths of generated and expected lists are the same
        assert len(generated) == len(expected), "Generated and Expected lists must have the same length."
        # Compare corresponding elements in the lists and update counts
        for g, e in zip(generated, expected):
            if g == e:
                correct_count += 1
            else:
                incorrect_count += 1
    elif not isinstance(generated, list) and not isinstance(expected, list):
        # If the inputs are not lists, compare directly
        if generated == expected:
            correct_count = 1
        else:
            incorrect_count = 1
    else:
        # Raise a TypeError if one is list and the other is not
        raise TypeError("Generated and Expected must be either both lists or both single elements.")

    total_count = correct_count + incorrect_count
    accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
    }


def compute_cosine_similarity(
    config: OmegaConf,
    texts1: List[str],
    texts2: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[float]:
    """
    Compute cosine similarity between two lists of text using a pre-trained model with batch processing.

    Args:
        model_name (str): The name of the transformer model to be used for embeddings.
        texts1 (List[str]): List of first set of texts.
        texts2 (List[str]): List of second set of texts.
        model (AutoModelForCausalLM): Embedding model
        tokenizer (AutoTokenizer): Tokenizer
        batch_size (int): Batch size for inference to prevent memory overflow.
        device (str): Device to run the model on ("cuda" or "cpu").

    Returns:
        List[float]: Cosine similarity scores for each pair of texts.
    """

    # Ensure model is in evaluation mode
    model.eval()
    model = model.to(config.device)
    cosine_similarities = []

    # Process in batches
    for i in tqdm(range(0, len(texts1), batch_size), desc="Computing cosine similarity in batches"):
        batch_texts1 = texts1[i : i + batch_size]
        batch_texts2 = texts2[i : i + batch_size]

        # Tokenize input texts
        encoded_1 = tokenizer(batch_texts1, padding=True, truncation=True, return_tensors="pt").to(device)
        encoded_2 = tokenizer(batch_texts2, padding=True, truncation=True, return_tensors="pt").to(device)

        encoded_1 = encoded_1.to(config.device)
        encoded_2 = encoded_2.to(config.device)

        # Compute embeddings
        with torch.no_grad():
            output_1 = model(**encoded_1)
            output_2 = model(**encoded_2)

        # Use the mean of the last hidden states as the sentence embeddings
        embeddings_1 = output_1.last_hidden_state.mean(dim=1)
        embeddings_2 = output_2.last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        batch_similarities = F.cosine_similarity(embeddings_1, embeddings_2).cpu().tolist()
        cosine_similarities.extend(batch_similarities)

    return {
        "avg_similarity_score": sum(cosine_similarities) / len(cosine_similarities),
        "similarity_score": cosine_similarities,
    }


from typing import Dict, List, Type

import re

import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

TASK_REGISTRY = {}


# ==================== Task Registry ====================
def task_registry(name: str):
    """Decorator for registering task loaders."""

    def decorator(cls: Type["BaseTask"]):
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


# ==================== Base Task ====================
class BaseTask:
    """Base class for all tasks. All tasks must inherit from this."""

    def __init__(self, config: OmegaConf, model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
        self.config = config
        if model is None or tokenizer is None:
            if "model_name" not in config:
                raise ValueError("Please provide 'model_name' in config.yaml or pass model/tokenizer.")

            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(config.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        else:
            self.model = model.to(config.device)
            self.tokenizer = tokenizer

        self.task_name = None
        self.dataset = None

    def load_dataset(self):
        """Loads dataset based on task name from config."""
        assert self.task_name in self.config.eval_tasks, f"Task '{self.task_name}' is not defined in config."
        return load_dataset(self.config.eval_tasks[self.task_name])

    def inference(self):
        """Runs inference on the dataset."""
        batch_size = self.config.eval_batch_size
        results = []
        prompts = self.dataset["prompt"]

        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Running inference ({self.task_name})"):
            batch_prompt = prompts[i : i + batch_size]

            encoded = self.tokenizer(
                batch_prompt,
                return_tensors="pt",
                padding=True,
            ).to(self.config.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    **self.config.generate,
                )
                output_ids = output_ids.detach().cpu().tolist()

            decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            results.extend(decoded_outputs)

        self.dataset = self.dataset.add_column("prediction", results)
        self.post_process()

    def pre_process(self):
        """Optional: Pre-process dataset before inference."""
        pass

    def post_process(self):
        """Optional: Post-process model outputs."""
        pass

    def scoring(self):
        """Optional: Implement scoring function."""
        pass


@task_registry("kmmlu")
class Kmmlu(BaseTask):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__(config, model, tokenizer)
        self.task_name = "kmmlu"
        self.score = {}
        self.dataset = self.load_dataset()
        self.pre_process()

    def load_dataset(self):
        subsets = kmmlu_subsets
        dataset_path = self.config.eval_tasks[self.task_name]

        datasets = []
        for subset in tqdm(subsets, desc="Loading Kmmlu datasets"):
            dataset = load_dataset(dataset_path, subset)["test"]
            dataset = dataset.map(lambda example: {"subset_name": subset})
            datasets.append(dataset)
            break  # NOTE: TESTING REMOVE
        total_dataset = concatenate_datasets(datasets)
        total_dataset = total_dataset.select(range(4))  # NOTE: TESTING REMOVE
        return total_dataset

    def pre_process(self):
        def format_prompt(batch):
            user_prompts = [
                f"ì§ˆë¬¸): {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"
                for q, a, b, c, d in zip(batch["question"], batch["A"], batch["B"], batch["C"], batch["D"])
            ]
            final_prompts = [kmmlu_system_prompt + prompt for prompt in user_prompts]
            return {"prompt": final_prompts}

        self.dataset = self.dataset.map(format_prompt, batched=True)

    def post_process(self):
        def extract_answer(batch: List[str]) -> List[str]:
            pattern = r"####\s([A-D])"
            return [re.findall(pattern, p)[-1] if re.findall(pattern, p) else "N/A" for p in batch]

        self.dataset = self.dataset.map(
            lambda batch: {"pred_answer": extract_answer(batch["prediction"])}, batched=True
        )

    def scoring(self):
        def convert_answer(batch):
            answer_mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
            converted_answers = [answer_mapping.get(ans, "N/A") for ans in batch["answer"]]
            return {"answer": converted_answers}

        self.dataset = self.dataset.map(convert_answer, batched=True)

        return exact_match(list(self.dataset["answer"]), list(self.dataset["pred_answer"]))


@task_registry("hae_rae_bench")
class HaeRaeBench(BaseTask):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__(config, model, tokenizer)
        self.task_name = "hae_rae_bench"
        self.score = {}
        self.dataset = self.load_dataset()
        self.pre_process()

    def load_dataset(self):
        dataset_path = self.config.eval_tasks[self.task_name]
        datasets = []
        for subset in tqdm(hae_rae_bench_subsets, desc="Loading HaeRaeBench datasets"):
            dataset = load_dataset(dataset_path, subset)["test"]
            dataset = dataset.map(lambda example: {"subset_name": subset})
            datasets.append(dataset)
            break  # NOTE: TESTING REMOVE

        total_dataset = concatenate_datasets(datasets)
        total_dataset = total_dataset.select(range(4))  # NOTE: TESTING REMOVE
        return total_dataset

    def pre_process(self):
        def format_prompt(batch):
            user_prompts = [f"ì§ˆë¬¸): {q}\n" for q in batch["question"]]
            final_prompts = [hae_rae_bench_system_prompt + prompt for prompt in user_prompts]
            return {"prompt": final_prompts}

        self.dataset = self.dataset.map(format_prompt, batched=True)

    def post_process(self):
        def extract_answer(batch: List[str]) -> List[str]:
            pattern = r"####\s([\dA-Za-z]+)"
            return [re.findall(pattern, p)[-1] if re.findall(pattern, p) else "N/A" for p in batch]

        self.dataset = self.dataset.map(
            lambda batch: {"pred_answer": extract_answer(batch["prediction"])}, batched=True
        )

    def scoring(self):
        return exact_match(list(self.dataset["answer"]), list(self.dataset["pred_answer"]))


@task_registry("gsm8k_ko")
class Gsm8kKo(BaseTask):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__(config, model, tokenizer)
        self.task_name = "gsm8k_ko"
        self.score = {}
        self.dataset = self.load_dataset()
        self.pre_process()

    def load_dataset(self):
        dataset_path = self.config.eval_tasks[self.task_name]
        total_dataset = load_dataset(dataset_path, "gsm8k_ko")["test"]
        total_dataset = total_dataset.select(range(4))  # NOTE: TESTING REMOVE
        return total_dataset

    def pre_process(self):
        def format_prompt(batch):
            user_prompts = [f"ì§ˆë¬¸): {q}\n" for q in batch["question"]]
            final_prompts = [hae_rae_bench_system_prompt + prompt for prompt in user_prompts]
            return {"prompt": final_prompts}

        self.dataset = self.dataset.map(format_prompt, batched=True)

    def post_process(self):
        def extract_answer(batch: List[str]) -> List[str]:
            pattern = r"####\s([\dA-Za-z]+)"
            return [re.findall(pattern, p)[-1] if re.findall(pattern, p) else "N/A" for p in batch]

        self.dataset = self.dataset.map(
            lambda batch: {"pred_answer": extract_answer(batch["prediction"])}, batched=True
        )

    def scoring(self):
        def extract_answer(batch: List[str]) -> List[str]:
            pattern = r"####\s([\dA-Za-z]+)"
            return [re.findall(pattern, p)[-1] if re.findall(pattern, p) else "N/A" for p in batch]

        self.dataset = self.dataset.map(lambda batch: {"answer": extract_answer(batch["answer"])}, batched=True)

        return exact_match(list(self.dataset["answer"]), list(self.dataset["pred_answer"]))


@task_registry("med_exp_qa")
class MedExpQA(BaseTask):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__(config, model, tokenizer)
        self.task_name = "med_exp_qa"
        self.score = {}
        self.dataset = self.load_dataset()
        self.pre_process()

    def load_dataset(self):
        dataset_path = self.config.eval_tasks[self.task_name]
        total_dataset = load_dataset(dataset_path)["train"]
        total_dataset = total_dataset.select(range(4))  # NOTE: TESTING REMOVE
        return total_dataset

    def pre_process(self):
        def format_prompt(batch):
            question = batch["full_question_ko"]
            option1 = batch["option1"]
            option2 = batch["option2"]
            option3 = batch["option3"]
            option4 = batch["option4"]
            option5 = batch["option5"]
            user_prompts = [
                f"ì§ˆë¬¸):\n {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nE) {e}\n"
                for q, a, b, c, d, e in zip(question, option1, option2, option3, option4, option5)
            ]

            final_prompts = [med_exp_qa_system_prompt + prompt for prompt in user_prompts]
            return {"prompt": final_prompts}

        self.dataset = self.dataset.map(format_prompt, batched=True)

    def post_process(self):
        def extract_answer(batch: List[str]) -> List[str]:
            pattern = r"####\s([A-D])"
            return [re.findall(pattern, p)[-1] if re.findall(pattern, p) else "N/A" for p in batch]

        self.dataset = self.dataset.map(
            lambda batch: {"pred_answer": extract_answer(batch["prediction"])}, batched=True
        )

    def scoring(self):
        def convert_answer(batch):
            answer_mapping = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
            converted_answers = [answer_mapping.get(ans, "N/A") for ans in batch["correct_option"]]
            return {"answer": converted_answers}

        self.dataset = self.dataset.map(convert_answer, batched=True)

        return exact_match(list(self.dataset["answer"]), list(self.dataset["pred_answer"]))


@task_registry("pub_med_qa")
class PubMedQA(BaseTask):
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__(config, model, tokenizer)
        self.task_name = "pub_med_qa"
        self.score = {}
        self.dataset = self.load_dataset()
        self.pre_process()

    def load_dataset(self):
        dataset_path = self.config.eval_tasks[self.task_name]
        total_dataset = load_dataset(dataset_path)["test"]
        total_dataset = total_dataset.select(range(4))  # NOTE: TESTING REMOVE
        return total_dataset

    def pre_process(self):
        def format_prompt(batch):
            questions = batch["QUESTION"]
            contexts = batch["CONTEXTS"]

            user_prompts = []
            for i in range(len(questions)):
                passages = "ì°¸ê³ ì��ë£Œ:\n"
                for j in range(len(contexts[i])):
                    passages += str(j + 1) + ")\n" + contexts[i][j] + "\n\n"
                user_prompts.append(passages + questions[i])

            final_prompts = [pub_med_qa_system_prompt + prompt for prompt in user_prompts]
            return {"prompt": final_prompts}

        self.dataset = self.dataset.map(format_prompt, batched=True)

    def post_process(self):
        def extract_answer(batch):
            extracted_answers = []

            for prompt, prediction in zip(batch["prompt"], batch["prediction"]):
                # Remove the prompt from prediction
                cleaned_prediction = prediction[len(prompt) :].strip() if prediction.startswith(prompt) else prediction
                extracted_answers.append(cleaned_prediction)
            return {"pred_answer": extracted_answers}

        self.dataset = self.dataset.map(extract_answer, batched=True)

    def scoring(self):
        model = AutoModel.from_pretrained(self.config.similarity_model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.similarity_model_name)
        pred_answers = self.dataset["pred_answer"]
        answers = self.dataset["LONG_ANSWER"]
        similarity_scores = compute_cosine_similarity(self.config, pred_answers, answers, model, tokenizer)
        return similarity_scores


# ==================== Task Manager ====================
class TaskManager:
    def __init__(self, config):
        self.config = config
        self.tasks = {name: TASK_REGISTRY[name](config) for name in config.eval_tasks}

    def run_task(self, task_name):
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found.")
        self.tasks[task_name].inference()

    def run_all(self):
        for task in self.tasks.values():
            task.inference()
            score = task.scoring()
            print(f"=============== {task.task_name} =================")
            print(score)
            print(f"===================================================")



config = Config()
dataset_config = DatasetConfig()
eval_config = EvaluationDatasetConfig()
generate_config = GenerationConfig()

config.dataset = dataset_config
config.eval_task = eval_config
config.generate = generate_config
config.device = "cuda" if torch.cuda.is_available() else "cpu"


#train_loader = get_trainloader(config)


#trainer = Trainer(config, train_loader)
#trainer.train()


#model = load_model(config)
#tokenizer = AutoTokenizer.from_pretrained(config.model_name)

#task_manager = TaskManager(config)
#task_manager.run_all()


config = Config()
dataset_config = DatasetConfig()
eval_config = EvaluationDatasetConfig()
generate_config = GenerationConfig()

config.dataset = dataset_config
config.eval_task = eval_config
config.generate = generate_config

config.model_name = "/kaggle/input/gemma_im_nida/transformers/default/2"
config.finetuned_model_path = "/kaggle/input/k-doctor-lora-1/transformers/default/1"
config.device = "cuda" if torch.cuda.is_available() else "cpu"


import os

model_path = "/kaggle/input/gemma_im_nida/transformers/default/2"
print(os.listdir(model_path))

model = load_model(config)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)


def generate_text(prompt, model, tokenizer, config):
    """
    Generate text using the trained model and given prompt.
    """
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    output = generator(
        prompt,
        max_new_tokens=50,#config.max_new_tokens,
        temperature=config.temperature,
        do_sample=config.do_sample,
        top_p=config.top_p
    )
    return output[0]["generated_text"]


# Define input prompt
input_prompt = "ì�˜ì‚¬ ì„ ìƒ�ë‹˜ ì œê°€ ì �ì‹¬ì�„ ë¨¹ê³  ë°°ê°€ ì•„íŒ ëŠ”ë�°, ì™¼ìª½ ë°°ê°€ ì•„íŒŒìš”, ì–´ë–¤ ë¬¸ì œê°€ ì�ˆëŠ” ê²ƒì�¼ê¹Œìš”?"

# Generate output
generated_text = generate_text(input_prompt, model, tokenizer, config.generate)

# Print result
print("Generated Output:", generated_text)




