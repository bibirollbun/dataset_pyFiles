!git clone https://github.com/rbiswasfc/eedi-mining-misconceptions.git
%cd eedi-mining-misconceptions
!pip install -r requirements.txt
# !pip install "flash_attn==2.6.3" --no-build-isolation


!python download_datasets.py


!pip install huggingface_hub[hf_transfer]
# !HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-Math-7B
# !HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-14B
# !HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-32B
# !HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-72B
# !HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download intfloat/e5-mistral-7b-instruct
!HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download BAAI/bge-en-icl


# %%writefile /kaggle/working/eedi-mining-misconceptions/code/llm_embedding/eedi_model.py
# from dataclasses import dataclass
# from typing import Optional

# import torch
# import torch.distributed as dist
# import torch.nn.functional as F
# from peft import LoraConfig, TaskType, get_peft_model
# from torch import Tensor, nn
# from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
# from transformers.file_utils import ModelOutput


# @dataclass
# class EmbedderOutput(ModelOutput):
#     q_reps: Optional[Tensor] = None
#     p_reps: Optional[Tensor] = None
#     loss: Optional[Tensor] = None
#     scores: Optional[Tensor] = None
#     nce_loss: Optional[Tensor] = None
#     distill_loss: Optional[Tensor] = None


# def get_base_model(cfg):
#     config = AutoConfig.from_pretrained(cfg.model.backbone_path, trust_remote_code=cfg.model.trust_remote_code)
#     config.use_cache = False

#     if cfg.model.use_bnb:
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,  # True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.float16,
#         )
#         model = AutoModel.from_pretrained(
#             cfg.model.backbone_path,
#             config=config,
#             quantization_config=bnb_config,
#             attn_implementation=cfg.model.attn_implementation,
#             trust_remote_code=cfg.model.trust_remote_code,
#         )
#     else:
#         model = AutoModel.from_pretrained(
#             cfg.model.backbone_path,
#             config=config,
#             attn_implementation=cfg.model.attn_implementation,
#             trust_remote_code=cfg.model.trust_remote_code,
#             torch_dtype=torch.float16,
#             device_map='balanced'
#         )
#     model.config.pretraining_tp = 1

#     # LoRA ---
#     if cfg.model.use_lora:
#         peft_config = LoraConfig(
#             r=cfg.model.lora.r,
#             lora_alpha=cfg.model.lora.lora_alpha,
#             lora_dropout=cfg.model.lora.lora_dropout,
#             bias="none",
#             task_type=TaskType.FEATURE_EXTRACTION,  # CAUSAL_LM, FEATURE_EXTRACTION
#             inference_mode=False,
#             target_modules=list(cfg.model.lora.target_modules),
#             modules_to_save=list(cfg.model.lora.modules_to_save),
#         )

#         model = get_peft_model(model, peft_config)
#         model.print_trainable_parameters()

#     return model


# class BiEncoderModel(nn.Module):
#     def __init__(self, cfg, base_model, accelerator):
#         super().__init__()

#         self.model = base_model
#         self.config = self.model.config
#         self.use_distillation = cfg.model.use_distillation

#         self.sub_batch_size = cfg.train_params.sub_batch_size

#         self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
#         self.sentence_pooling_method = cfg.model.sentence_pooling_method

#         self.accelerator = accelerator
#         self.negatives_cross_device = cfg.model.negatives_cross_device  # accelerator.use_distributed
#         if self.negatives_cross_device:
#             assert accelerator.use_distributed, "Distributed training is required for negatives_cross_device"

#         self.world_size = accelerator.num_processes
#         self.process_rank = accelerator.process_index

#         accelerator.print(f"negatives_cross_device: {self.negatives_cross_device}")
#         accelerator.print(f"world_size: {self.world_size}")
#         accelerator.print(f"process_rank: {self.process_rank}")

#     def gradient_checkpointing_enable(self, **kwargs):
#         self.model.gradient_checkpointing_enable(**kwargs)

#     def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#         left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

#     def sentence_embedding(self, hidden_state, mask):
#         if self.sentence_pooling_method == "mean":
#             s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
#             d = mask.sum(axis=1, keepdim=True).float()
#             return s / d
#         elif self.sentence_pooling_method == "cls":
#             return hidden_state[:, 0]
#         elif self.sentence_pooling_method == "last":
#             return self.last_token_pool(hidden_state, mask)

#     def encode(self, features):
#         if self.sub_batch_size is not None and self.sub_batch_size > 0:
#             all_p_reps = []
#             for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
#                 end_inx = min(i + self.sub_batch_size, len(features["attention_mask"]))
#                 sub_features = {k: v[i:end_inx] for k, v in features.items()}
#                 last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
#                 p_reps = self.sentence_embedding(last_hidden_state, sub_features["attention_mask"])
#                 all_p_reps.append(p_reps)
#             all_p_reps = torch.cat(all_p_reps, 0).contiguous()
#         else:
#             last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
#             all_p_reps = self.sentence_embedding(last_hidden_state, features["attention_mask"])

#         return all_p_reps.contiguous()

#     def compute_similarity(self, q_reps, p_reps, temperature=0.02):
#         x = q_reps.unsqueeze(1)
#         y = p_reps.unsqueeze(0)

#         sim = nn.CosineSimilarity(dim=-1)(x, y) / temperature
#         return sim

#     def forward(self, queries, contents, teacher_scores, temperature=0.02):
#         q_reps = self.encode(queries)
#         p_reps = self.encode(contents)
#         # self.accelerator.print(f"before dist_gather: q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")

#         if self.negatives_cross_device:
#             q_reps = self._dist_gather_tensor(q_reps)
#             p_reps = self._dist_gather_tensor(p_reps)
#             # self.accelerator.print(f"after dist_gather: q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")

#         group_size = p_reps.size(0) // q_reps.size(0)  # group_size = 1
#         scores = self.compute_similarity(q_reps, p_reps, temperature)
#         scores = scores.view(q_reps.size(0), -1)  # (n, m)

#         target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
#         target = target * group_size
#         nce_loss = self.compute_loss(scores, target)

#         # compute distillation loss
#         if self.use_distillation:
#             distill_loss = self.compute_distillation_loss(scores, teacher_scores)
#             loss = 0.5 * nce_loss + 0.5 * distill_loss
#         else:
#             distill_loss = torch.tensor(0.0, device=scores.device)
#             loss = nce_loss

#         # print(f"loss: {loss}, nce_loss: {nce_loss}, distill_loss: {distill_loss}")

#         return EmbedderOutput(loss=loss, scores=scores, nce_loss=nce_loss, distill_loss=distill_loss)

#     def compute_loss(self, scores, target):
#         return self.cross_entropy(scores, target)

#     def compute_distillation_loss(self, scores, teacher_scores, teacher_temperature=1.25):
#         # print(f"scores: {scores.shape}, teacher_scores: {teacher_scores.shape}")
#         # print(f"scores: {scores}")
#         # print(f"teacher_scores: {teacher_scores}")
#         teacher_targets = F.softmax(teacher_scores.detach() / teacher_temperature, dim=-1)  # (n, m)
#         # print(f"teacher_targets: {teacher_targets}")
#         loss = -torch.mean(torch.sum(torch.log_softmax(scores, dim=-1) * teacher_targets, dim=-1))
#         return loss

#     def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
#         if t is None:
#             return None
#         t = t.contiguous()

#         all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
#         dist.all_gather(all_tensors, t)

#         all_tensors[self.process_rank] = t
#         all_tensors = torch.cat(all_tensors, dim=0)

#         return all_tensors

#     def save(self, output_dir: str):
#         state_dict = self.model.state_dict()
#         state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
#         self.model.save_pretrained(output_dir, state_dict=state_dict)


%%writefile /kaggle/working/eedi-mining-misconceptions/code/llm_embedding/eedi_model.py
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.file_utils import ModelOutput


@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    nce_loss: Optional[Tensor] = None
    distill_loss: Optional[Tensor] = None


def get_base_model(cfg):
    config = AutoConfig.from_pretrained(cfg.model.backbone_path, trust_remote_code=cfg.model.trust_remote_code)
    config.use_cache = False

    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModel.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    else:
        model = AutoModel.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch.float16,
        )
    model.config.pretraining_tp = 1

    # LoRA ---
    if cfg.model.use_lora:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # CAUSAL_LM, FEATURE_EXTRACTION
            inference_mode=False,
            target_modules=list(cfg.model.lora.target_modules),
            modules_to_save=list(cfg.model.lora.modules_to_save),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model


class BiEncoderModel(nn.Module):
    def __init__(self, cfg, base_model, accelerator):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.use_distillation = cfg.model.use_distillation

        self.sub_batch_size = cfg.train_params.sub_batch_size

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.sentence_pooling_method = cfg.model.sentence_pooling_method

        self.accelerator = accelerator
        self.negatives_cross_device = cfg.model.negatives_cross_device  # accelerator.use_distributed
        if self.negatives_cross_device:
            assert accelerator.use_distributed, "Distributed training is required for negatives_cross_device"

        self.world_size = accelerator.num_processes
        self.process_rank = accelerator.process_index

        accelerator.print(f"negatives_cross_device: {self.negatives_cross_device}")
        accelerator.print(f"world_size: {self.world_size}")
        accelerator.print(f"process_rank: {self.process_rank}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "last":
            return self.last_token_pool(hidden_state, mask)

    def encode(self, features):
        if self.sub_batch_size is not None and self.sub_batch_size > 0:
            all_p_reps = []
            for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                end_inx = min(i + self.sub_batch_size, len(features["attention_mask"]))
                sub_features = {k: v[i:end_inx] for k, v in features.items()}
                last_hidden_state = self.model(**sub_features, return_dict=True).last_hidden_state
                p_reps = self.sentence_embedding(last_hidden_state, sub_features["attention_mask"])
                all_p_reps.append(p_reps)
            all_p_reps = torch.cat(all_p_reps, 0).contiguous()
        else:
            last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
            all_p_reps = self.sentence_embedding(last_hidden_state, features["attention_mask"])

        return all_p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps, temperature=0.02):
        x = q_reps.unsqueeze(1)
        y = p_reps.unsqueeze(0)

        sim = nn.CosineSimilarity(dim=-1)(x, y) / temperature
        return sim

    def forward(self, queries, contents, teacher_scores, temperature=0.02):
        q_reps = self.encode(queries)
        p_reps = self.encode(contents)
        # self.accelerator.print(f"before dist_gather: q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")

        if self.negatives_cross_device:
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
            # self.accelerator.print(f"after dist_gather: q_reps: {q_reps.shape}, p_reps: {p_reps.shape}")

        group_size = p_reps.size(0) // q_reps.size(0)  # group_size = 1
        scores = self.compute_similarity(q_reps, p_reps, temperature)
        scores = scores.view(q_reps.size(0), -1)  # (n, m)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * group_size
        nce_loss = self.compute_loss(scores, target)

        # compute distillation loss
        if self.use_distillation:
            distill_loss = self.compute_distillation_loss(scores, teacher_scores)
            loss = 0.5 * nce_loss + 0.5 * distill_loss
        else:
            distill_loss = torch.tensor(0.0, device=scores.device)
            loss = nce_loss

        # print(f"loss: {loss}, nce_loss: {nce_loss}, distill_loss: {distill_loss}")

        return EmbedderOutput(loss=loss, scores=scores, nce_loss=nce_loss, distill_loss=distill_loss)

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def compute_distillation_loss(self, scores, teacher_scores, teacher_temperature=1.25):
        # print(f"scores: {scores.shape}, teacher_scores: {teacher_scores.shape}")
        # print(f"scores: {scores}")
        # print(f"teacher_scores: {teacher_scores}")
        teacher_targets = F.softmax(teacher_scores.detach() / teacher_temperature, dim=-1)  # (n, m)
        # print(f"teacher_targets: {teacher_targets}")
        loss = -torch.mean(torch.sum(torch.log_softmax(scores, dim=-1) * teacher_targets, dim=-1))
        return loss

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)


%%writefile /kaggle/working/eedi-mining-misconceptions/conf/llm_embedding/conf_bge.yaml
debug: false
seed: 234

fold: 0
enable_cuda_optimizations: true
full_fit: false
local_rank: # will be populated by training script

use_wandb: false

dataset:
  comp_dataset:  eedi-mining-misconceptions-in-mathematics
  input_dataset: conjuring92/eedi-embed-mix-silver-v3
  fold_dataset: conjuring92/eedi-five-folds

model:
  backbone_path: BAAI/bge-en-icl
  trust_remote_code: false
  max_length: 256
  sentence_pooling_method: last
  gradient_checkpointing: true
  compile: true
  attn_implementation: sdpa #eager
  negatives_cross_device: false
  padding_side: left
  add_eos_token: true

  use_bnb: true
  use_lora: true

  lora:
    r: 64
    lora_alpha: 128
    lora_dropout: 0.05

    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

    modules_to_save: []

  max_temperature: 0.01
  min_temperature: 0.01

  n_neighbour: 256
  use_distillation: true


train_params:
  retriever_bs: 64
  sub_batch_size: 8
  query_bs: 64
  content_bs: 64

  load_hard_negatives: true
  hard_negative_dataset: conjuring92/eedi-embed-mix-silver-v3
  hard_negative_file: hn_mapping.json
  teacher_logits_file: teacher_mapping.json

  num_hard_negatives: 15
  negative_depth_end: 24

  iterative_hard_negatives: false
  iterative_hard_negatives_trigger: 0
  negative_depth_start: 0

  warmup_pct: 0.1
  num_epochs: 3
  gradient_accumulation_steps: 1
  patience: 20
  eval_at_start: false

  batch_sampling:
    - random

  batch_sampling_weights:
    - 1.0

optimizer:
  name: AdamW8bit

  lr: 1e-5
  lr_lora_a: 1e-5
  lr_lora_b: 5e-5
  lr_embed_tokens: 8e-6

  max_grad_norm: 32.0
  adam_beta_1: 0.9
  adam_beta_2: 0.95
  adam_epsilon: 1e-8
  weight_decay: 1e-2

outputs:
  model_dir: ../models/eedi_embed_bge

wandb:
  project: eedi-dev
  run_name: encode-bge
  all_data_flag: false
  tags:
    - retriever


# %%writefile /kaggle/working/eedi-mining-misconceptions/code/train_llm_embedding.py
# import json
# import os
# import random
# import time
# from copy import deepcopy
# from dataclasses import dataclass

# import hydra
# import pandas as pd
# import torch
# from datasets import Dataset

# # local imports
# from llm_embedding.eedi_dataset import MathDataset
# from llm_embedding.eedi_loader import RetrieverDataCollator, TextCollator, show_batch, show_batch_fs
# from llm_embedding.eedi_model import BiEncoderModel, get_base_model
# from llm_embedding.eedi_optimizer import get_optimizer
# from omegaconf import OmegaConf
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from transformers.utils import logging as hf_logging
# from utils.metric_utils import compute_retrieval_metrics, mapk
# from utils.retriever_utils import semantic_search
# from utils.train_utils import as_minutes, download_kaggle_dataset, eedi_process_df, get_custom_cosine_schedule_with_warmup, get_lr, setup_training_run, train_valid_split

# # options
# pd.options.display.max_colwidth = 1000
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# hf_logging.set_verbosity_error()


# # utils -------------------------------------------------------------------------------------------#


# def get_temperature(cfg, progress):
#     """use for temperature scheduling"""
#     max_temp = cfg.model.max_temperature
#     min_temp = cfg.model.min_temperature
#     temp = max_temp - (max_temp - min_temp) * progress
#     return temp


# @dataclass
# class IDTracker:
#     """Track different IDs during training and evaluation"""

#     query_train_ids: list
#     query_valid_ids: list
#     content_train_ids: list
#     content_comp_ids: list


# # -------- Mine Hard Negatives --------------------------------------------------------------------#
# def mine_hard_negatives_online(cfg, accelerator, model, query_dl, content_dl, label_df, id_tracker):
#     query2content = label_df.groupby("query_id")["content_id"].apply(list).to_dict()

#     model.eval()

#     # query embeddings ---
#     query_embeddings = []
#     progress_bar = tqdm(range(len(query_dl)))
#     for batch in query_dl:
#         with torch.no_grad():
#             batch_query_embeddings = accelerator.unwrap_model(model).encode(batch)
#         batch_query_embeddings = accelerator.gather_for_metrics(batch_query_embeddings)
#         query_embeddings.append(batch_query_embeddings)
#         progress_bar.update(1)
#     progress_bar.close()

#     query_embeddings = torch.cat(query_embeddings, dim=0)
#     query_ids = id_tracker.query_train_ids
#     accelerator.print(f"shape of query embeddings (train): {query_embeddings.shape}")
#     assert query_embeddings.shape[0] == len(query_ids)

#     # content embeddings ---
#     content_embeddings = []
#     progress_bar = tqdm(range(len(content_dl)))
#     for batch in content_dl:
#         with torch.no_grad():
#             batch_content_embeddings = accelerator.unwrap_model(model).encode(batch)
#         batch_content_embeddings = accelerator.gather_for_metrics(batch_content_embeddings)
#         content_embeddings.append(batch_content_embeddings)
#         progress_bar.update(1)
#     progress_bar.close()

#     content_embeddings = torch.cat(content_embeddings, dim=0)
#     content_ids = id_tracker.content_train_ids
#     accelerator.print(f"shape of content embeddings (train): {content_embeddings.shape}")
#     assert content_embeddings.shape[0] == len(content_ids)

#     # search for negatives
#     hard_negatives_map = dict()
#     results = semantic_search(query_embeddings, content_embeddings, top_k=cfg.train_params.negative_depth_end + 1)

#     for idx, re_i in enumerate(results):
#         query_id = query_ids[idx]
#         hit_i = [node["corpus_id"] for node in re_i]
#         top_content_ids_i = [content_ids[pos] for pos in hit_i]

#         true_ids = query2content[query_id]
#         negative_ids = [x for x in top_content_ids_i if x not in true_ids]

#         hard_negatives_map[query_id] = negative_ids[cfg.train_params.negative_depth_start : cfg.train_params.negative_depth_end]
#     return hard_negatives_map


# # -------- Evaluation -----------------------------------------------------------------------------#


# def run_evaluation(cfg, accelerator, model, query_dl, content_dl, label_df, id_tracker):
#     cutoffs = [1, 2, 4, 8, 16, 25, 32, 64]
#     label_df = deepcopy(label_df)
#     query2content = label_df.groupby("query_id")["content_id"].apply(list).to_dict()

#     model.eval()

#     query_embeddings = []
#     progress_bar = tqdm(range(len(query_dl)))
#     for batch in query_dl:
#         with torch.no_grad():
#             batch_query_embeddings = accelerator.unwrap_model(model).encode(batch)

#         batch_query_embeddings = accelerator.gather_for_metrics(batch_query_embeddings)
#         query_embeddings.append(batch_query_embeddings)
#         progress_bar.update(1)
#     progress_bar.close()

#     query_embeddings = torch.cat(query_embeddings, dim=0)
#     query_ids = id_tracker.query_valid_ids
#     accelerator.print(f"shape of query embeddings: {query_embeddings.shape}")
#     assert query_embeddings.shape[0] == len(query_ids)

#     # get content embeddings ---
#     content_embeddings = []
#     progress_bar = tqdm(range(len(content_dl)))

#     for batch in content_dl:
#         with torch.no_grad():
#             batch_content_embeddings = accelerator.unwrap_model(model).encode(batch)
#         batch_content_embeddings = accelerator.gather_for_metrics(batch_content_embeddings)
#         content_embeddings.append(batch_content_embeddings)
#         progress_bar.update(1)
#     progress_bar.close()

#     content_embeddings = torch.cat(content_embeddings, dim=0)
#     content_ids = id_tracker.content_comp_ids
#     accelerator.print(f"shape of content embeddings: {content_embeddings.shape}")
#     assert content_embeddings.shape[0] == len(content_ids)

#     # ------ evaluation ----------------------------------------------------------------#
#     results = semantic_search(query_embeddings, content_embeddings, top_k=cfg.model.n_neighbour)

#     true_content_ids = []
#     pred_content_ids = []
#     pred_scores = []

#     for idx, re_i in enumerate(results):  # loop over query
#         query_id = query_ids[idx]
#         hit_i = [node["corpus_id"] for node in re_i]
#         top_scores_i = [node["score"] for node in re_i]
#         top_content_ids_i = [content_ids[pos] for pos in hit_i]
#         pred_content_ids.append(top_content_ids_i)
#         pred_scores.append(top_scores_i)
#         true_content_ids.append(query2content[query_id])

#     result_df = pd.DataFrame()
#     result_df["query_id"] = query_ids
#     result_df["true_ids"] = true_content_ids
#     result_df["pred_ids"] = pred_content_ids
#     result_df["pred_scores"] = pred_scores

#     # compute metric ----
#     eval_dict = dict()

#     for cutoff in cutoffs:
#         cdf = result_df.copy()
#         cdf["pred_ids"] = cdf["pred_ids"].apply(lambda x: x[:cutoff])
#         m = compute_retrieval_metrics(cdf["true_ids"].values, cdf["pred_ids"].values)

#         eval_dict[f"precision@{cutoff}"] = m["precision_score"]
#         eval_dict[f"recall@{cutoff}"] = m["recall_score"]

#     # get mapk ---
#     eval_dict["lb"] = mapk(result_df["true_ids"].values, result_df["pred_ids"].values, k=25)
#     accelerator.print(f">>> LB: {eval_dict['lb']}")

#     # seen vs unseen
#     content_train_ids = id_tracker.content_train_ids
#     result_df["seen"] = result_df["true_ids"].apply(lambda x: True if x[0] in content_train_ids else False)

#     seen_df = result_df[result_df["seen"]].reset_index(drop=True)
#     unseen_df = result_df[~result_df["seen"]].reset_index(drop=True)

#     eval_dict["seen_lb"] = mapk(seen_df["true_ids"].values, seen_df["pred_ids"].values, k=25)
#     eval_dict["unseen_lb"] = mapk(unseen_df["true_ids"].values, unseen_df["pred_ids"].values, k=25)

#     # get oof df
#     oof_df = result_df.copy()
#     oof_df = oof_df.drop(columns=["true_ids"])
#     oof_df = oof_df.rename(columns={"query_id": "QuestionId_Answer"})
#     oof_df = oof_df.rename(columns={"pred_ids": "MisconceptionId"})
#     oof_df["MisconceptionId"] = oof_df["MisconceptionId"].apply(lambda x: list(map(str, x)))
#     oof_df["MisconceptionId"] = oof_df["MisconceptionId"].apply(lambda x: " ".join(x))

#     to_return = {"lb": eval_dict["lb"], "scores": eval_dict, "result_df": result_df, "oof_df": oof_df}

#     # logs -----
#     scores_dict = eval_dict
#     accelerator.print("--------------------------------")
#     accelerator.print(f">>> LB: {scores_dict['lb']}")
#     accelerator.print(f">>> Seen LB: {scores_dict['seen_lb']}")
#     accelerator.print(f">>> Unseen LB: {scores_dict['unseen_lb']}")
#     accelerator.print("--------------------------------")

#     for pt in cutoffs:
#         accelerator.print(f">>> Current Recall@{pt} = {round(scores_dict[f'recall@{pt}'], 4)}")

#     return to_return


# # -------- Main Function --------------------------------------------------------------------------#
# @hydra.main(version_base=None, config_path="../conf/llm_embedding", config_name="conf_embedding")
# def run_training(cfg):
#     # ------- Runtime Configs ---------------------------------------------------------------------#
#     accelerator = setup_training_run(cfg)

#     def print_line(print_fn=accelerator.print):
#         prefix, unit, suffix = "#", "~~", "#"
#         print_fn(prefix + unit * 50 + suffix)

#     cfg.local_rank = accelerator.process_index
#     rng = random.Random(cfg.seed)

#     # ------- load data --------------------------------------------------------------------------#
#     with accelerator.main_process_first():
#         data_dir_comp = download_kaggle_dataset(cfg.dataset.comp_dataset)
#         data_dir = download_kaggle_dataset(cfg.dataset.input_dataset)
#         df = pd.read_csv(os.path.join(data_dir, "train.csv")).rename(columns={"QuestionId": "query_id"})
#         content_df = pd.read_csv(os.path.join(data_dir, "misconception_mapping.csv")).rename(columns={"MisconceptionId": "content_id"})
#         content_df_comp = pd.read_csv(os.path.join(data_dir_comp, "misconception_mapping.csv")).rename(columns={"MisconceptionId": "content_id"})
#         train_df, valid_df = train_valid_split(cfg, df)
#     accelerator.wait_for_everyone()

#     # process data --------------------------------------------------------------------------------#

#     query_df_train, corr_df_train, content2query_train = eedi_process_df(train_df, debug=False)
#     query_df_valid, corr_df_valid, _ = eedi_process_df(valid_df, debug=True)

#     train_content_ids = corr_df_train["content_id"].unique().tolist()
#     content_df_train = content_df[content_df["content_id"].isin(train_content_ids)]

#     print_line()
#     accelerator.print(f"# of queries (train): {query_df_train.shape[0]}")
#     accelerator.print(f"# of queries (valid): {query_df_valid.shape[0]}")

#     accelerator.print(f"# shape of content data (train): {content_df_train.shape}")
#     accelerator.print(f"# shape of content data (comp): {content_df_comp.shape}")

#     accelerator.print(f"shape of corr_df_train: {corr_df_train.shape}")
#     print_line()

#     # mappings ---
#     query2subject_train = dict(zip(query_df_train["query_id"], query_df_train["SubjectId"]))
#     query2content_train = dict(zip(corr_df_train["query_id"], corr_df_train["content_id"]))
#     subject2query_train = query_df_train.groupby("SubjectId")["query_id"].apply(list).to_dict()

#     # ------- Datasets ----------------------------------------------------------------------------#
#     ds_handle = MathDataset(cfg)
#     tokenizer = ds_handle.tokenizer

#     # query datasets ---
#     query_train_ds = ds_handle.get_dataset(query_df_train, is_query=True, is_train=True, rng=rng).sort("input_length")
#     query_valid_ds = ds_handle.get_dataset(query_df_valid, is_query=True).sort("input_length")

#     # content datasets ---
#     content_train_ds = ds_handle.get_dataset(content_df_train, is_query=False).sort("input_length")
#     content_comp_ds = ds_handle.get_dataset(content_df_comp, is_query=False).sort("input_length")

#     # retrieval dataset ---
#     retrieval_ds = Dataset.from_pandas(corr_df_train)

#     # manage ids ---
#     id_tracker = IDTracker(
#         query_train_ids=query_train_ds["query_id"],
#         query_valid_ids=query_valid_ds["query_id"],
#         content_train_ids=content_train_ds["content_id"],
#         content_comp_ids=content_comp_ds["content_id"],
#     )

#     # ------- data collators ----------------------------------------------------------------------#
#     text_collator = TextCollator(tokenizer=tokenizer)

#     # load hard negatives
#     print_line()
#     if cfg.train_params.load_hard_negatives:
#         accelerator.print(f"Loading hard negatives from {cfg.train_params.hard_negative_dataset}")
#         with accelerator.main_process_first():
#             hn_dir = download_kaggle_dataset(
#                 cfg.train_params.hard_negative_dataset
#             )  # cfg.train_params.hard_negative_dataset  #  download_kaggle_dataset(cfg.train_params.hard_negative_dataset)
#         accelerator.wait_for_everyone()
#         accelerator.print(f"hn_dir: {hn_dir}")

#         with open(os.path.join(hn_dir, cfg.train_params.hard_negative_file), "r") as f:
#             hard_negatives = json.load(f)

#         for k, v in hard_negatives.items():
#             hard_negatives[k] = v[: cfg.train_params.negative_depth_end]

#         # load teacher logits
#         with open(os.path.join(hn_dir, cfg.train_params.teacher_logits_file), "r") as f:
#             teacher_logits = json.load(f)
#         online_hard_negatives = dict()

#     else:
#         accelerator.print("Not loading hard negatives")
#         hard_negatives = dict()
#         teacher_logits = dict()
#         online_hard_negatives = dict()
#     print_line()

#     kwargs = dict(
#         cfg=cfg,
#         query_ds=query_train_ds,
#         content_ds=content_train_ds,
#         content2query=content2query_train,
#         negative_map=hard_negatives,
#         online_negative_map=online_hard_negatives,
#         teacher_logits=teacher_logits,
#         subject2queries=subject2query_train,
#         query2subject=query2subject_train,
#         query2content=query2content_train,
#     )

#     retriever_collator = RetrieverDataCollator(tokenizer=tokenizer, kwargs=kwargs)

#     # ------- data loaders ------------------------------------------------------------------------#
#     query_train_dl = DataLoader(
#         query_train_ds,
#         batch_size=cfg.train_params.query_bs,
#         shuffle=False,
#         collate_fn=text_collator,
#     )

#     query_valid_dl = DataLoader(
#         query_valid_ds,
#         batch_size=cfg.train_params.query_bs,
#         shuffle=False,
#         collate_fn=text_collator,
#     )

#     content_train_dl = DataLoader(
#         content_train_ds,
#         batch_size=cfg.train_params.content_bs,
#         shuffle=False,
#         collate_fn=text_collator,
#     )

#     content_comp_dl = DataLoader(
#         content_comp_ds,
#         batch_size=cfg.train_params.content_bs,
#         shuffle=False,
#         collate_fn=text_collator,
#     )

#     retrieval_dl = DataLoader(
#         retrieval_ds,
#         batch_size=cfg.train_params.retriever_bs,
#         shuffle=True,
#         collate_fn=retriever_collator,
#         num_workers=1,
#         drop_last=True,
#     )

#     # --- show batch ------------------------------------------------------------------------------#
#     print_line()

#     accelerator.print("showing a batch...")
#     for b in retrieval_dl:
#         break
#     show_batch(b, tokenizer, print_fn=accelerator.print)

#     print_line()
#     accelerator.print("showing first valid batch...")
#     for b in query_valid_dl:
#         break
#     show_batch_fs(b, tokenizer, print_fn=accelerator.print)
#     print_line()

#     # ------- Config ------------------------------------------------------------------------------#
#     print_line()
#     accelerator.print("Config for the current run")
#     cfg_dict = OmegaConf.to_container(cfg, resolve=True)
#     accelerator.print(json.dumps(cfg_dict, indent=4))
#     print_line()

#     # ------- Model -------------------------------------------------------------------------------#
#     print_line()
#     accelerator.print("Loading model....")
#     base_model = get_base_model(cfg)
#     model = BiEncoderModel(cfg, base_model, accelerator)

#     if cfg.model.gradient_checkpointing:
#         accelerator.print("enabling gradient checkpointing")
#         model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
#     if cfg.model.compile:
#         accelerator.print("compiling model")
#         model = torch.compile(model)
#     accelerator.print("Model loaded")
#     print_line()

#     # ------- Optimizer ---------------------------------------------------------------------------#
#     optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

#     # ------- Accelerator -------------------------------------------------------------------------#
#     model, optimizer, query_train_dl, query_valid_dl, content_train_dl, retrieval_dl, content_comp_dl = accelerator.prepare(
#         model, optimizer, query_train_dl, query_valid_dl, content_train_dl, retrieval_dl, content_comp_dl
#     )

#     # ------- Scheduler ---------------------------------------------------------------------------#
#     print_line()

#     num_epochs = cfg.train_params.num_epochs
#     grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
#     warmup_pct = cfg.train_params.warmup_pct

#     num_update_steps_per_epoch = len(retrieval_dl) // grad_accumulation_steps
#     num_training_steps = num_epochs * num_update_steps_per_epoch
#     num_warmup_steps = int(warmup_pct * num_training_steps)

#     accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
#     accelerator.print(f"# training steps: {num_training_steps}")
#     accelerator.print(f"# warmup steps: {num_warmup_steps}")

#     scheduler = get_custom_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

#     # ------- training setup ----------------------------------------------------------------------#
#     best_lb = -1.0
#     patience_tracker = 0
#     current_iteration = 0
#     progress = 0.0

#     start_time = time.time()
#     progress_bar = None

#     # initial evaluation --------------------------------------------------------------------------#
#     if cfg.train_params.eval_at_start:
#         model.eval()
#         eval_response = run_evaluation(cfg, accelerator, model, query_valid_dl, content_comp_dl, corr_df_valid, id_tracker)

#     for epoch in range(num_epochs):
#         if progress_bar:
#             progress_bar.close()

#         progress_bar = tqdm(range(num_update_steps_per_epoch))
#         print_line()
#         accelerator.print(f"Current epoch: {epoch+1}/{num_epochs}")
#         print_line()

#         # Training ------
#         model.train()
#         print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
#         for step, batch in enumerate(retrieval_dl):
#             temperature = torch.tensor(get_temperature(cfg, progress)).to(accelerator.device)
#             with accelerator.accumulate(model):
#                 # gives sync vs no sync context manager
#                 outputs = model(**batch, temperature=temperature)
#                 loss = outputs.loss
#                 accelerator.backward(loss)

#             if accelerator.sync_gradients:
#                 grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

#                 optimizer.step()
#                 scheduler.step()
#                 optimizer.zero_grad()

#             if accelerator.sync_gradients:
#                 progress_bar.set_description(
#                     f"STEP: {step+1:5}/{num_update_steps_per_epoch:5}. "
#                     f"T-STEP: {current_iteration+1:5}/{num_training_steps:5}. "
#                     f"LR: {get_lr(optimizer):.4f}. "
#                     f"Loss: {loss.item():.4f}. "
#                 )

#                 progress_bar.update(1)
#                 current_iteration += 1
#                 progress = current_iteration / num_training_steps

#                 if cfg.use_wandb:
#                     accelerator.log({"train_loss": round(loss.item(), 5)}, step=current_iteration)
#                     accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)
#                     accelerator.log({"total_grad_l2_norm": round(grad_norm.item(), 5)}, step=current_iteration)
#                     accelerator.log({"temperature": temperature.item()}, step=current_iteration)
#                     accelerator.log({"nce_loss": round(outputs.nce_loss.item(), 5)}, step=current_iteration)
#                     accelerator.log({"distill_loss": round(outputs.distill_loss.item(), 5)}, step=current_iteration)

#         # Evaluation -----
#         print_line()
#         et = as_minutes(time.time() - start_time)
#         accelerator.print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")

#         model.eval()
#         eval_response = run_evaluation(cfg, accelerator, model, query_valid_dl, content_comp_dl, corr_df_valid, id_tracker)

#         lb, scores_dict, result_df, oof_df = eval_response["lb"], eval_response["scores"], eval_response["result_df"], eval_response["oof_df"]
#         print_line()

#         # best scores and saving -----
#         is_best = False
#         if lb >= best_lb:
#             best_lb = lb
#             is_best = True
#             patience_tracker = 0

#             # -----
#             best_dict = dict()
#             for k, v in scores_dict.items():
#                 best_dict[f"{k}_at_best"] = v
#         else:
#             patience_tracker += 1

#         if is_best:
#             oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "oof_df_best.csv"), index=False)
#             result_df.to_csv(os.path.join(cfg.outputs.model_dir, "result_df_best.csv"), index=False)
#         else:
#             accelerator.print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
#             accelerator.print(f">>> current best score: {round(best_lb, 4)}")

#         oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "oof_df_last.csv"), index=False)
#         result_df.to_csv(os.path.join(cfg.outputs.model_dir, "result_df_last.csv"), index=False)

#         # saving -----
#         accelerator.wait_for_everyone()

#         # save checkpoint ---
#         if accelerator.is_main_process:
#             unwrapped_model = accelerator.unwrap_model(model)
#             unwrapped_model.save(cfg.outputs.model_dir)
#             tokenizer.save_pretrained(cfg.outputs.model_dir)

#         # logging ----
#         if cfg.use_wandb:
#             accelerator.log({"lb": lb}, step=current_iteration)
#             accelerator.log({"best_lb": best_lb}, step=current_iteration)
#             accelerator.log({"seen_lb": scores_dict["seen_lb"]}, step=current_iteration)
#             accelerator.log({"unseen_lb": scores_dict["unseen_lb"]}, step=current_iteration)

#             # -- log scores dict
#             for k, v in scores_dict.items():
#                 accelerator.log({k: round(v, 4)}, step=current_iteration)

#         # mine hard negatives & refresh dataloader ---
#         if (cfg.train_params.iterative_hard_negatives) & (cfg.train_params.num_hard_negatives > 0):
#             if epoch == num_epochs - 1:
#                 continue  # no need to mine hard negatives at the last epoch

#             if epoch >= cfg.train_params.iterative_hard_negatives_trigger:
#                 accelerator.print(f">>> Mining hard negatives at epoch {epoch+1}")
#                 online_hard_negatives = mine_hard_negatives_online(cfg, accelerator, model, query_train_dl, content_train_dl, corr_df_train, id_tracker)

#                 kwargs = dict(
#                     cfg=cfg,
#                     query_ds=query_train_ds,
#                     content_ds=content_train_ds,
#                     content2query=content2query_train,
#                     negative_map=hard_negatives,
#                     online_negative_map=online_hard_negatives,
#                     teacher_logits=teacher_logits,
#                     subject2queries=subject2query_train,
#                     query2subject=query2subject_train,
#                     query2content=query2content_train,
#                 )

#                 retriever_collator = RetrieverDataCollator(tokenizer=tokenizer, kwargs=kwargs)

#                 retrieval_dl = DataLoader(
#                     retrieval_ds,
#                     batch_size=cfg.train_params.retriever_bs,
#                     shuffle=True,
#                     collate_fn=retriever_collator,
#                     num_workers=1,
#                     drop_last=True,
#                 )
#                 retrieval_dl = accelerator.prepare(retrieval_dl)

#         # -- post eval
#         model.train()
#         torch.cuda.empty_cache()
#         print_line()

#         # early stopping ----
#         if patience_tracker >= cfg.train_params.patience:
#             return


# if __name__ == "__main__":
#     run_training()


# !accelerate config default


!pip install triton


!accelerate launch /kaggle/working/eedi-mining-misconceptions/code/train_llm_embedding.py --config-name conf_bge full_fit=true

