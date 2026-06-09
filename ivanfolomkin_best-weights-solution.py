import os
import sys
os.environ["MAMBA_SELECTIVE_SCAN"] = "0"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"



%pip install mambapy einops



%pip install matplotlib pandas


import time
import json
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch import optim
from torch.nn.functional import scaled_dot_product_attention

from pydantic import BaseModel
import einops
import warnings
warnings.filterwarnings('ignore')

try:
    from mambapy.mamba import Mamba, MambaConfig
    MAMBA2_AVAILABLE = True
except ImportError:
    MAMBA2_AVAILABLE = False

IGNORE_LABEL_ID = -100

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor

CosSin = Tuple[torch.Tensor, torch.Tensor]

def _find_multiple(a, b):
    return (-(a // -b)) * b

def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)

class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value))
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.register_buffer('weights', trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        if not self.training:
            return self.weights[inputs].to(self.cast_to)
        local_weights = self.weights[inputs].clone().requires_grad_(True)
        return local_weights.to(self.cast_to)

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, norm_eps=1e-5):
        super().__init__()
        if not MAMBA2_AVAILABLE:
            raise ImportError("mambapy is required for Mamba2Block")
        try:
            config = MambaConfig(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                n_layers=1,
            )
            self.mamba = Mamba(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mamba: {e}")

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Mamba2Block expects 3D input [batch, seq, dim], got {x.dim()}D")
        return self.mamba(x)

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False
    use_mamba2: bool = False
    puzzle_emb_len: int = 16
    no_ACT_continue: bool = True

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(0.05)
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,
                expansion=config.expansion,
            )
        elif self.config.use_mamba2:
            mamba_d_state = getattr(config, 'mamba_d_state', 16)
            mamba_d_conv = getattr(config, 'mamba_d_conv', 4)
            self.mamba2 = Mamba2Block(
                d_model=config.hidden_size,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=int(config.expansion)
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + self.dropout(out), variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        elif self.config.use_mamba2:
            hidden_states = rms_norm(hidden_states + self.dropout(self.mamba2(hidden_states)), variance_epsilon=self.norm_eps)
        else:
            hidden_states = rms_norm(hidden_states + self.dropout(self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)), variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + self.dropout(out), variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.output_dropout = nn.Dropout(0.15)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        H_init_tensor = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        L_init_tensor = trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1)
        self.register_buffer('H_init', H_init_tensor, persistent=True)
        self.register_buffer('L_init', L_init_tensor, persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        input_truncated = input[:, :self.config.seq_len]
        embedding = self.embed_tokens(input_truncated.to(torch.int32))
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        expected_seq_len = self.config.seq_len + self.puzzle_emb_len
        if embedding.shape[1] < expected_seq_len:
            pad_size = expected_seq_len - embedding.shape[1]
            embedding = F.pad(embedding, (0, 0, 0, pad_size))
        elif embedding.shape[1] > expected_seq_len:
            embedding = embedding[:, :expected_seq_len]
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        device = next(self.parameters()).device
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        batch_size = carry.z_H.shape[0]
        seq_len = carry.z_H.shape[1]
        device = carry.z_H.device
        reset_flag = reset_flag.to(device)
        H_init_tensor = self.H_init.to(device)
        L_init_tensor = self.L_init.to(device)
        H_init_expanded = H_init_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        L_init_expanded = L_init_tensor.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init_expanded, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init_expanded, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        z_H, z_L = carry.z_H, carry.z_L
        batch_size = z_H.shape[0]
        device = z_H.device
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                H_init_expanded = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, z_H.shape[1], -1).to(device)
                L_init_expanded = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, z_L.shape[1], -1).to(device)
                z_H = z_H * 0.9 + H_init_expanded * 0.1
                z_L = z_L * 0.9 + L_init_expanded * 0.1
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        H_init_expanded = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, z_H.shape[1], -1).to(device)
        L_init_expanded = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, z_L.shape[1], -1).to(device)
        z_H = z_H * 0.9 + H_init_expanded * 0.1
        z_L = z_L * 0.9 + L_init_expanded * 0.1
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(self.output_dropout(z_H))[:, self.puzzle_emb_len:]
        output = output[:, :self.config.seq_len]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))
        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

IGNORE_LABEL_ID = -100

class Puzzle:
    def __init__(self, puzzle_id, train_examples, test_examples):
        self.id = puzzle_id
        self.train = train_examples
        self.test = test_examples

def load_arc_data_from_kaggle(data_dir="TinyRecursiveModels/kaggle/combined"):
    train_puzzles = []
    eval_puzzles = []

    train_challenges_path = os.path.join(data_dir, "arc-agi_training_challenges.json")
    train_solutions_path = os.path.join(data_dir, "arc-agi_training_solutions.json")
    eval_challenges_path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    eval_solutions_path = os.path.join(data_dir, "arc-agi_evaluation_solutions.json")

    print(f"Looking for data files in: {os.path.abspath(data_dir)}")
    print(f"Train challenges exists: {os.path.exists(train_challenges_path)}")
    print(f"Train solutions exists: {os.path.exists(train_solutions_path)}")
    print(f"Eval challenges exists: {os.path.exists(eval_challenges_path)}")
    print(f"Eval solutions exists: {os.path.exists(eval_solutions_path)}")

    if os.path.exists(train_challenges_path):
        with open(train_challenges_path, "r") as f:
            train_challenges = json.load(f)

        train_solutions = {}
        if os.path.exists(train_solutions_path):
            with open(train_solutions_path, "r") as f:
                train_solutions = json.load(f)

        for puzzle_id, puzzle_data in train_challenges.items():
            train_examples = []
            for example in puzzle_data.get("train", []):
                train_examples.append({
                    "input": np.array(example["input"]),
                    "output": np.array(example["output"])
                })

            test_examples = []
            if puzzle_id in train_solutions:
                solution_data = train_solutions[puzzle_id]
                if isinstance(solution_data, dict):
                    solution_tests = solution_data.get("test", [])
                elif isinstance(solution_data, list):
                    solution_tests = solution_data
                else:
                    solution_tests = []

                for test_example in solution_tests:
                    if isinstance(test_example, dict):
                        test_examples.append({
                            "input": np.array(test_example["input"]),
                            "output": np.array(test_example["output"])
                        })
                    elif isinstance(test_example, list) and len(test_example) >= 2:
                        test_examples.append({
                            "input": np.array(test_example[0]),
                            "output": np.array(test_example[1])
                        })
            else:
                for test_example in puzzle_data.get("test", []):
                    test_input = np.array(test_example["input"])
                    test_examples.append({
                        "input": test_input,
                        "output": np.array([[0]])
                    })

            if len(train_examples) > 0 and len(test_examples) > 0:
                puzzle = Puzzle(puzzle_id, train_examples, test_examples)
                train_puzzles.append(puzzle)

    if os.path.exists(eval_challenges_path):
        with open(eval_challenges_path, "r") as f:
            eval_challenges = json.load(f)

        eval_solutions = {}
        if os.path.exists(eval_solutions_path):
            with open(eval_solutions_path, "r") as f:
                eval_solutions = json.load(f)

        for puzzle_id, puzzle_data in eval_challenges.items():
            train_examples = []
            for example in puzzle_data.get("train", []):
                train_examples.append({
                    "input": np.array(example["input"]),
                    "output": np.array(example["output"])
                })

            test_examples = []
            if puzzle_id in eval_solutions:
                solution_data = eval_solutions[puzzle_id]
                if isinstance(solution_data, dict):
                    solution_tests = solution_data.get("test", [])
                elif isinstance(solution_data, list):
                    solution_tests = solution_data
                else:
                    solution_tests = []

                for test_example in solution_tests:
                    if isinstance(test_example, dict):
                        test_examples.append({
                            "input": np.array(test_example["input"]),
                            "output": np.array(test_example["output"])
                        })
                    elif isinstance(test_example, list) and len(test_example) >= 2:
                        test_examples.append({
                            "input": np.array(test_example[0]),
                            "output": np.array(test_example[1])
                        })
            else:
                for test_example in puzzle_data.get("test", []):
                    test_input = np.array(test_example["input"])
                    test_examples.append({
                        "input": test_input,
                        "output": np.array([[0]])
                    })

            if len(train_examples) > 0:
                puzzle = Puzzle(puzzle_id, train_examples, test_examples)
                eval_puzzles.append(puzzle)

    return train_puzzles, eval_puzzles

class ARCReasoningDataset(Dataset):
    def __init__(self, arc_data, seq_len: int = 900, pad_token_id: int = 0):
        self.arc_data = arc_data
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.puzzle_to_id = {}
        self._build_puzzle_mapping()

    def _build_puzzle_mapping(self):
        for idx, puzzle in enumerate(self.arc_data):
            self.puzzle_to_id[puzzle.id] = idx

    def _create_reasoning_sequence(self, puzzle) -> Tuple[torch.Tensor, torch.Tensor]:
        context_parts = []
        for demo in puzzle.train:
            if isinstance(demo, dict):
                input_grid = demo["input"]
                output_grid = demo["output"]
            elif isinstance(demo, tuple):
                input_grid, output_grid = demo
            else:
                input_grid = demo.input
                output_grid = demo.output

            input_flat = input_grid.flatten()
            output_flat = output_grid.flatten()

            demo_seq = np.concatenate([
                input_flat + 1,
                output_flat + 1,
                [11]
            ])
            context_parts.append(demo_seq)

        context = np.concatenate(context_parts)

        if isinstance(puzzle.test, list):
            test_example = puzzle.test[0]
            if isinstance(test_example, dict):
                test_input_grid = test_example["input"]
                test_output_grid = test_example["output"]
            elif isinstance(test_example, tuple):
                test_input_grid, test_output_grid = test_example
            else:
                test_input_grid = test_example.input
                test_output_grid = test_example.output
        elif isinstance(puzzle.test, tuple):
            test_input_grid, test_output_grid = puzzle.test
        elif isinstance(puzzle.test, dict):
            test_input_grid = puzzle.test["input"]
            test_output_grid = puzzle.test["output"]
        else:
            test_input_grid = puzzle.test.input
            test_output_grid = puzzle.test.output

        test_input = test_input_grid.flatten() + 1

        full_input = np.concatenate([context, test_input])
        full_target = np.concatenate([
            np.full_like(context, IGNORE_LABEL_ID),
            test_output_grid.flatten() + 1
        ])

        input_tensor = torch.from_numpy(full_input).long()
        target_tensor = torch.from_numpy(full_target).long()

        # --- NEW: robust fixed-length packing ---
        seq_len = self.seq_len

        fixed_inputs = torch.full(
            (seq_len,),
            self.pad_token_id,
            dtype=torch.long
        )
        fixed_targets = torch.full(
            (seq_len,),
            IGNORE_LABEL_ID,
            dtype=torch.long
        )

        L_inp = min(seq_len, input_tensor.size(0))
        L_tgt = min(seq_len, target_tensor.size(0))

        fixed_inputs[:L_inp] = input_tensor[:L_inp]
        fixed_targets[:L_tgt] = target_tensor[:L_tgt]
        # --- end new block ---

        return fixed_inputs, fixed_targets

    def __len__(self):
        return len(self.arc_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        puzzle = self.arc_data[idx]
        inputs, targets = self._create_reasoning_sequence(puzzle)

        # these should now always be true
        assert inputs.size(0) == self.seq_len
        assert targets.size(0) == self.seq_len

        return {
            "inputs": inputs,
            "targets": targets,
            "labels": targets.clone(),
            "puzzle_identifiers": torch.tensor(
                self.puzzle_to_id[puzzle.id],
                dtype=torch.long
            )
        }


def create_arc_reasoning_dataloaders(
    train_data,
    eval_data,
    batch_size: int = 2,
    seq_len: int = 900,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:

    train_loader = None
    eval_loader = None

    if len(train_data) > 0:
        train_dataset = ARCReasoningDataset(train_data, seq_len=seq_len)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        raise ValueError("Cannot create DataLoader with empty training data!")

    if len(eval_data) > 0:
        eval_dataset = ARCReasoningDataset(eval_data, seq_len=seq_len)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        print("Warning: No evaluation data, creating empty eval_loader")
        eval_loader = DataLoader([], batch_size=batch_size)

    return train_loader, eval_loader



DATA_DIR = "/kaggle/input/arc-prize-2025"
train_data, eval_data = load_arc_data_from_kaggle(DATA_DIR)
print(f"Loaded {len(train_data)} training puzzles, {len(eval_data)} evaluation puzzles")

if len(train_data) == 0:
    raise ValueError("No training data loaded! Please check the data directory path and files.")
if len(eval_data) == 0:
    print("Warning: No evaluation data loaded!")

train_loader, eval_loader = create_arc_reasoning_dataloaders(
    train_data,
    eval_data,
    batch_size=2,
    seq_len=900,
)

if len(train_data) > 0:
    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(k, v.shape, v.dtype)



mamba_config = {
    "batch_size": 1,
    "seq_len": 300,
    "num_puzzle_identifiers": 1000,
    "vocab_size": 12,
    "H_cycles": 2,
    "L_cycles": 1,
    "H_layers": 4,
    "L_layers": 2,
    "hidden_size": 512,
    "expansion": 2.0,
    "num_heads": 6,
    "pos_encodings": "rope",
    "halt_max_steps": 8,
    "halt_exploration_prob": 0.1,
    "use_mamba2": True,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "puzzle_emb_ndim": 32,
    "forward_dtype": "float32",
    "no_ACT_continue": True
}


class ARCTrainingComparator:
    def __init__(self, train_loader, eval_loader, device="cuda", checkpoint_dir="checkpoints"):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.training_history = {}

    def setup_models(self, configs):
        for name, config in configs.items():
            model_config = config.copy()
            for key in ["learning_rate", "weight_decay", "epochs"]:
                model_config.pop(key, None)
            if "forward_dtype" not in model_config:
                model_config["forward_dtype"] = "float32"

            if model_config.get("use_mamba2", False):
                if self.device != "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("Mamba2 requires CUDA but CUDA is not available.")

            model = TinyRecursiveReasoningModel_ACTV1(model_config)
            model.to(self.device)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_size_mb = total_params * 4 / (1024 ** 2)

            print(f"\n{name}:")
            print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")
            print(f"  Model size: {param_size_mb:.2f} MB")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"  GPU memory: {memory_allocated:.2f} MB allocated, {memory_reserved:.2f} MB reserved")

            base_lr = config.get("learning_rate", 1e-4)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=base_lr,
                weight_decay=config.get("weight_decay", 0.3),
                eps=1e-8
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get("epochs", 100)
            )

            self.models[name] = model
            self.optimizers[name] = optimizer
            self.schedulers[name] = scheduler
            self.training_history[name] = {
                'train_loss': [], 'train_accuracy': [], 'train_perplexity': [],
                'eval_loss': [], 'eval_accuracy': [], 'eval_perplexity': [],
                'halt_steps': [], 'epoch_time': []
            }

    def save_checkpoint(self, model_name, model, optimizer, scheduler, epoch, metrics):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pt")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': model.config.__dict__
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def clear_memory(self, model_name=None):
        if model_name and model_name in self.models:
            del self.models[model_name]
            del self.optimizers[model_name]
            del self.schedulers[model_name]
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

    def compute_metrics(self, logits, labels):
        with torch.no_grad():
            batch_size = labels.shape[0]
            target_seq_len = labels.shape[1]

            if logits.shape[0] != batch_size:
                logits = logits[:batch_size]
            if logits.shape[1] != target_seq_len:
                if logits.shape[1] > target_seq_len:
                    logits = logits[:, :target_seq_len]
                else:
                    pad_size = target_seq_len - logits.shape[1]
                    logits = F.pad(logits, (0, 0, 0, pad_size))

            predictions = torch.argmax(logits, dim=-1)
            mask = labels != IGNORE_LABEL_ID
            if mask.sum() == 0:
                return {"accuracy": 0.0, "loss": float('inf'), "perplexity": float('inf')}
            accuracy = (predictions[mask] == labels[mask]).float().mean().item()
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_LABEL_ID,
                reduction='mean'
            ).item()
            perplexity = torch.exp(torch.tensor(loss)).item()
            return {"accuracy": accuracy, "loss": loss, "perplexity": perplexity}

    def train_epoch(self, model, optimizer, model_name, epoch=0, gradient_accumulation_steps=1):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        total_halt_steps = 0
        num_batches = 0
        accumulated_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f"Training {model_name}")

        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            carry = model.initial_carry(batch)
            all_logits = []
            halt_steps_list = []

            for step in range(model.config.halt_max_steps):
                carry, outputs = model(carry, batch)
                if torch.isnan(outputs["logits"]).any() or torch.isinf(outputs["logits"]).any():
                    break
                all_logits.append(outputs["logits"])
                halt_steps_list.append(carry.steps.float().mean().item())
                if carry.halted.all():
                    break

            if len(all_logits) == 0:
                del batch, carry
                continue

            logits = torch.stack(all_logits).mean(0)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                del batch, carry, logits
                continue

            labels = batch["labels"]
            batch_size = labels.shape[0]
            target_seq_len = labels.shape[1]

            if logits.shape[0] != batch_size:
                logits = logits[:batch_size]
            if logits.shape[1] != target_seq_len:
                if logits.shape[1] > target_seq_len:
                    logits = logits[:, :target_seq_len]
                else:
                    pad_size = target_seq_len - logits.shape[1]
                    logits = F.pad(logits, (0, 0, 0, pad_size))

            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)

            assert logits_flat.shape[0] == labels_flat.shape[0], f"Size mismatch: logits_flat={logits_flat.shape}, labels_flat={labels_flat.shape}"

            mask = labels_flat != IGNORE_LABEL_ID
            if mask.sum() > 0:
                logits_masked = logits_flat[mask]
                labels_masked = labels_flat[mask]
                vocab_size = logits_masked.size(-1)
                log_probs = torch.log_softmax(logits_masked, dim=-1)
                smooth_weight = 0.1
                ce_loss = nn.functional.nll_loss(log_probs, labels_masked, reduction='mean')
                uniform_loss = -log_probs.mean()
                loss = ((1.0 - smooth_weight) * ce_loss + smooth_weight * uniform_loss) / gradient_accumulation_steps
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True) / gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                del batch, carry, logits, loss
                continue

            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                max_grad_norm = 0.5 if "mamba" in model_name.lower() else 1.0
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if not (has_nan_grad or torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                    optimizer.step()
                optimizer.zero_grad()

                metrics = self.compute_metrics(logits, batch["labels"])

                total_loss += metrics["loss"]
                total_accuracy += metrics["accuracy"]
                total_perplexity += metrics["perplexity"]
                total_halt_steps += np.mean(halt_steps_list)
                num_batches += 1
                accumulated_loss = 0



                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "acc": f"{metrics['accuracy']:.4f}",
                        "halt": f"{np.mean(halt_steps_list):.1f}"
                    })

            del batch, carry, logits, loss
            if self.device == "cuda" and batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start
        if num_batches == 0:
            return {
                "train_loss": float('nan'),
                "train_accuracy": 0.0,
                "train_perplexity": float('nan'),
                "halt_steps": 0.0,
                "epoch_time": epoch_time
            }

        num_batches_safe = max(num_batches, 1)
        return {
            "train_loss": total_loss / num_batches_safe,
            "train_accuracy": total_accuracy / num_batches_safe,
            "train_perplexity": total_perplexity / num_batches_safe,
            "halt_steps": total_halt_steps / num_batches_safe,
            "epoch_time": epoch_time
        }

    def evaluate(self, model, model_name):
        model.eval()
        total_loss = 0
        total_accuracy = 0
        total_perplexity = 0
        total_halt_steps = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.eval_loader, desc=f"Evaluating {model_name}")):
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                carry = model.initial_carry(batch)
                all_logits = []
                halt_steps_list = []

                for step in range(model.config.halt_max_steps):
                    carry, outputs = model(carry, batch)
                    all_logits.append(outputs["logits"])
                    halt_steps_list.append(carry.steps.float().mean().item())
                    if carry.halted.all():
                        break

                logits = torch.stack(all_logits).mean(0)
                metrics = self.compute_metrics(logits, batch["labels"])

                total_loss += metrics["loss"]
                total_accuracy += metrics["accuracy"]
                total_perplexity += metrics["perplexity"]
                total_halt_steps += np.mean(halt_steps_list)
                num_batches += 1

                del batch, carry, logits
                if self.device == "cuda" and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

        if num_batches == 0:
            return {
                "eval_loss": float('nan'),
                "eval_accuracy": 0.0,
                "eval_perplexity": float('nan'),
                "eval_halt_steps": 0.0
            }

        num_batches_safe = max(num_batches, 1)
        return {
            "eval_loss": total_loss / num_batches_safe,
            "eval_accuracy": total_accuracy / num_batches_safe,
            "eval_perplexity": total_perplexity / num_batches_safe,
            "eval_halt_steps": total_halt_steps / num_batches_safe
        }

    def train_single_model(self, model_name, epochs, gradient_accumulation_steps=1):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        scheduler = self.schedulers[model_name]

        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()} - {epochs} EPOCHS")
        print(f"{'='*80}")

        for epoch in range(epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{epochs} - {model_name.upper()}")
            print(f"{'='*80}")

            train_metrics = self.train_epoch(
                model,
                optimizer,
                model_name,
                epoch,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            scheduler.step()
            eval_metrics = self.evaluate(model, model_name)

            all_metrics = {**train_metrics, **eval_metrics}

            for key, value in all_metrics.items():
                if key not in self.training_history[model_name]:
                    self.training_history[model_name][key] = []
                self.training_history[model_name][key].append(value)

            print(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Train Acc: {train_metrics['train_accuracy']:.4f} | "
                  f"Eval Acc: {eval_metrics['eval_accuracy']:.4f} | "
                  f"Halt Steps: {train_metrics['halt_steps']:.2f} | "
                  f"Time: {train_metrics['epoch_time']:.2f}s")

            self.save_checkpoint(model_name, model, optimizer, scheduler, epoch, all_metrics)

            if self.device == "cuda":
                torch.cuda.empty_cache()

        print(f"\n{model_name.upper()} training completed!")

    def train_comparison(self, epochs=50, gradient_accumulation_steps=1):
        model_names = list(self.models.keys())

        for model_name in model_names:
            self.train_single_model(model_name, epochs, gradient_accumulation_steps)
            print(f"\nClearing memory after {model_name} training...")
            self.clear_memory(model_name)
            print(f"Memory cleared. GPU memory: {torch.cuda.memory_allocated() / (1024**2):.2f} MB" if self.device == "cuda" else "Memory cleared.")

        self._final_analysis()

    def _print_epoch_comparison(self, results, epoch):
        print(f"\n{'-'*100}")
        print(f"COMPARISON - Epoch {epoch + 1}")
        print(f"{'-'*100}")
        print(f"{'Model':<20} {'Train Acc':<12} {'Eval Acc':<12} {'Eval Loss':<12} {'Halt Steps':<12} {'Time (s)':<12}")
        print(f"{'-'*100}")

        for model_name, metrics in results.items():
            print(f"{model_name:<20} "
                  f"{metrics['train_accuracy']:<12.4f} "
                  f"{metrics['eval_accuracy']:<12.4f} "
                  f"{metrics['eval_loss']:<12.4f} "
                  f"{metrics['halt_steps']:<12.2f} "
                  f"{metrics['epoch_time']:<12.2f}")

    def _final_analysis(self):
        print(f"\n{'='*120}")
        print("FINAL COMPARISON ANALYSIS")
        print(f"{'='*120}")

        analysis_data = []
        for model_name, history in self.training_history.items():
            if not history['eval_accuracy']:
                continue
            eval_acc_valid = [x for x in history['eval_accuracy'] if not (np.isnan(x) or np.isinf(x))]
            if not eval_acc_valid:
                continue

            final_accuracy = history['eval_accuracy'][-1]
            if np.isnan(final_accuracy) or np.isinf(final_accuracy):
                final_accuracy = eval_acc_valid[-1] if eval_acc_valid else 0.0

            best_accuracy = max(eval_acc_valid)
            convergence_epoch = np.argmax(history['eval_accuracy']) + 1
            avg_epoch_time = np.mean([x for x in history['epoch_time'] if not (np.isnan(x) or np.isinf(x))]) if history['epoch_time'] else 0.0
            avg_halt_steps = np.mean([x for x in history['halt_steps'] if not (np.isnan(x) or np.isinf(x))]) if history['halt_steps'] else 0.0

            analysis_data.append({
                'Model': model_name,
                'Final Accuracy': final_accuracy,
                'Best Accuracy': best_accuracy,
                'Convergence Epoch': convergence_epoch,
                'Avg Epoch Time (s)': avg_epoch_time,
                'Avg Halt Steps': avg_halt_steps
            })

        if not analysis_data:
            return

        df = pd.DataFrame(analysis_data)
        print("\nPERFORMANCE SUMMARY:")
        print(f"{'-'*100}")
        print(f"{'Model':<20} {'Final Acc':<12} {'Best Acc':<12} {'Conv Epoch':<12} {'Epoch Time':<12} {'Halt Steps':<12}")
        print(f"{'-'*100}")

        for _, row in df.iterrows():
            print(f"{row['Model']:<20} "
                  f"{row['Final Accuracy']:<12.4f} "
                  f"{row['Best Accuracy']:<12.4f} "
                  f"{row['Convergence Epoch']:<12} "
                  f"{row['Avg Epoch Time (s)']:<12.2f} "
                  f"{row['Avg Halt Steps']:<12.2f}")

        best_acc_valid = df['Best Accuracy'].dropna()
        if not best_acc_valid.empty:
            best_overall = df.loc[best_acc_valid.idxmax()]
            print(f"\nBest Overall: {best_overall['Model']} (Accuracy: {best_overall['Best Accuracy']:.4f})")

    def plot_training_results(self, model_name, save_dir="plots"):
        if model_name not in self.training_history:
            print(f"No training history found for {model_name}")
            return

        os.makedirs(save_dir, exist_ok=True)
        history = self.training_history[model_name]

        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Results: {model_name}', fontsize=16, fontweight='bold')

        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['eval_loss'], 'r-', label='Eval Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['eval_accuracy'], 'r-', label='Eval Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(epochs, history['train_perplexity'], 'b-', label='Train Perplexity', linewidth=2)
        axes[0, 2].plot(epochs, history['eval_perplexity'], 'r-', label='Eval Perplexity', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Perplexity')
        axes[0, 2].set_title('Perplexity')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, history['halt_steps'], 'g-', label='Halt Steps', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_title('Halt Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, history['epoch_time'], 'm-', label='Epoch Time', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        if 'eval_halt_steps' in history:
            axes[1, 2].plot(epochs, history['eval_halt_steps'], 'c-', label='Eval Halt Steps', linewidth=2)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Steps')
            axes[1, 2].set_title('Eval Halt Steps')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis('off')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f'{model_name}_training_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.show()

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle(f'Training vs Evaluation: {model_name}', fontsize=14, fontweight='bold')

        axes2[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        axes2[0].plot(epochs, history['eval_loss'], 'r-', label='Eval Loss', linewidth=2, marker='s', markersize=4)
        axes2[0].set_xlabel('Epoch', fontsize=12)
        axes2[0].set_ylabel('Loss', fontsize=12)
        axes2[0].set_title('Loss Comparison', fontsize=12)
        axes2[0].legend(fontsize=10)
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
        axes2[1].plot(epochs, history['eval_accuracy'], 'r-', label='Eval Accuracy', linewidth=2, marker='s', markersize=4)
        axes2[1].set_xlabel('Epoch', fontsize=12)
        axes2[1].set_ylabel('Accuracy', fontsize=12)
        axes2[1].set_title('Accuracy Comparison', fontsize=12)
        axes2[1].legend(fontsize=10)
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path2 = os.path.join(save_dir, f'{model_name}_comparison.png')
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {plot_path2}")
        plt.show()


mamba_training_config = {
    "mamba": {
        **mamba_config,
        "learning_rate": 1e-5,
        "weight_decay": 0.1,
        "epochs": 10
    }
}

train_set, eval_set = load_arc_data_from_kaggle(DATA_DIR)
train_loader, eval_loader = create_arc_reasoning_dataloaders(
    train_set,
    eval_set,
    batch_size=1,
    seq_len=900
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")



print("="*80)
print("TRAINING MAMBA MODEL")
print("="*80)

mamba_comparator = ARCTrainingComparator(
    train_loader, eval_loader, device,
    checkpoint_dir="checkpoints/mamba"
)
mamba_comparator.setup_models(mamba_training_config)
mamba_comparator.train_single_model("mamba", epochs=10, gradient_accumulation_steps=2)
mamba_comparator.save_checkpoint("mamba",
    mamba_comparator.models["mamba"],
    mamba_comparator.optimizers["mamba"],
    mamba_comparator.schedulers["mamba"],
    9, {})
print("\nClearing memory after mamba training...")
mamba_comparator.clear_memory("mamba")
if device == "cuda":
    print(f"GPU memory after clearing: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")

print(f"\nMamba training completed!")
print(f"Checkpoints saved to: {mamba_comparator.checkpoint_dir}")

print("\n" + "="*80)
print("PLOTTING TRAINING RESULTS")
print("="*80)
mamba_comparator.plot_training_results("mamba", save_dir="plots")






