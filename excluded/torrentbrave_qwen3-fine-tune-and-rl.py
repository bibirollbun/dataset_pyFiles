def clean_memory(deep=False):
    import gc
    import ctypes
    import torch
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# 及时清理 预防显存满了
# del llm
# clean_memory(deep=True)




# evalute-offical 中不要使用多卡测试结果
# 不要从base上开始，因为不是全量微调的话会有乱码，并且没有思考非思考标签并且如果用unsloth做，会连结束符都输出不出来，会一直在重复输出
# 蒸馏的前提是要训练大的教师模型
# 如果后训练部分只有一个微调，就没必要了，除非还有强化学习


# unsloth用的都是free,只能支持单卡,所以如果模型量很小,单卡很快,是超过deepspeed的
# llama-factory是在web-UI界面进行,自由度低


clean_memory(deep=True)


# clean_memory(deep=True)


# import subprocess
# import os

# result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
# output = result.stdout
# for line in output.splitlines():
#     if '=' in line:
#         var, value = line.split('=', 1)
#         os.environ[var] = value

# from huggingface_hub import notebook_login

# notebook_login()


## 大模型趋势是一切任务token化, 所以都是CAUSAL_LM


import warnings
warnings.filterwarnings("ignore")


!pip install /kaggle/input/trl-install/trl-0.19.0-py3-none-any.whl


# 把网关了再开就又通上了，才能再下载


!pip install deepspeed


!pip install trl


!pip show trl


!pip show deepspeed


!pip install /kaggle/input/bitsandbytes-install/bitsandbytes-0.46.0-py3-none-manylinux_2_24_x86_64.whl


# !pip install -qq swanlab


# %%writefile /kaggle/working/train.py
# import numpy


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from peft import LoraConfig, TaskType, get_peft_model
import deepspeed
DS_CONFIG = "/kaggle/input/qwen3pipline/微调/基于原生transformers/推理数据集微调/ds_zero2_no_offload.json"
from typing import Optional, List, Union
import sys


%%writefile /kaggle/working/train.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datasets import load_dataset,Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from peft import LoraConfig, TaskType, get_peft_model
import deepspeed
DS_CONFIG = "/kaggle/input/qwen3pipline/微调/基于原生transformers/推理数据集微调/ds_zero2_no_offload.json"
from typing import Optional, List, Union
import sys

model_name = "/kaggle/input/qwen-3/transformers/4b/1"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

ds = load_dataset(
    path="/kaggle/input/medical-o1-reasoning-sft-t0619",
    name="zh",  # 如果存在多个配置（如 'zh', 'en' 等），需要指定
    split="train[:500]"
)

# ---- 对数据做的处理 | 只有这两个代码块 ----
def generate_conversation(examples):
    questions  = examples["Question"]
    cots = examples["Complex_CoT"]
    solutions = examples["Response"]
    conversations = []
    for question,cot,solution in zip(questions,cots, solutions):
        conversations.append([ 
            {"role" : "user",      "content" : question},
            {"role" : "assistant", "content" : f'<think>{cot}</think>{solution}'}, 
        ])
    return { "conversations": conversations, }

# 将转换后的推理数据集应用对话模板
reasoning_conversations = tokenizer.apply_chat_template(
    ds.map(generate_conversation, batched = True)["conversations"],
    tokenize = False
)
# ---- 对数据做的处理 | 只有这两个代码块 ----

df = pd.DataFrame({"text": reasoning_conversations})
train_ds = Dataset.from_pandas(df).shuffle(seed = 3407)

# 以往需要截断, 要把长度统一, 这里SFTTrainer一股脑输入进去就好
trainer = SFTTrainer(
    model = model,
    processing_class=tokenizer,
    train_dataset = train_ds,
    eval_dataset = None,  # 可以设置评估数据集
    args = SFTConfig(
        output_dir="./lora_model",
        per_device_train_batch_size = 1,  # 每个设备的训练批次大小
        gradient_accumulation_steps = 16,  # 使用梯度累积模拟更大批次大小 | 越大占用的显存越小, 但是精度也越差
        warmup_steps = 5,  # 预热步数
        num_train_epochs = 4,  # 训练轮数设置为1以进行完整训练
        learning_rate = 2e-4,   # 学习率（长期训练可降至2e-5）
        logging_steps = 5,  # 日志记录间隔
        optim = "adamw_8bit",  # 优化器 在训练时优化器对于显存占很大部分 | 显存不够 | 模型参数用 bf16 | 优化器用8bit
        weight_decay = 0.01,  # 权重衰减
        lr_scheduler_type = "linear",  # 学习率调度类型
        seed = 3407,  # 随机种子
        report_to = "none",   # 可设置为"wandb"等进行实验追踪
        fp16=True,
        max_grad_norm=1.0, # 梯度裁剪
        deepspeed=DS_CONFIG, # 设置多卡配置, 训练的时候指定一下文件,再放进去
        logging_first_step=5,
        save_steps=100,
    ),
)

# 显示当前内存统计
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 显示最终内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


class CaptureStreamer(TextStreamer):
    """
    把流式输出捕获到, 因为只有先捕获到才能写入文档
    """
    def __init__(self, tokenizer, skip_prompt: bool = False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.generated_text = ""  # 用于存储完整输出

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """重写方法捕获最终文本"""
        self.generated_text += text  # 累积输出
        super().on_finalized_text(text, stream_end=stream_end)  # 保持原样输出到终端

    def get_output(self) -> str:
        """获取完整生成内容"""
        return self.generated_text.strip()

def ask(question, is_thinking=True, save_to_file=None):
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 使用自定义的 CaptureStreamer
    streamer = CaptureStreamer(tokenizer, skip_prompt=True)

    # 生成响应
    model.eval()  # 确保模型在推理模式
    with torch.no_grad():
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=1024,
            temperature=0.6, # 翻译任务需要小一点，稳定，越大越发散 
            top_p=0.95, # 0.8 0.1 0.04 0.03 超过的就不要 0.03 只在前三个中选  两种采样策略 top_p优先级更高
            top_k=20, # 自回归模型每步预测词表大小个概率值,概率和为1,在概率最高的而是个词中进行采样
            streamer=streamer,  # 关键：使用自定义的 streamer, 流式输出,直接把结果放到打印台打印出来了, 原来是要generated_ids=model.generate={**model_inputs, max_new_tokens=32768}拿出来, .tolist()再截取一下就是output_ids
        )
    # 流式输出发现看不懂,不是因为模型没训练好，而是因为用多卡分布式训练同时输出了,只有单卡训练时才不会乱
    # 为了解决这个问题就重写方法捕获最终的输出
    # 获取完整输出
    full_output = streamer.get_output()

    # 保存到文件
    if save_to_file:
        try:
            with open(save_to_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"✅ 成功写入文件: {save_to_file}")
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

    return full_output

# 测试调用
ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",
    save_to_file='./output.txt')
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")
print("#############################################################################################")


# ask("根据描述，一个1岁的孩子在夏季头皮出现多处小结节，长期不愈合，且现在疮大如梅，溃破流脓，口不收敛，头皮下有空洞，患处皮肤增厚。这种病症在中医中诊断为什么病？",is_thinking=False)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")


!deepspeed --include 'localhost:0,1,2,3' /kaggle/working/train.py


# ![图片.png](attachment:ee2eece3-a51f-49b6-84fb-3ccb1be2aa2d.png)
# ![图片.png](attachment:810d4d6c-b33d-42cd-96f8-945a1d35b276.png)
# ![图片.png](attachment:810d4d6c-b33d-42cd-96f8-945a1d35b276.png)
# ![图片.png](attachment:63a4ac18-3713-41fb-ba0b-a5617a825056.png)


# 不想要老选某个FFN的Expert,肯定不能靠梯度反传,因为会变差,不想让它变差,就使用load balancing loss, 既能不选它给其他experts多些训练机会,还不会让它变差
# deepseek-v3中就是给load balancing loss 乘上一个很大的权重, 给更新参数造成损失间接影响expert性能的乘上一个很小的权重,一般都是混合用,但是用权重权衡


# 使用字节级别的分词器 Byte-level 151669 - 151642 vocab.json中差的部分(specital token)在tokenizer_config.json中存着


# Qwen3的扩展性好, 不管什么模态在LLM看来都是token, 只是用specital token标志一下就好


# 现在大模型基本上都是decoder-only


# ![图片.png](attachment:4f487f21-6d1f-4718-868d-4d833572e1bd.png)


# KVcache：把历史上用过的K、V保存下来，这样在下一次计算的时候就不用再重算一次了
# Key和Value是一组的,有多少Key就有对应多少Value,因为KVcache中K和V是成对保存的
# 更极端的做法是就一个K一个V,但是效果肯定变差了，KV就一个，存不住cache,根据Scaling Laws，模型参数越多越好，基本上只在小模型上用

# hidden_state * c，就保存这个c，把K、V的大小也改变了，就像是lora微调一样，保存的是小矩阵
# 就像是lora中用两个小矩阵存储参数，但是最后两个小矩阵相乘又能还原成一个大矩阵
# 这里的MLA不是使用两个小矩阵相乘，而是直接使用 latent KV 和 hidden_state 相乘，就可以作为新的K和v了

# MHA是乘以一个正方形的矩阵得到的K、V，而MLA是乘以一个小矩阵的到的，秩会更小


# 都要搭配KVcache使用


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # 门控控制，决定哪些特征要保留或抑制
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # 升维
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False) # 降维 projection的中文意思是从一个空间
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# 相对位置编码是给每个位置加的不是绝对的数字，而是相对于这一行或者这一列的值，结果就是不能做KV-cache, 
# 每个token都是向量，向量之间都有角度，角度变化了就做不了KV cache了
# 旋转位置编码同时包含相对位置编码(能实现长文本，但不能做K V cache)的信息和绝对位置编码(不能做长文本, 但是能做 K V cache)的信息
# 到了旋转位置编码，就只需要在Q、K上加位置编码了
# Q和K相乘得到的正方形矩阵就是关联度---注意力得分矩阵


# ![图片.png](attachment:8243ce78-7dc2-4cb2-9f6f-6940588643b1.png)
# ![图片.png](attachment:bd214c9d-ab70-4639-af8d-2f023fd3fd0d.png)

# - Value是每个token的内容或"携带的信息"
# - 已经知道了每个token对其他token的关注权重,所以乘上V后，相当于对所有信息的信息加权平均,得到一个新的表示


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = getattr(config, "sliding_window", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # qk_norm也是RMSNorm
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# #                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# #           This file was automatically generated from src/transformers/models/qwen3_moe/modular_qwen3_moe.py.
# #               Do NOT edit this file manually as any edits will be overwritten by the generation of
# #             the file from the modular. If any change should be done, please apply the change to the
# #                          modular_qwen3_moe.py file directly. One of our CI enforces this.
# #                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# # coding=utf-8
# # Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from typing import Callable, Optional, Union

# import torch
# import torch.nn.functional as F
# from torch import nn

# from ...activations import ACT2FN
# from ...cache_utils import Cache, DynamicCache
# from ...generation import GenerationMixin
# from ...integrations import use_kernel_forward_from_hub
# from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
# from ...modeling_flash_attention_utils import FlashAttentionKwargs
# from ...modeling_layers import GradientCheckpointingLayer
# from ...modeling_outputs import (
#     BaseModelOutputWithPast,
#     MoeCausalLMOutputWithPast,
#     MoeModelOutputWithPast,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutputWithPast,
#     TokenClassifierOutput,
# )
# from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# from ...processing_utils import Unpack
# from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
# from .configuration_qwen3_moe import Qwen3MoeConfig


# logger = logging.get_logger(__name__)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# def eager_attention_forward(
#     module: nn.Module,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attention_mask: Optional[torch.Tensor],
#     scaling: float,
#     dropout: float = 0.0,
#     **kwargs,
# ):
#     key_states = repeat_kv(key, module.num_key_value_groups)
#     value_states = repeat_kv(value, module.num_key_value_groups)

#     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
#     if attention_mask is not None:
#         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#         attn_weights = attn_weights + causal_mask

#     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
#     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
#     attn_output = torch.matmul(attn_weights, value_states)
#     attn_output = attn_output.transpose(1, 2).contiguous()

#     return attn_output, attn_weights


# class Qwen3MoeAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = self.head_dim**-0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = True

#         self.q_proj = nn.Linear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.v_proj = nn.Linear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.o_proj = nn.Linear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
#         )
#         self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
#         self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
#         self.sliding_window = getattr(config, "sliding_window", None)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         past_key_value: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)

#         query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#         key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
#         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         attention_interface: Callable = eager_attention_forward
#         if self.config._attn_implementation != "eager":
#             attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             sliding_window=self.sliding_window,  # diff with Llama
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         attn_output = self.o_proj(attn_output)
#         return attn_output, attn_weights


# class Qwen3MoeMLP(nn.Module):
#     def __init__(self, config, intermediate_size=None):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(self, x):
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj


# class Qwen3MoeSparseMoeBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_experts = config.num_experts
#         self.top_k = config.num_experts_per_tok
#         self.norm_topk_prob = config.norm_topk_prob

#         # gating
#         self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
#         self.experts = nn.ModuleList(
#             [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
#         )

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         """ """
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         # router_logits: (batch * sequence_length, n_experts)
#         router_logits = self.gate(hidden_states)

#         routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
#         routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
#         if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
#             routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
#         # we cast back to the input dtype
#         routing_weights = routing_weights.to(hidden_states.dtype)

#         final_hidden_states = torch.zeros(
#             (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
#         )

#         # One hot encode the selected experts to create an expert mask
#         # this will be used to easily index which expert is going to be sollicitated
#         expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

#         # Loop over all available experts in the model and perform the computation on each expert
#         expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
#         for expert_idx in expert_hitted:
#             expert_layer = self.experts[expert_idx]
#             idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

#             # Index the correct hidden states and compute the expert hidden state for
#             # the current expert. We need to make sure to multiply the output hidden
#             # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
#             current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
#             current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

#             # However `index_add_` only support torch tensors for indexing so we'll use
#             # the `top_x` tensor here.
#             final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
#         final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
#         return final_hidden_states, router_logits


# @use_kernel_forward_from_hub("RMSNorm")
# class Qwen3MoeRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Qwen3MoeRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)

#     def extra_repr(self):
#         return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
#     def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = Qwen3MoeAttention(config, layer_idx)

#         if (layer_idx not in config.mlp_only_layers) and (
#             config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
#         ):
#             self.mlp = Qwen3MoeSparseMoeBlock(config)
#         else:
#             self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

#         self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[tuple[torch.Tensor]] = None,
#         output_attentions: Optional[bool] = False,
#         output_router_logits: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
#                 `(batch, sequence_length)` where padding elements are indicated by 0.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_router_logits (`bool`, *optional*):
#                 Whether or not to return the logits of all the routers. They are useful for computing the router loss,
#                 and should not be returned during inference.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#             cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#                 Indices depicting the position of the input sequence tokens in the sequence.
#             position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
#                 Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
#                 with `head_dim` being the embedding dimension of each attention head.
#             kwargs (`dict`, *optional*):
#                 Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
#                 into the model
#         """

#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)

#         hidden_states = self.mlp(hidden_states)
#         if isinstance(hidden_states, tuple):
#             hidden_states, router_logits = hidden_states
#         else:
#             router_logits = None

#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if output_router_logits:
#             outputs += (router_logits,)

#         return outputs


# class Qwen3MoeRotaryEmbedding(nn.Module):
#     def __init__(self, config: Qwen3MoeConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     @torch.no_grad()
#     @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
#     def forward(self, x, position_ids):
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
#         position_ids_expanded = position_ids[:, None, :].float()

#         device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):  # Force float32
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos() * self.attention_scaling
#             sin = emb.sin() * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# @auto_docstring
# class Qwen3MoePreTrainedModel(PreTrainedModel):
#     config_class = Qwen3MoeConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = True
#     _no_split_modules = ["Qwen3MoeDecoderLayer"]
#     _skip_keys_device_placement = ["past_key_values"]
#     _supports_flash_attn_3 = True
#     _supports_flash_attn_2 = True
#     _supports_sdpa = True
#     _supports_flex_attn = True
#     _supports_cache_class = True
#     _supports_quantized_cache = True
#     _supports_static_cache = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
#     _supports_attention_backend = True

#     def _init_weights(self, module):
#         std = self.config.initializer_range
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, Qwen3MoeRMSNorm):
#             module.weight.data.fill_(1.0)


# @auto_docstring
# class Qwen3MoeModel(Qwen3MoePreTrainedModel):
#     def __init__(self, config: Qwen3MoeConfig):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size

#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
#         self.layers = nn.ModuleList(
#             [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
#         )
#         self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[list[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         output_router_logits: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
#     ) -> MoeModelOutputWithPast:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_router_logits = (
#             output_router_logits if output_router_logits is not None else self.config.output_router_logits
#         )
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         if use_cache and past_key_values is None:
#             past_key_values = DynamicCache()

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )
#         if position_ids is None:
#             position_ids = cache_position.unsqueeze(0)

#         mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
#         causal_mask = mask_function(
#             config=self.config,
#             input_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             cache_position=cache_position,
#             past_key_values=past_key_values,
#         )

#         hidden_states = inputs_embeds

#         # create position embeddings to be shared across the decoder layers
#         position_embeddings = self.rotary_emb(hidden_states, position_ids)

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         all_router_logits = () if output_router_logits else None

#         for decoder_layer in self.layers:
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             layer_outputs = decoder_layer(
#                 hidden_states,
#                 attention_mask=causal_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_values,
#                 output_attentions=output_attentions,
#                 output_router_logits=output_router_logits,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#                 **flash_attn_kwargs,
#             )

#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

#             if output_router_logits:
#                 all_router_logits += (layer_outputs[-1],)

#         hidden_states = self.norm(hidden_states)

#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         return MoeModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=past_key_values,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#             router_logits=all_router_logits,
#         )


# class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# def load_balancing_loss_func(
#     gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
#     num_experts: Optional[int] = None,
#     top_k=2,
#     attention_mask: Optional[torch.Tensor] = None,
# ) -> Union[torch.Tensor, int]:
#     r"""
#     Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

#     See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
#     function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
#     experts is too unbalanced.

#     Args:
#         gate_logits:
#             Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
#             shape [batch_size X sequence_length, num_experts].
#         num_experts:
#             Number of experts
#         top_k:
#             The number of experts to route per-token, can be also interpreted as the `top-k` routing
#             parameter.
#         attention_mask (`torch.Tensor`, *optional*):
#             The attention_mask used in forward function
#             shape [batch_size X sequence_length] if not None.

#     Returns:
#         The auxiliary loss.
#     """
#     if gate_logits is None or not isinstance(gate_logits, tuple):
#         return 0

#     if isinstance(gate_logits, tuple):
#         compute_device = gate_logits[0].device
#         concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

#     routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

#     _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

#     expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

#     if attention_mask is None:
#         # Compute the percentage of tokens routed to each experts
#         tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

#         # Compute the average probability of routing to these experts
#         router_prob_per_expert = torch.mean(routing_weights, dim=0)
#     else:
#         batch_size, sequence_length = attention_mask.shape
#         num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

#         # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
#         expert_attention_mask = (
#             attention_mask[None, :, :, None, None]
#             .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
#             .reshape(-1, top_k, num_experts)
#             .to(compute_device)
#         )

#         # Compute the percentage of tokens routed to each experts
#         tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
#             expert_attention_mask, dim=0
#         )

#         # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
#         router_per_expert_attention_mask = (
#             attention_mask[None, :, :, None]
#             .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
#             .reshape(-1, num_experts)
#             .to(compute_device)
#         )

#         # Compute the average probability of routing to these experts
#         router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
#             router_per_expert_attention_mask, dim=0
#         )

#     overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
#     return overall_loss * num_experts


# @auto_docstring
# class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}
#     _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = Qwen3MoeModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
#         self.router_aux_loss_coef = config.router_aux_loss_coef
#         self.num_experts = config.num_experts
#         self.num_experts_per_tok = config.num_experts_per_tok

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def set_decoder(self, decoder):
#         self.model = decoder

#     def get_decoder(self):
#         return self.model

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[list[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         output_router_logits: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         **kwargs: Unpack[KwargsForCausalLM],
#     ) -> MoeCausalLMOutputWithPast:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

#         >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
#         >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_router_logits = (
#             output_router_logits if output_router_logits is not None else self.config.output_router_logits
#         )

#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs: MoeModelOutputWithPast = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             output_router_logits=output_router_logits,
#             cache_position=cache_position,
#             **kwargs,
#         )

#         hidden_states = outputs.last_hidden_state
#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

#         aux_loss = None
#         if output_router_logits:
#             aux_loss = load_balancing_loss_func(
#                 outputs.router_logits,
#                 self.num_experts,
#                 self.num_experts_per_tok,
#                 attention_mask,
#             )
#             if labels is not None:
#                 loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

#         return MoeCausalLMOutputWithPast(
#             loss=loss,
#             aux_loss=aux_loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             router_logits=outputs.router_logits,
#         )


# @auto_docstring(
#     custom_intro="""
#     The Qwen3Moe Model transformer with a sequence classification head on top (linear layer).

#     [`Qwen3MoeForSequenceClassification`] uses the last token in order to do the classification, as other causal models
#     (e.g. GPT-2) do.

#     Since it does classification on the last token, it requires to know the position of the last token. If a
#     `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
#     no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
#     padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
#     each row of the batch).
#     """
# )
# class Qwen3MoeForSequenceClassification(Qwen3MoePreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = Qwen3MoeModel(config)
#         self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#     ) -> SequenceClassifierOutputWithPast:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """

#         transformer_outputs: BaseModelOutputWithPast = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         hidden_states = transformer_outputs.last_hidden_state
#         logits = self.score(hidden_states)

#         if input_ids is not None:
#             batch_size = input_ids.shape[0]
#         else:
#             batch_size = inputs_embeds.shape[0]

#         if self.config.pad_token_id is None and batch_size != 1:
#             raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
#         if self.config.pad_token_id is None:
#             last_non_pad_token = -1
#         elif input_ids is not None:
#             # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
#             non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
#             token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
#             last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
#         else:
#             last_non_pad_token = -1
#             logger.warning_once(
#                 f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
#                 "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
#             )

#         pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

#         return SequenceClassifierOutputWithPast(
#             loss=loss,
#             logits=pooled_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#         )


# @auto_docstring
# class Qwen3MoeForTokenClassification(Qwen3MoePreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.model = Qwen3MoeModel(config)
#         if getattr(config, "classifier_dropout", None) is not None:
#             classifier_dropout = config.classifier_dropout
#         elif getattr(config, "hidden_dropout", None) is not None:
#             classifier_dropout = config.hidden_dropout
#         else:
#             classifier_dropout = 0.1
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.score = nn.Linear(config.hidden_size, config.num_labels)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.model.embed_tokens

#     def set_input_embeddings(self, value):
#         self.model.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#     ) -> TokenClassifierOutput:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """

#         outputs: BaseModelOutputWithPast = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         sequence_output = outputs.last_hidden_state
#         sequence_output = self.dropout(sequence_output)
#         logits = self.score(sequence_output)

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits, labels, self.config)

#         return TokenClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @auto_docstring
# class Qwen3MoeForQuestionAnswering(Qwen3MoePreTrainedModel):
#     base_model_prefix = "transformer"

#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = Qwen3MoeModel(config)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.transformer.embed_tokens

#     def set_input_embeddings(self, value):
#         self.transformer.embed_tokens = value

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         start_positions: Optional[torch.LongTensor] = None,
#         end_positions: Optional[torch.LongTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         **kwargs,
#     ) -> QuestionAnsweringModelOutput:
#         outputs: BaseModelOutputWithPast = self.transformer(
#             input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )

#         sequence_output = outputs.last_hidden_state

#         logits = self.qa_outputs(sequence_output)
#         start_logits, end_logits = logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         loss = None
#         if start_positions is not None and end_positions is not None:
#             loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

#         return QuestionAnsweringModelOutput(
#             loss=loss,
#             start_logits=start_logits,
#             end_logits=end_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# __all__ = [
#     "Qwen3MoeForCausalLM",
#     "Qwen3MoeForQuestionAnswering",
#     "Qwen3MoeModel",
#     "Qwen3MoePreTrainedModel",
#     "Qwen3MoeForSequenceClassification",
#     "Qwen3MoeForTokenClassification",
# ]







%%writefile /kaggle/working/train_distillation.py


from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments,DataCollatorForSeq2Seq
# from dataset import SFTDataset
from utils import compute_fkl
import os
import json
from datasets import load_dataset,Dataset
from bitsandbytes.optim import AdamW8bit
from transformers import get_cosine_schedule_with_warmup
import deepspeed
DS_CONFIG = "ds_zero2_no_offload.json"
from dataset_med import MedicalQANoPaddingDataset
from torch.serialization import add_safe_globals
import numpy._core.multiarray


class KGTrainer(Trainer):
    def __init__(
        self,
        model=None,
        teacher_model=None,
        if_use_entropy=False,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
    def _load_rng_state(self, checkpoint):
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if os.path.exists(rng_file):
            # 强制禁用 weights_only
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
            torch.set_rng_state(checkpoint_rng_state["torch"])
            np.random.set_state(checkpoint_rng_state["numpy"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint_rng_state["cuda"])

    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # 注意这里的缩进

        device = next(model.parameters()).device
        # 将所有张量移动到指定的设备上 , 传入到学生模型之前，需要先传到学生模型同样的设备上
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        
        outputs = model(**inputs) # **inputs中包含了label、input_ids、attentation_mask
        with torch.no_grad(): # 教师模型不更新参数
            device = next(self.teacher_model.parameters()).device
            # 将所有张量移动到指定的设备上
            for key, value in inputs.items():
                inputs[key] = value.to(device)
            teacher_outputs = self.teacher_model(**inputs)
        
        loss = outputs.loss/4
        
        labels = inputs['labels']
        logits = outputs.logits


        teacher_logits = teacher_outputs.logits

        # 8B的hidden_size是4096，17B的hidden_size是2048
        # 教师和学生要对齐一下，只取前2048个维度的hidden_szie
        # 维度对齐：蒸馏时需确保教师和学生模型的特征空间一致，通常通过截断或投影实现
        # hidden_size：模型内部表示的向量维度，决定表达能力。
        # logits：模型最后一层的原始输出，用于计算预测概率
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]
        
        
        labels.to(device)
        kl = compute_fkl(logits, teacher_logits, labels, padding_id=-100, temp=1.0)
        # print(f"kl=={kl} loss=={loss}\n\n")
        if self.if_use_entropy:
            loss_total = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl

            
        return (loss_total, outputs) if return_outputs else loss_total

if __name__ == '__main__':
    add_safe_globals([numpy._core.multiarray._reconstruct])
    
    # 学生模型
    student_name = "/kaggle/input/qwen-3/transformers/4b/1"
    # teacher_name = "merged_model_4b"
    teacher_name = "/kaggle/input/qwen-3/transformers/8b/1"
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
    model = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        # device_map="cuda:1",
        attn_implementation="flash_attention_2"
    )
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
    # 使用lora方法训练
    model = get_peft_model(model, lora_config)
    
    print(model.print_trainable_parameters())
    
    tokenizer = AutoTokenizer.from_pretrained(student_name,use_cache=False)
    
    # 教师模型
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        # device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )

    teacher_model.eval()
    
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=6, 
                            do_train=True, 
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=4,
                            gradient_checkpointing=True,
                            logging_steps=10,
                            report_to='tensorboard',
                            save_strategy='epoch',
                            save_total_limit=1,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            warmup_ratio=0.1,
                            max_grad_norm=1.0,
                            optim = "adamw_8bit",
                            fp16=False,
                            dataloader_num_workers=2,
                            gradient_checkpointing_kwargs={"use_reentrant": False},
                            deepspeed=DS_CONFIG,
                            dataloader_pin_memory=True
                            )
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)


    dataset = MedicalQANoPaddingDataset(
        data_path="combined_medical_data.json",
        tokenizer=tokenizer,
        max_seq_len=2048
    )
    data_collator = DefaultDataCollator()
    # 计算总训练步数
    total_steps = len(dataset) // args.per_device_train_batch_size * args.num_train_epochs
    




    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        if_use_entropy = True,
                        args=args, 
                        train_dataset=dataset, 
                        tokenizer=tokenizer, 
                        # optimizers=(optimizer,lr_scheduler),
                        data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./results')
    trainer.save_state()




