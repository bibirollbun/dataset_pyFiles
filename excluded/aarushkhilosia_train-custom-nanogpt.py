!pip install -U torch torchvision torchaudio datasets trl accelerate torchao transformers --extra-index-url https://download.pytorch.org/whl/cu121


!HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Aarushhh/SEWY3-246M-untrained


!HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Aarushhh/finemath-refined --repo-type dataset





# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Tuple, Union
from torch import Tensor, nn
import torch
import torch.nn as nn
import torch.utils.checkpoint
import  torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, HybridCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaMLP,
    GemmaModel,
    GemmaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)


_CHECKPOINT_FOR_DOC = "google/Sewy3-7b"

logger = logging.get_logger(__name__)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))
class Sewy3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Sewy3Model`]. It is used to instantiate an Sewy3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Sewy3-7B.
    e.g. [google/Sewy3-7b](https://huggingface.co/google/Sewy3-7b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Sewy3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Sewy3Model`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
            if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256): scaling factor used on the attention scores
        sliding_window (`int`, *optional*, defaults to 4096): in Sewy3, every other layer uses sliding window attention. This is the
            size of the sliding window.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0): scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

    ```python
    >>> from transformers import Sewy3Model, Sewy3Config
    >>> # Initializing a Sewy3 Sewy3-7b style configuration
    >>> configuration = Sewy3Config()
    >>> # Initializing a model from the Sewy3-7b style configuration
    >>> model = Sewy3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "Sewy3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        cache_implementation="hybrid",
        num_ve = 3,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation
        self.num_ve = num_ve


class Sewy3RMSNorm(GemmaRMSNorm):
    pass


class Sewy3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = CastedLinear(self.hidden_size, self.intermediate_size)
        self.up_proj = CastedLinear(self.hidden_size, self.intermediate_size)
        self.down_proj = CastedLinear(self.intermediate_size, self.hidden_size)
        self.act_fn = ACT2FN[config.hidden_activation]
        self.down_proj.weight.detach().zero_()
    def forward(self, x):
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Sewy3Attention(GemmaAttention):
    def __init__(self, config: Sewy3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        std = 0.5 * (config.hidden_size ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        self.q_proj = CastedLinear(
            config.hidden_size, config.num_attention_heads * self.head_dim
        )
        self.k_proj = CastedLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim
        )
        self.v_proj = CastedLinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim
        )
        self.o_proj = CastedLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size
        )
        self.o_proj.weight.detach().zero_()
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = True
        # self.scaling = config.query_pre_attn_scalar**-0.5
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None
        self.scaling = 0.12
        self.num_heads = config.num_attention_heads
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))



    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ve = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # bsz, seq_len, _ = hidden_states.shape(0), hidden_states.shape(1), hidden_states.shape(2)
        # query_states, key_states, value_states = F.linear(hidden_states, self.qkv_proj.flatten(end_dim=1).type_as(hidden_states)).view(bsz, seq_len, 3*self.num_heads, -1).chunk(3, dim=-2)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # print(f"ve shape: {ve.shape}")
        # print(f"value_states shape: {value_states.shape}")
        # print(f"hidden_states shape: {hidden_states.shape}")
        # print(f"lambdas[0] shape: {self.lambdas[0].shape}")
        # print(f"lambdas[1] shape: {self.lambdas[1].shape}")
        
        # if ve is not None:
        #     value_states = self.lambdas[0] * value_states + self.lambdas[1] * ve.view_as(value_states) # @KoszarskyB & @Grad62304977
        if ve is not None:
            # Reshape ve to match value_states dimensions
            batch_size, num_heads, seq_len, head_dim = value_states.shape
            ve = ve.unsqueeze(1).expand(-1, num_heads, -1, -1)  # Expand across num_heads dimension
            value_states = self.lambdas[0] * value_states + self.lambdas[1] * ve
        # else: # skip mid-layers token value embeddings by @YouJiacheng
        #     value_states = self.lambdas[0] * value_states

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Sewy3DecoderLayer(nn.Module):
    def __init__(self, config: Sewy3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.is_sliding = not bool(layer_idx % 2)
        self.self_attn = Sewy3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Sewy3MLP(self.config)
        self.input_layernorm = Sewy3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Sewy3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx
        self.pre_feedforward_layernorm = Sewy3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Sewy3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        ve =None,
        x0: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # Flash-attn is a 2D tensor
            if self.config._attn_implementation == "flash_attention_2":
                if past_key_value is not None:  # when decoding
                    attention_mask = attention_mask[:, -self.sliding_window :]
            else:
                min_dtype = torch.finfo(hidden_states.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                if attention_mask.shape[-1] <= 1:  # when decoding
                    attention_mask = attention_mask[:, :, :, -self.sliding_window :]

        # print("x0 type is ", x0.dtype)
        # print("lambdas: ", self.lambdas)
        # print("lambas [0] is ", self.lambdas[0])
        # print("lambas [1] is ", self.lambdas[1])
        hidden_states = self.lambdas[0] * hidden_states + self.lambdas[1] * x0


        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            ve=ve
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def norm(x):
    return F.rms_norm(x, (x.size(-1),))
class Sewy3Model(GemmaModel):
    def __init__(self, config: Sewy3Config):
        super().__init__(config)
        self.layers = nn.ModuleList()
        self.value_embeds = ValueEmbedding(config.vocab_size, config)
        
        self.num_encoder_layers = config.num_hidden_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_hidden_layers - self.num_encoder_layers # Remaining for decoder
        self.encoder_layers = nn.ModuleList(
            [Sewy3DecoderLayer(config, layer_idx) for layer_idx in range(self.num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [Sewy3DecoderLayer(config, layer_idx) for layer_idx in range(self.num_decoder_layers)]
        )
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                device=self.device,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )


        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Sewy3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        x0 = hidden_states

        ve = self.value_embeds(input_ids)
        assert len(ve) == len(self.encoder_layers) + len(self.decoder_layers)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers

        skip_connections = []


        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None


        for encoding_layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoding_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    ve_enc[encoding_layer.layer_idx],
                    x0, 
                )
            else:
                layer_outputs = encoding_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    ve=ve_enc[encoding_layer.layer_idx],
                    x0 = x0,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            skip_connections.append(hidden_states)


        for decoder_layer in self.decoder_layers:

            hidden_states = hidden_states + self.skip_weights[decoder_layer.layer_idx] * skip_connections.pop()
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    ve_dec[decoder_layer.layer_idx],
                    x0
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    ve=ve_dec[decoder_layer.layer_idx],
                    x0 = x0,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)




        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()
 
    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Sewy3 work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, config: Sewy3Config): 
        super().__init__()
        num_layers = config.num_hidden_layers
        self.num_layers = num_layers
        hidden_size = config.head_dim
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, hidden_size) for _ in range(config.num_ve)])

    def forward(self, input_seq) -> list[Tensor | None]:
        ve = [emb(input_seq) for emb in self.embed]
        # Distribute VEs across layers with None padding
        result = [None] * self.num_layers
        # Place VEs at start and end with equal spacing
        ve_positions = list(range(0, len(ve))) + list(range(-len(ve), 0)) 
        for i, pos in enumerate(ve_positions):
            result[pos] = ve[i % len(ve)]
        return result



class Sewy3ForCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Sewy3Model(config)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size)
        self.lm_head.weight.detach().zero_()        
        # self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""

        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Sewy3 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten: has a special cache type, `HybridCache`

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s
                # `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride
                # during the decoding. Here, simply using `.contiguous()` is not sufficient as in the
                # batch size = 1 case, `position_ids` is already contiguous but with varying stride
                # which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if (
            isinstance(past_key_values, HybridCache)
            and attention_mask.ndim == 2
            and not self.config._attn_implementation == "flash_attention_2"
        ):
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class Sewy3ForSequenceClassification(GemmaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = Sewy3Model(config)
        self.post_init()


class Sewy3ForTokenClassification(GemmaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.model = Sewy3Model(config)
        self.post_init()


__all__ = [
    "Sewy3Config",
    "Sewy3ForCausalLM",
    "Sewy3Model",
    "Sewy3PreTrainedModel",  # noqa: F822
    "Sewy3ForSequenceClassification",
    "Sewy3ForTokenClassification",
]


# Load model 
import torch, torchao
from transformers import AutoModelForCausalLM, AutoTokenizer,TorchAoConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
print(torch.__version__)
quantization_config=None
model = Sewy3ForCausalLM.from_pretrained("Aarushhh/SEWY3-246M-untrained", device_map='cuda:0', torch_dtype=torch.float16, max_position_embeddings=2048, quantization_config=quantization_config , attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("Aarushhh/SEWY3-246M-untrained", trust_remote_code=True, use_fast=True)


dataset = load_dataset("Aarushhh/finemath-refined", split="train[:10%]")


model, dataset


# model.model.encoder_layers[0].self_attn.o_proj.weight, model.lm_head.weight, model.model.encoder_layers[0].mlp.down_proj.weight


import os
import torch
import torch.distributed as dist

# @torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class SWAN(torch.optim.Optimizer):
    """
    SWAN - SGD with normalization and Newton-schulz orthogonalization
    used Muon as base
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.8, 0.95), adamw_eps=1e-8, adamw_wd=0, model=None):
        self.ngpt_model = model
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        self.gradnorms = []            
        super().__init__(params, defaults)
    
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['use_muon'] = False
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2:
                if p.size(0) < 10000 and p.size(1) < 10000:
                    self.state[p]['use_muon'] = True
                    self.gradnorms.append(torch.nn.LayerNorm(p.size(), elementwise_affine=False).to(p.device))
                else:
                    self.state[p]['use_muon'] = False
            else:
                self.state[p]['use_muon'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure=None):
        
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_muon']]
            gradnorms = self.gradnorms
            lr = group['lr']
            # momentum = group['momentum']

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    # state = self.state[p]
                    # if 'momentum_buffer' not in state:
                    #     state['momentum_buffer'] = torch.zeros_like(g)
                    # buf = state['momentum_buffer']
                    # buf.mul_(momentum).add_(g)
                    # if group['nesterov']:
                    #     g = g.add(buf, alpha=momentum)
                    # else:
                    #     g = buf
                    g = gradnorms[i](g)
                    g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in params:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['adamw_lr_ratio'] * group['lr'] # in order for lr schedule to work
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1
                step = state['step']
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)

        return loss


optimizer = SWAN(model.parameters(), lr=0.02,adamw_lr=0.01, model=model)


from torch.optim.lr_scheduler import _LRScheduler

class WarmupCooldownScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, cooldown_steps, final_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.final_lr = final_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [self.initial_lr * warmup_factor for _ in self.base_lrs]
        
        elif self.last_epoch < (self.warmup_steps + self.cooldown_steps):
            # Linear decay
            cooldown_factor = 1.0 - ((self.last_epoch - self.warmup_steps) / self.cooldown_steps)
            current_lr = self.initial_lr * cooldown_factor + self.final_lr * (1 - cooldown_factor)
            return [current_lr for _ in self.base_lrs]
        
        else:
            # Constant final learning rate
            return [self.final_lr for _ in self.base_lrs]


scheduler = WarmupCooldownScheduler(optimizer, 100, 1000, 0.01)


max_seq_length= 2048

sft_config = SFTConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    save_total_limit=5,
    # Use num_train_epochs = 1, warmup_ratio for full training runs!
    num_train_epochs =1,
    warmup_ratio = 0.0,
    # learning_rate = 0.008,
    fp16 = True,
    bf16 = False,
    logging_steps = 1,
    # weight_decay = 0.0,
    lr_scheduler_type = "constant",
    seed = 6969,
    output_dir = "outputs",
    # optim='adamw_torch_fused',
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    gradient_checkpointing=True,    
    
)





trainer = SFTTrainer(
    
    model = model,
    optimizers=(optimizer, scheduler),
    tokenizer = tokenizer,
    train_dataset = dataset,    
    args = sft_config
)


import os
os.environ['WANDB_API_KEY'] = ""
os.environ['WANDB_PROJECT'] = 'try sewy3 246M'


# model = torch.compile(model, fullgraph=True, mode='max-autotune')


from accelerate import notebook_launcher
notebook_launcher(trainer.train(), num_processes=2,gradient_as_bucket_view=True)


# trainer.train()








# !printenv > /kaggle/working/kaggle_env_vars.txt
# !git clone https://github.com/buidai123/Kaggle_VSCode_Remote_SSH.git /kaggle/working/Kaggle_VSCode_Remote_SSH
%cd /kaggle/working/Kaggle_VSCode_Remote_SSH
!pip install -r requirements.txt
!chmod +x setup_kaggle_ssh.py setup_ssh.sh
!./setup_ssh.sh https://raw.githubusercontent.com/AarushCodes/SSH_public_key/refs/heads/main/authorized_keys
!python3 setup_kaggle_ssh.py 1ZqxbRGbcXZIPxa2ZBR1YqAczby_35RbHQTQMkTpAWJpLs8AB


[rank0]:[W121 13:38:04.571165924 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
[rank0]:[W121 13:38:06.024848453 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

