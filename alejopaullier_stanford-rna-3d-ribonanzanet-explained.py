


import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from functools import partialmethod
from torch import einsum
from torch.nn.parameter import Parameter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


%%writefile config.yaml

learning_rate: 0.001  # The learning rate for the optimizer
batch_size: 4  # Number of samples per batch
test_batch_size: 8  # Number of samples per batch
epochs: 30  # Total training epochs
optimizer: "ranger"  # Optimization algorithm
dropout: 0.05  # Dropout regularization rate
weight_decay: 0.0001
k: 5
ninp: 256
nlayers: 9
nclass: 2
ntoken: 5  # AUGC + padding/N token
nhead: 8
use_bpp: False
use_flip_aug: true
bpp_file_folder: "../../input/bpp_files/"
gradient_accumulation_steps: 1
use_triangular_attention: false
pairwise_dimension: 64
use_bpp: False

#Data scaling
use_data_percentage: 1
use_dirty_data: true  # turn off for data scaling and data dropout experiments

# Other configurations
fold: 0
nfolds: 6
input_dir: "../../input/"
gpu_id: "0"


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)
        

def default(val: Any, d: Any) -> Any:
    """
    Returns `val` if it is not None, otherwise returns the default value `d`.
    :param val: The primary value.
    :param d: The default value to return if `val` is None.
    :return: `val` if it is not None, otherwise `d`.
    """
    return val if exists(val) else d


def exists(val: Any) -> bool:
    """
    Checks whether a given value is not None.
    :param val: The value to check.
    :return: True if `val` is not None, otherwise False.
    """
    return val is not None


def init_weights(m: torch.nn.Module) -> None:
    """
    Initializes the weights of a given module if it is an instance of `torch.nn.Linear`. 
    Currently, the function does not apply any initialization but has commented-out 
    Xavier initialization methods.
    :param m: The module to initialize, expected to be a `torch.nn.Linear` instance.
    :return: None
    """
    if m is not None and isinstance(m, nn.Linear):
        pass


def load_config_from_yaml(file_path):
    """Load YAML file"""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)


def sep():
    print("â€”"*100)


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class GeM(nn.Module):
    """
    1-dimensional GeM pooling.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        kernel_size = (x.size(-1))
        output = F.avg_pool1d(
            x.clamp(min=self.eps).pow(self.p), 
            kernel_size
        ).pow(1./self.p)
        return output

    def __repr__(self):
        return f'GeM(p={self.p}, eps={self.eps})'


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention module, computing attention scores based on query and key similarity.
    '''
    
    def __init__(self, temperature: float, attn_dropout: float = 0.1) -> None:
        """
        Initializes the Scaled Dot-Product Attention module.
        
        :param temperature: Scaling factor for the dot product attention scores.
        :param attn_dropout: Dropout rate applied to attention weights.
        """
        super().__init__()
        self.temperature: float = temperature
        self.dropout: nn.Dropout = nn.Dropout(attn_dropout)

    def forward(
        self, 
        q: torch.Tensor,  # (B, nhead, L, d_k)
        k: torch.Tensor,  # (B, nhead, L, d_k)
        v: torch.Tensor,  # (B, nhead, L, d_v)
        mask: torch.Tensor | None = None,  # (B, 1, L, L) or None
        attn_mask: torch.Tensor | None = None  # (B, 1, L, L) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Scaled Dot-Product Attention.
        
        :param q: Query tensor of shape (B, nhead, L, d_k), where B is batch size, nhead is the number of attention heads,
                  L is the sequence length, and d_k is the key/query dimension.
        :param k: Key tensor of shape (B, nhead, L, d_k).
        :param v: Value tensor of shape (B, nhead, L, d_v), where d_v is the value dimension.
        :param mask: Optional bias mask tensor of shape (B, 1, L, L), used for causal masking or padding.
        :param attn_mask: Optional attention mask tensor of shape (B, 1, L, L), where -1 values indicate positions to mask.
        :return: Tuple containing:
            - output (torch.Tensor): The result of the attention mechanism, shape (B, nhead, L, d_v).
            - attn (torch.Tensor): Attention weights after softmax and dropout, shape (B, nhead, L, L).
        """
        
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature  # (B, nhead, L, L)
        
        if mask is not None:
            attn = attn + mask  # Apply bias mask (B, nhead, L, L)
        
        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float('-1e-9'))  # Apply attention mask (B, nhead, L, L)
        
        attn = self.dropout(F.softmax(attn, dim=-1))  # (B, nhead, L, L)
        output = torch.matmul(attn, v)  # (B, nhead, L, d_v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    :param d_model: The number of input features.
    :param n_head: The number of heads to use.
    :param d_k: The dimensionality of the keys.
    :param d_v: The dimensionality of the values.
    :param dropout: The dropout rate to apply to the attention weights.
    """
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)  # (d_model) -> (n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  # (d_model) -> (n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)  # (d_model) -> (n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)  # (n_head * d_v) -> (d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(
        self, 
        q: torch.Tensor,  # Shape: [batch_size, len_q, d_model]
        k: torch.Tensor,  # Shape: [batch_size, len_k, d_model]
        v: torch.Tensor,  # Shape: [batch_size, len_v, d_model]
        mask: Optional[torch.Tensor] = None,  # Optional attention mask
        src_mask: Optional[torch.Tensor] = None  # Optional source mask
               
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Returns (output, attention)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        bs, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q  # Shape: [bs, len_q, d_model]

        # Linear projections and reshape to multiple heads
        q = self.w_qs(q).view(bs, len_q, n_head, d_k)  # Shape: [bs, len_q, n_head, d_k]
        k = self.w_ks(k).view(bs, len_k, n_head, d_k)  # Shape: [bs, len_k, n_head, d_k]
        v = self.w_vs(v).view(bs, len_v, n_head, d_v)  # Shape: [bs, len_v, n_head, d_v]

        # Transpose for multi-head attention computation
        q, k, v = (
            q.transpose(1, 2),  # Shape: [bs, n_head, len_q, d_k]
            k.transpose(1, 2),  # Shape: [bs, n_head, len_k, d_k]
            v.transpose(1, 2)
        )  # Shape: [bs, n_head, len_v, d_v]

        if mask is not None:
            mask = mask  # Shape remains unchanged

        if src_mask is not None:
            src_mask[src_mask == 0] = -1
            src_mask = src_mask.unsqueeze(-1).float()  # Shape: [bs, len_k, 1]
            attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(1)  
            # Shape: [bs, 1, len_k, len_k]
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)

        # Reshape back to original format
        q = q.transpose(1, 2).contiguous().view(bs, len_q, -1)  # Shape: [bs, len_q, n_head * d_v]
        q = self.dropout(self.fc(q))  # Shape: [bs, len_q, d_model]
        q += residual  # Shape: [bs, len_q, d_model]

        q = self.layer_norm(q)  # Shape: [bs, len_q, d_model]

        return q, attn  # Output: [bs, len_q, d_model], Attention: [bs, n_head, len_q, len_k]



class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 200
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor of shape (seq_len, batch_size, d_model)
        :return: tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class OuterProductMean(nn.Module):
    """
    Outer Product Mean class.
    :param in_dim: Dimensionality of the input sequence representations (default: 256).
    :param dim_msa: Intermediate lower-dimensional representation (default: 32).
    :param pairwise_dim: Final dimensionality of the pairwise output (default: 64).
    """
    def __init__(
        self,
        in_dim: int = 256,
        dim_msa: int = 32,
        pairwise_dim: int = 64
    ):
        super(OuterProductMean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)  # projects the input sequence representation into a lower dimensional space
        self.proj_down2 = nn.Linear(dim_msa ** 2, pairwise_dim)  # projects the outer product representation (reshaped) to the final pairwise_dim.

    def forward(
        self,
        seq_rep: torch.Tensor,  # shape: (batch_size, seq_length, in_dim)
        pair_rep: torch.Tensor = None  # shape: (batch_size, seq_length, seq_length, pairwise_dim)
    ):
        seq_rep = self.proj_down1(seq_rep)  # output shape: (batch_size, seq_length, dim_msa)
        outer_product = torch.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)  # output shape: (batch_size, seq_length, seq_length, dim_msa, dim_msa)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')  # flattens the last two dimensions: (batch_size, seq_length, seq_length, dim_msa ** 2).
        outer_product = self.proj_down2(outer_product)  # output shape: (batch_size, seq_length, seq_length, pairwise_dim)

        if pair_rep is not None:
            outer_product = outer_product + pair_rep

        return outer_product 


def exists(val):
    return val is not None

def default(val, d):
    return val if val is not None else d

class TriangleMultiplicativeModule(nn.Module):
    """
    This class is applied to the pairwise residue representations, ensuring that the predicted distances 
    between residues adhere to the triangle inequality principle.
    """
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: Optional[int] = None,
        mix: str = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # Initialize all gating to identity
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x: torch.Tensor,                  # (batch_size, seq_len, seq_len, dim)
        src_mask: Optional[torch.Tensor] = None  # (batch_size, seq_len)
    ) -> torch.Tensor:                    # Output: (batch_size, seq_len, seq_len, dim)
        if exists(src_mask):
            src_mask = src_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
            mask = rearrange(mask, 'b i j -> b i j ()')  # (batch_size, seq_len, seq_len, 1)

        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        
        x = self.norm(x)  # (batch_size, seq_len, seq_len, dim)

        left = self.left_proj(x)  # (batch_size, seq_len, seq_len, hidden_dim)
        right = self.right_proj(x) # (batch_size, seq_len, seq_len, hidden_dim)

        if exists(src_mask):
            left = left * mask  # (batch_size, seq_len, seq_len, hidden_dim)
            right = right * mask # (batch_size, seq_len, seq_len, hidden_dim)

        left_gate = self.left_gate(x).sigmoid()   # (batch_size, seq_len, seq_len, hidden_dim)
        right_gate = self.right_gate(x).sigmoid() # (batch_size, seq_len, seq_len, hidden_dim)
        out_gate = self.out_gate(x).sigmoid()     # (batch_size, seq_len, seq_len, hidden_dim)

        left = left * left_gate   # (batch_size, seq_len, seq_len, hidden_dim)
        right = right * right_gate # (batch_size, seq_len, seq_len, hidden_dim)

        out = einsum(self.mix_einsum_eq, left, right)  # (batch_size, seq_len, seq_len, hidden_dim)

        out = self.to_out_norm(out)  # (batch_size, seq_len, seq_len, hidden_dim)
        out = out * out_gate         # (batch_size, seq_len, seq_len, hidden_dim)
        return self.to_out(out)      # (batch_size, seq_len, seq_len, dim)



class TriangleAttention(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,
        dim: int = 32,
        n_heads: int = 4,
        wise: Literal['row', 'col'] = 'row'
    ):
        """
        Implements Triangle Attention Mechanism.
        :param in_dim: Input feature dimension.
        :param dim: Dimension of query, key, and value per head.
        :param n_heads: Number of attention heads.
        :param wise: Whether to apply row-wise or column-wise attention.
        """
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TriangleAttention.
        :param z: Input tensor of shape (B, I, J, in_dim).
        :param src_mask: Source mask of shape (B, I, J).
        :return: Output tensor of shape (B, I, J, in_dim).
        """
        # Spawn pair mask
        src_mask = src_mask.clone()
        src_mask[src_mask == 0] = -1
        src_mask = src_mask.unsqueeze(-1).float()  # (B, I, J, 1)
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))  # (B, I, J, I)

        wise = self.wise
        z = self.norm(z)  # (B, I, J, in_dim)

        # Compute bias and gate
        gate = self.to_gate(z)  # [1] (B, I, J, in_dim)
        b = self.linear_for_pair(z)  # [5] (B, I, J, n_heads) 

        # Compute Q, K, V
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)  # [2], [3], [4]: each (B, I, J, n_heads * dim)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))  
        # Each: (B, I, J, n_heads, dim)
        scale = q.size(-1) ** 0.5  # Scalar

        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)  # (B, 1, I, J, n_heads)
            softmax_dim = 3
            attn_mask = rearrange(attn_mask, 'b i j->b 1 i j 1')  # (B, 1, I, J, 1)
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)  # (B, I, J, 1, n_heads)
            softmax_dim = 2
            attn_mask = rearrange(attn_mask, 'b i j->b i j 1 1')  # (B, I, J, 1, 1)
        else:
            raise ValueError('wise should be col or row!')

        # Compute attention logits
        logits = (torch.einsum(eq_attn, q, k) / scale + b)  # [6], [7] (B, I, J, I, n_heads) or (B, I, J, J, n_heads)
        logits = logits.masked_fill(attn_mask == -1, float('-1e-9'))  # Apply mask

        # Compute attention weights
        attn = logits.softmax(softmax_dim)  # [8] (B, I, J, I, n_heads) or (B, I, J, J, n_heads)

        # Compute attention output
        out = torch.einsum(eq_multi, attn, v)  # [9] (B, I, J, n_heads, dim)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)')  # [10] (B, I, J, in_dim)

        # Final projection
        z_ = self.to_out(out)  # (B, I, J, in_dim)

        return z_



class ConvTransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder Layer with convolutional enhancements and pairwise feature processing.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        pairwise_dimension: int,
        use_triangular_attention: bool,
        dropout: float = 0.1,
        k: int = 3,
    ):
        """
        :param d_model: Dimension of the input embeddings
        :param nhead: Number of attention heads
        :param dim_feedforward: Hidden layer size in feedforward network
        :param pairwise_dimension: Dimension of pairwise features
        :param use_triangular_attention: Whether to use triangular attention modules
        :param dropout: Dropout rate
        :param k: Kernel size for the 1D convolution
        """
        super(ConvTransformerEncoderLayer, self).__init__()

        # === Attention Layers ===
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # === Layer Norms ===
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # === Dropout Layers ===
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv = nn.Conv1d(d_model, d_model, k, padding=k // 2)

        self.triangle_update_out = TriangleMultiplicativeModule(dim=pairwise_dimension, mix='outgoing')
        self.triangle_update_in = TriangleMultiplicativeModule(dim=pairwise_dimension, mix='ingoing')

        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)

        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension,
                dim=pairwise_dimension // 4,
                wise='row'
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension,
                dim=pairwise_dimension // 4,
                wise='col'
            )

            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)

        self.OuterProductMean = OuterProductMean(in_dim=d_model, pairwise_dim=pairwise_dimension)

        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension)
        )

    def forward(
        self,
        src: torch.Tensor,  # Shape: (batch_size, seq_len, d_model)
        pairwise_features: torch.Tensor,  # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)
        src_mask: torch.Tensor = None,  # Shape: (batch_size, seq_len) or None
        return_aw: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the ConvTransformerEncoderLayer.

        :param src: Input tensor of shape (batch_size, seq_len, d_model)
        :param pairwise_features: Pairwise feature tensor of shape (batch_size, seq_len, seq_len, pairwise_dimension)
        :param src_mask: Optional mask tensor of shape (batch_size, seq_len)
        :param return_aw: Whether to return attention weights
        :return: Tuple containing processed src and pairwise_features (and optionally attention weights)
        """
        src = src * src_mask.float().unsqueeze(-1)  # Shape: (batch_size, seq_len, d_model)
        res = src  # residual
        src = src + self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)  # Shape: (batch_size, seq_len, d_model)
        src = self.norm3(src)

        pairwise_bias = self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0, 3, 1, 2)
        src2, attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)  # Shape: (batch_size, seq_len, d_model)
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # Shape: (batch_size, seq_len, d_model)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.OuterProductMean(src)  # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)
        pairwise_features = pairwise_features + self.pair_dropout_out(self.triangle_update_out(pairwise_features, src_mask))
        pairwise_features = pairwise_features + self.pair_dropout_in(self.triangle_update_in(pairwise_features, src_mask))
        
        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(self.triangle_attention_out(pairwise_features, src_mask))
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(self.triangle_attention_in(pairwise_features, src_mask))
        
        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)  # Shape: (batch_size, seq_len, seq_len, pairwise_dimension)

        if return_aw:
            return src, pairwise_features, attention_weights  # Shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len, pairwise_dimension), (batch_size, nhead, seq_len, seq_len)
        else:
            return src, pairwise_features  # Shapes: (batch_size, seq_len, d_model), (batch_size, seq_len, seq_len, pairwise_dimension)



class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encoding for sequence-based models.
    :param dim: (int) The output embedding dimension. Default is 64.
    """
    
    def __init__(self, dim: int = 64):
        super(RelativePositionalEncoding, self).__init__()
        self.linear = nn.Linear(17, dim)  # (17,) -> (dim,)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Computes the relative positional encodings for a given sequence.

        :param src: Input tensor of shape (B, L, D), where:
            - B: Batch size
            - L: Sequence length
            - D: Feature dimension (ignored in this module)
        :return: Relative positional encoding of shape (L, L, dim)
        """
        L = src.shape[1]  # Sequence length
        res_id = torch.arange(L, device=src.device).unsqueeze(0)  # (1, L)
        
        device = res_id.device
        bin_values = torch.arange(-8, 9, device=device)  # (17,)

        d = res_id[:, :, None] - res_id[:, None, :]  # (1, L, L)
        bdy = torch.tensor(8, device=device)

        # Clipping the values within the range [-8, 8]
        d = torch.minimum(torch.maximum(-bdy, d), bdy)  # (1, L, L)

        # One-hot encoding of relative positions
        d_onehot = (d[..., None] == bin_values).float()  # (1, L, L, 17)

        assert d_onehot.sum(dim=-1).min() == 1  # Ensure proper one-hot encoding

        # Linear transformation to embedding space
        p = self.linear(d_onehot)  # (1, L, L, 17) -> (1, L, L, dim)

        return p.squeeze(0)  # (L, L, dim)



import torch
import torch.nn as nn

class RibonanzaNet(nn.Module):
    """
    A transformer-based neural network for sequence processing, incorporating convolutional transformer encoder layers,
    outer product mean operations, and relative positional encoding.
    """
    def __init__(self, config: object):
        """
        Initializes the RibonanzaNet model.
        
        :param config: Configuration object containing model hyperparameters.
            - ninp (int): Input embedding dimension.
            - ntoken (int): Vocabulary size for embedding layer.
            - nclass (int): Number of output classes.
            - nhead (int): Number of attention heads.
            - nlayers (int): Number of transformer encoder layers.
            - dropout (float): Dropout probability.
            - pairwise_dimension (int): Dimension of pairwise features.
            - use_triangular_attention (bool): Whether to use triangular attention.
            - use_bpp (bool): Whether to use base-pairing probability features.
            - k (int): Kernel size for convolutions in transformer layers.
        """
        super(RibonanzaNet, self).__init__()
        self.config = config
        nhid = config.ninp * 4
        
        self.transformer_encoder = []
        print(f"Constructing {config.nlayers} ConvTransformerEncoderLayers")
        for i in range(config.nlayers):
            k = config.k if i != config.nlayers - 1 else 1
            self.transformer_encoder.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp, nhead=config.nhead,
                    dim_feedforward=nhid,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dropout=config.dropout, k=k)
            )
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)
        
        if config.use_bpp:
            self.mask_dense = nn.Conv2d(2, config.nhead // 4, 1)
        else:
            self.mask_dense = nn.Conv2d(1, config.nhead // 4, 1)
        
        self.OuterProductMean = OuterProductMean(in_dim=config.ninp, pairwise_dim=config.pairwise_dimension)
        self.pos_encoder = RelativePositionalEncoding(config.pairwise_dimension)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, return_aw: bool = False):
        """
        Forward pass of the RibonanzaNet model.
        
        :param src: Input tensor of shape (B, L), where B is the batch size and L is the sequence length.
        :param src_mask: Optional mask tensor of shape (B, L, L), used for attention masking.
        :param return_aw: Boolean flag indicating whether to return attention weights.
        :return: Output tensor of shape (B, L, nclass) if return_aw is False, or a tuple (output, attention_weights).
        """
        B, L = src.shape  # (Batch size, Sequence length)
        src = self.encoder(src).reshape(B, L, -1)  # (B, L, ninp)
        
        pairwise_features = self.OuterProductMean(src)  # (B, L, L, pairwise_dimension)
        pairwise_features = pairwise_features + self.pos_encoder(src)  # (B, L, L, pairwise_dimension)
        
        attention_weights = []
        for i, layer in enumerate(self.transformer_encoder):
            if src_mask is not None:
                if return_aw:
                    src, aw = layer(src, pairwise_features, src_mask, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(src, pairwise_features, src_mask, return_aw=return_aw)
            else:
                if return_aw:
                    src, aw = layer(src, pairwise_features, return_aw=return_aw)
                    attention_weights.append(aw)
                else:
                    src, pairwise_features = layer(src, pairwise_features, return_aw=return_aw)
        
        output = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0  # (B, L, nclass)
        
        if return_aw:
            return output, attention_weights
        else:
            return output



config = load_config_from_yaml("config.yaml")
model = RibonanzaNet(config).cuda()
x = torch.ones(4, 128).long().cuda()
mask = torch.ones(4, 128).long().cuda()
mask[:,120:] = 0
print(f"Output shape: {model(x,src_mask=mask).shape}"), sep()
model




