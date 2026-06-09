import os 
import pandas as pd
import numpy as np
import pickle


#submission notebook doesn't have internet access, so I simply copy pasted code: https://huggingface.co/bird-of-paradise/deepseek-mla
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import math



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate input dimensions
    assert xq.shape[-1] == xk.shape[-1], "Query and Key must have same embedding dimension"
    assert xq.shape[-1] % 2 == 0, "Embedding dimension must be even"

    # Get sequence lengths
    q_len = xq.shape[1]
    k_len = xk.shape[1]
    
    # Use appropriate part of freqs_cis for each sequence
    q_freqs = freqs_cis[:q_len]
    k_freqs = freqs_cis[:k_len]
    
    # Apply rotary embeddings separately
    # split last dimention to [xq.shape[:-1]/2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
 
    # Reshape freqs for each
    q_freqs = reshape_for_broadcast(q_freqs, xq_)
    k_freqs = reshape_for_broadcast(k_freqs, xk_)
    
    # Works for both [bsz, seqlen, n_heads*head_dim] and [bsz, seqlen, n_heads, head_dim]
    xq_out = torch.view_as_real(xq_ * q_freqs).flatten(xq.ndim-1) 
    xk_out = torch.view_as_real(xk_ * k_freqs).flatten(xk.ndim-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)




class MultiHeadLatentAttention(nn.Module):
    """
        Multi-Head Latent Attention(MLA) Module As in DeepSeek_V2 pape
        Key innovation from standard MHA:
             1. Low-Rank Key-Value Joint Compression 
             2. Decoupled Rotary Position Embedding
             
    Args:
        d_model:  Total dimension of the model.
        num_head: Number of attention heads.
        d_embed:  Embedding dimension
        d_c:      K/V compression dimension
        d_c1:     Q compression dimension
        d_rotate: Dimension for Rotary Position Embedding
        dropout:  Dropout rate for attention scores.
        bias:     Whether to include bias in linear projections.
        d_head:   Inferred from d_model//num_head
    Inputs:
        sequence: input sequence for self-attention and the query for cross-attention
        key_value_state: input for the key, values for cross-attention
    """
    def __init__(
        self, 
        d_model,             # Infer d_head from d_model
        num_head, 
        d_embed, 
        d_c, 
        d_c1, 
        d_rotate, 
        dropout=0.1, 
        bias=True,
        max_batch_size=32,   # For KV cache sizing
        max_seq_len=2048     # For KV cache sizing 
        ):
        super().__init__()
        
        assert d_model % num_head == 0, "d_model must be divisible by num_head"
        assert d_c < d_embed, "Compression dim should be smaller than embedding dim"
        assert d_c1 < d_embed, "Query compression dim should be smaller than embedding dim"
        
        self.d_model = d_model
        self.num_head = num_head
        # Verify dimensions match up
        assert d_model % num_head == 0, f"d_model ({d_model}) must be divisible by num_head ({num_head})"
        self.d_head=d_model//num_head
        self.d_embed = d_embed
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate
        self.dropout_rate = dropout  # Store dropout rate separately

        # Linear down-projection(compression) transformations
        self.DKV_proj = nn.Linear(d_embed, d_c, bias=bias)
        self.DQ_proj = nn.Linear(d_embed, d_c1, bias=bias)
        
        # linear up-projection transformations
        self.UQ_proj = nn.Linear(d_c1, d_model, bias=bias)
        self.UK_proj = nn.Linear(d_c, d_model, bias=bias)
        self.UV_proj = nn.Linear(d_c, d_model, bias=bias)

        # Linear RoPE-projection
        self.RQ_proj = nn.Linear(d_c1, num_head*d_rotate, bias=bias)
        self.RK_proj = nn.Linear(d_embed, d_rotate, bias=bias)
        
        # linear output transformations
        self.output_proj = nn.Linear( d_model, d_model, bias=bias)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Initiialize scaler
        self.scaler = float(1.0 / math.sqrt(self.d_head + d_rotate)) # Store as float in initialization

        # Initialize C_KV and R_K cache for inference
        self.cache_kv = torch.zeros(
            (max_batch_size, max_seq_len, d_c)
        )
        self.cache_rk = torch.zeros(
            (max_batch_size, max_seq_len, d_rotate)
        )

        # Initialize freqs_cis for RoPE
        self.freqs_cis = precompute_freqs_cis(
            d_rotate, max_seq_len * 2
        )
    

    def forward(
        self, 
        sequence, 
        key_value_states = None, 
        att_mask=None,
        use_cache=False,
        start_pos: int = 0
    ):

        """
        Forward pass supporting both standard attention and cached inference
        Input shape: [batch_size, seq_len, d_model=num_head * d_head]
        Args:
            sequence: Input sequence [batch_size, seq_len, d_model]
            key_value_states: Optional states for cross-attention
            att_mask: Optional attention mask
            use_cache: Whether to use KV caching (for inference)
            start_pos: Position in sequence when using KV cache
        """
        batch_size, seq_len, model_dim = sequence.size()
        # prepare for RoPE
        self.freqs_cis = self.freqs_cis.to(sequence.device)
        freqs_cis = self.freqs_cis[start_pos : ]

        # Check only critical input dimensions
        assert model_dim == self.d_model, f"Input dimension {model_dim} doesn't match model dimension {self.d_model}"
        if key_value_states is not None:
            assert key_value_states.size(-1) == self.d_model, \
            f"Cross attention key/value dimension {key_value_states.size(-1)} doesn't match model dimension {self.d_model}"

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # Determine kv_seq_len early
        kv_seq_len = key_value_states.size(1) if is_cross_attention else seq_len
        
        # Linear projections and reshape for multi-head, in the order of Q, K/V
        # Down and up projection for query
        C_Q = self.DQ_proj(sequence)     #[batch_size, seq_len, d_c1]
        Q_state = self.UQ_proj(C_Q)      #[batch_size, seq_len, d_model]
        # Linear projection for query RoPE pathway
        Q_rotate = self.RQ_proj(C_Q)      #[batch_size, seq_len, num_head*d_rotate]


        if use_cache:
            #Equation (41) in DeepSeek-v2 paper: cache c^{KV}_t
            self.cache_kv = self.cache_kv.to(sequence.device)

            # Get current compressed KV states
            current_kv = self.DKV_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, d_c]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_kv[:batch_size, start_pos:start_pos + kv_seq_len] = current_kv
            # Use cached compressed KV up to current position
            C_KV = self.cache_kv[:batch_size, :start_pos + kv_seq_len]

            #Equation (43) in DeepSeek-v2 paper: cache the RoPE pathwway for shared key k^R_t
            assert self.cache_rk.size(-1) == self.d_rotate, "RoPE cache dimension mismatch"
            self.cache_rk = self.cache_rk.to(sequence.device)
            # Get current RoPE key
            current_K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, d_rotate]
            # Update cache using kv_seq_len instead of seq_len
            self.cache_rk[:batch_size, start_pos:start_pos + kv_seq_len] = current_K_rotate
            # Use cached RoPE key up to current position
            K_rotate = self.cache_rk[:batch_size, :start_pos + kv_seq_len] #[batch_size, cached_len, d_rotate]
            
            
            """handling attention mask"""
            if att_mask is not None:
                # Get the original mask shape
                mask_size = att_mask.size(-1)
                cached_len = start_pos + kv_seq_len        # cached key_len, including previous key
                assert C_KV.size(1) == cached_len, \
            f"Cached key/value length {C_KV.size(1)} doesn't match theoretical length {cached_len}"
                
                # Create new mask matching attention matrix shape
                extended_mask = torch.zeros(
                    (batch_size, 1, seq_len, cached_len),  # [batch, head, query_len, key_len]
                    device=att_mask.device,
                    dtype=att_mask.dtype
                )
                
                # Fill in the mask appropriately - we need to be careful about the causality here
                # For each query position, it should only attend to cached positions up to that point
                for i in range(seq_len):
                    extended_mask[:, :, i, :(start_pos + i + 1)] = 0  # Can attend
                    extended_mask[:, :, i, (start_pos + i + 1):] = float('-inf')  # Cannot attend
                    
                att_mask = extended_mask
        else:
            # Compression projection for C_KV
            C_KV = self.DKV_proj(key_value_states if is_cross_attention else sequence) #[batch_size, kv_seq_len, d_c]\
            # RoPE pathway for *shared* key
            K_rotate = self.RK_proj(key_value_states if is_cross_attention else sequence)
            

        # Up projection for key and value
        K_state = self.UK_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, d_model]
        V_state = self.UV_proj(C_KV)               #[batch_size, kv_seq_len/cached_len, d_model]

        
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head)

        # After getting K_state from projection, get its actual sequence length
        actual_kv_len = K_state.size(1)    # kv_seq_len or start_pos + kv_seq_len
        # in cross-attention, key/value sequence length might be different from query sequence length
        # Use actual_kv_len instead of kv_seq_len for reshaping
        K_state = K_state.view(batch_size, actual_kv_len, self.num_head, self.d_head) 
        V_state = V_state.view(batch_size, actual_kv_len, self.num_head, self.d_head)


        #Apply RoPE to query and shared key
        Q_rotate = Q_rotate.view(batch_size, seq_len, self.num_head, self.d_rotate)
        K_rotate = K_rotate.unsqueeze(2).expand(-1, -1, self.num_head, -1)  # [batch, cached_len, num_head, d_rotate]
        Q_rotate, K_rotate = apply_rotary_emb(Q_rotate, K_rotate, freqs_cis=freqs_cis)


        # Concatenate along head dimension
        Q_state = torch.cat([Q_state, Q_rotate], dim=-1)  # [batch_size, seq_len, num_head, d_head + d_rotate]
        K_state = torch.cat([K_state, K_rotate], dim=-1)  # [batch_size, actual_kv_len, num_head, d_head + d_rotate]


        # Scale Q by 1/sqrt(d_k)
        Q_state = Q_state * self.scaler
        Q_state = Q_state.transpose(1, 2)  # [batch_size, num_head, seq_len, head_dim]
        K_state = K_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]
        V_state = V_state.transpose(1, 2)  # [batch_size, num_head, actual_kv_len, head_dim]

    
        # Compute attention matrix: QK^T
        self.att_matrix = torch.matmul(Q_state, K_state.transpose(-1,-2)) 
    
        # apply attention mask to attention matrix
        if att_mask is not None and not isinstance(att_mask, torch.Tensor):
            raise TypeError("att_mask must be a torch.Tensor")

        if att_mask is not None:
            self.att_matrix = self.att_matrix + att_mask
        
        # apply softmax to the last dimension to get the attention score: softmax(QK^T)
        att_score = F.softmax(self.att_matrix, dim = -1)
    
        # apply drop out to attention score
        att_score = self.dropout(att_score)
    
        # get final output: softmax(QK^T)V
        att_output = torch.matmul(att_score, V_state)
        assert att_output.size(0) == batch_size, "Batch size mismatch"
        assert att_output.size(2) == seq_len, "Output sequence length should match query sequence length"
        
            
        # concatinate all attention heads
        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_head*self.d_head) 


        # final linear transformation to the concatenated output
        att_output = self.output_proj(att_output)

        assert att_output.size() == (batch_size, seq_len, self.d_model), \
        f"Final output shape {att_output.size()} incorrect"

        return att_output


#Well, I think it's a good practice nowadays, to explicitly write, where LLMs were used
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

@dataclass
class AudioConfig:
    """
    Central configuration for Audio Processing.
    """
    sr: int = 32000
    n_fft: int = 2048
    win_length: int = n_fft
    hop_length: int = win_length // 2
    n_mels: int = 256
    top_db: int = 80
    
    # Slicing/Striping params
    stripe_width: int = sr // (n_fft // 4) #~1 secnod
    stripe_overlap: int = stripe_width // 5 #~0.2 seconds
    
    # Helper to inject external slicing logic
    slicing_func: Any = None

    prior_cut_sec: int = 5


import torch
import torchaudio.transforms as transforms
import torchaudio
#import librosa
from joblib import Parallel, delayed
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BirdDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_df,
        label_encoder,
        config: AudioConfig,
        group_mode:bool = True,
        return_id: bool = True,
        prediction_as_target: bool = False,
        root_dir:str = '/kaggle/input/birdclef-2025/train_audio',
        multitarget:bool = True,
    ):
        super().__init__()
        self.cfg = config
        #sklearn labelencoder
        self.label_encoder = label_encoder
        #files root dir
        self.root_dir = root_dir
        #flag to retrieve whole sequnces instead of just tokens
        self.group_mode = group_mode
        #flag to retrieve filenames to sequence of tensors
        self.return_id = return_id
        #includes the secondary labels as primary
        self.multitarget = multitarget
        #flag to retrive pseudo labels
        self.prediction_as_target = prediction_as_target
        #will serve as pseudo labels storage later
        self.predictions = None
        
        #creating spectograms and preparing labels
        results_with_index = self._parallel_prepare(meta_df)
        
        #extract spectograms and labels
        self.spectrs, self.labels, self.idx = zip(*results_with_index)

        if not group_mode:
            #few row down we do item assignment
            self.labels = list(self.labels)
            
            #expand labels with respect to token_num in each sample
            for i, tokenized_tensor in enumerate(self.spectrs):
                expand_size = tokenized_tensor.shape[-1]
                self.labels[i] = np.tile(self.labels[i], (expand_size,1))
                
            #concatenate all tensor into 1 huge brick, because we doesn't care about their relations
            self.spectrs = torch.cat(self.spectrs, dim = -1)
            self.labels = [arr for sublist in self.labels for arr in sublist]

        #label tensor init with n_classes
        label_tensor = torch.zeros(len(self.labels), len(self.label_encoder.classes_)) #B, L
        #ohe assigning 
        #todo it seems very slow, but I don't know better approach for multi label ohe assigning
        #because once again, self.labels isn't a size=1 array
        for i, label in enumerate(self.labels):
            label_tensor[i, label] = 1

        self.labels = label_tensor
        self.idx = np.array(self.idx, dtype = object)
        
        self.token_h, self.token_w =  config.n_mels, config.stripe_width
    
    def preprocess_label_file(
        self, 
        file_path, 
        label, 
        secondary_labels, 
        root_dir = None,
    ):
        """
        Reads audio -> STFT -> Mel/Mel**2/Linear Features -> Stacks them.
        Reads id(str) -> id from label_encoder
        """
        if root_dir is None:
            root_dir = self.root_dir
        path = os.path.join(root_dir, file_path)
        waveform, sr = torchaudio.load(path, num_frames = self.cfg.sr * self.cfg.prior_cut_sec)

        if sr != self.cfg.sr:
            waveform = transforms.Resample(sr, self.cfg.sr)(waveform)
            
        waveform = waveform[0]
        
        label_id = self.label_encoder.transform([label])

        #secondary label processing
        #str -> list
        secondary_labels = ast.literal_eval(secondary_labels)
        #if secondary labels actually exist:
        if (secondary_labels != ['']) and self.multitarget:
            secondary_labels = self.label_encoder.transform(secondary_labels)
            label_id = np.concatenate((label_id, secondary_labels))
        
        slices = self.cfg.slicing_func( #[C, H, W] -> [C, H, W, T]
                    ar = waveform.unsqueeze(0).unsqueeze(0),
                    stripe = self.cfg.stripe_width * self.cfg.hop_length, 
                    overlap = self.cfg.stripe_overlap * self.cfg.hop_length,
                    pad_value = 0,
                )
      
        return slices, label_id, file_path

    def _parallel_prepare(self, df):
        """
        parallel file processing using joblib
        """

        columns_of_interest = ['filename', 'primary_label', 'secondary_labels']        
        
        results_with_index = Parallel(n_jobs=-1)(
                    delayed(self.preprocess_label_file)(
                        file_path = row.filename, 
                        label = row.primary_label,
                        secondary_labels = row.secondary_labels,
                    ) 
                    for _, row in df[columns_of_interest].iterrows()
                )

        return results_with_index

    def update_pseudo_labels(self, obj:pd.Series):
        """
        appends new pseudolabels into existing pseudolabels storage
        """
        if self.predictions is None:
            self.predictions = pd.Series()

        self.predictions = pd.concat([self.predictions, obj])
        
    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, index):
        """
        returns batch with ohe target
        """
        if self.group_mode:
            #it is the only option to fix dataloader select random samples via list problem
            #we can't select tuple[list], so instead we have to itterate
            #yes, it's the ONLY PLACE, we can't even use collate_fn
            if isinstance(index, list) or isinstance(index, np.ndarray):
                batch_tensor = [self.spectrs[i] for i in index]
            else:
                batch_tensor = self.spectrs[index]

        else:
            #because in not group mode we iterate through tokens and they are in the last dim
            batch_tensor = self.spectrs[..., index]

        str_id = self.idx[index]
        #pseudo labels
        if self.prediction_as_target:
            label_tensor = self.predictions[str_id].values
        else:
            label_tensor = self.labels[index]

        #in predict mode, I also want to retrieve a file name
        if self.return_id:
            return batch_tensor, label_tensor, str_id
        else:
            return batch_tensor, label_tensor


class OnlineDataset(torch.utils.data.Dataset):
    def __init__(        
        self,
        meta_df,
        label_encoder,
        config: AudioConfig,
        group_mode:bool = True,
        return_id: bool = True,
        prediction_as_target: bool = False,
        root_dir:str = '/kaggle/input/birdclef-2025/train_audio/',
        multitarget:bool = True,
    ):
        super().__init__()
        self.cfg = config
        #sklearn labelencoder
        self.label_encoder = label_encoder
        #files root dir
        self.root_dir = root_dir
        #flag to retrieve whole sequnces instead of just tokens
        self.group_mode = group_mode
        #flag to retrieve filenames to sequence of tensors
        self.return_id = return_id
        #includes the secondary labels as primary
        self.multitarget = multitarget
        #flag to retrive pseudo labels
        self.prediction_as_target = prediction_as_target
        
        #df that contains info about each file
        self.meta_df = meta_df
        self.predictions = None 

    def update_pseudo_labels(self, obj:pd.Series):
        """
        appends new pseudolabels into existing pseudolabels storage
        """
        if self.predictions is None:
            self.predictions = pd.Series()

        self.predictions = pd.concat([self.predictions, obj])

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, index):
        temp_ds = BirdDataset(
            meta_df = self.meta_df.iloc[index],
            label_encoder = self.label_encoder,
            config = self.cfg,
            group_mode = self.group_mode,
            return_id = self.return_id,
            prediction_as_target = self.prediction_as_target,
            root_dir = self.root_dir,
            multitarget = self.multitarget,
        )
        if self.prediction_as_target:
            temp_ds.predictions = self.predictions[index]
            
        return temp_ds[:]
        


#AI generated (not slope)
from torch.utils.data import Sampler

class IndexBatchSampler(Sampler):
    def __init__(self, data_source, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Yields a list of indices at each iteration instead of a single index.
        
        Args:
            data_source: The dataset (used to determine length).
            batch_size: Number of indices to yield per iteration.
            shuffle: Whether to shuffle indices before batching.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        # Create a list of all indices
        indices = list(range(len(self.data_source)))
        
        if self.shuffle:
            np.random.shuffle(indices)
            
        # Yield chunks (batches) of indices
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Handle the remaining items
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Calculate how many batches this sampler will produce
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


from torch import nn
import timm 
class Encoder(nn.Module):
    def __init__(self, backbone_name:str, original_weights: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=original_weights, 
            num_classes=0,
        )
        #init data-transform functions
        self.stft_transform = None
        self.db_transform = None
        self.mel_transform = None
        self.pooler = None

    def set_audiopreprocessing(self, cfg):
        self.stft_transform = transforms.Spectrogram(
            n_fft=cfg.n_fft, 
            hop_length=cfg.hop_length,
            power=1.0,
        )
        self.db_transform = transforms.AmplitudeToDB(
            stype="magnitude", 
            top_db=cfg.top_db
        )
        self.mel_transform = transforms.MelScale(
            sample_rate=cfg.sr,
            n_stft=cfg.n_fft // 2 + 1,
            n_mels=cfg.n_mels,
        )
        self.pooler = nn.AdaptiveMaxPool1d(cfg.n_mels)
        
    def stripe_w_overlap(
        self, 
        ar:torch.Tensor, 
        stripe:int, 
        overlap:int,
        pad_value:float,
    ):
        """
        splits tensor into overlaping chunks along time dim
        the last token is dropped
        if the time < token_width => pad with constant value of pad_value
        returns (B(depends on input), C, H, W, tokens)
        """
        step = stripe - overlap
        time_len = ar.shape[-1] #time

        #if time < stripe => pad with zeros, because we can't stack None values
        if time_len < stripe:
            ar = torch.nn.functional.pad(
                input = ar,
                pad = (0, stripe - time_len), 
                mode = 'constant', 
                value = pad_value
                
            )
            num_steps = 1
        #if time < stripe+step means that there is possible only 1 step 
        elif time_len < stripe+step:
            num_steps = 1
        #number of full chunks
        else:
            num_steps = (time_len - stripe) // step

        try:
            striped_tensor = torch.stack(
                [
                    ar[:,..., i * step : i * step + stripe] 
                     for i in range(num_steps)
                ]
            ) #(T, ...)
        except: 
            raise RuntimeError(f"error in stripes. here is the tensor shape: {ar.shape} ; the stripe: {stripe} ; and overlap: {overlap} ; and steps: {num_steps} ")

        striped_tensor = torch.moveaxis(striped_tensor, 0, -1).contiguous() # (T, ...) -> (..., T)
        
        return striped_tensor
        
    def forward(self, x, **kwargs):
        #(B*T, C, H, W) -> (B*T, C_backbone)
        
        #==prerpcoessing raw waves into images
        #(B*T, C, H, W) -> (B, C_spects, C_mel, W_stft)
        
        stft = self.stft_transform(x)
        stft = stft[..., :320] #todo idk why, but I started to receive 321 windows after moving stiping on raw waves
        #conver to dB spectrs
        mel_out_stft = self.db_transform(self.mel_transform(stft))
        mel_out_stft_2 = self.db_transform(self.mel_transform(stft**2))
        stft = self.db_transform(stft)
        
        #normalize stft to mel size
        mega_B, H, W, C_stft, W_stft = stft.shape
        stft = torch.transpose(stft, -1, -2) # mega_B, H, W, C_stft, W_stft -> mega_B, H, W, W_stft, C_stft

        #flatten B, H, W, T into a single *B* dimension 
        stft = stft.reshape(-1, 1, C_stft) #(B,H,W,W_s,C) -> (*B*, 1, C)
        pooled_lin = self.pooler(stft) # (*B*, 1, C_stft) -> (*B*, 1, C_mel) 
        pooled_lin = pooled_lin.reshape(mega_B, H, W, W_stft, -1) #(mega_B, H, W, W_stft, C_mel)
        
        x = torch.stack((mel_out_stft, mel_out_stft_2, pooled_lin), dim = 1) #B, C_spects, H, W, C_mel, W_stft 
        x = x.squeeze(2).squeeze(2) #B, C_spects, C_mel, W_stft 
        #==prerpcoessing raw waves into images
        
        return self.backbone(x) #(B*T, C, H, W) -> (B*T, C)

class Classifier(nn.Module):
    def __init__(
        self, 
        seq_mode = True, #to process instance wise or sequence wise
        single_head:nn.Module = None,
        multi_head:nn.Module = None,
        single_activation:nn.Module = None,
        multi_activation:nn.Module = None,

    ):
        super().__init__()
        self.seq_mode = seq_mode
        self.single_target_model = single_head
        self.multi_target_model = multi_head
        #activation fn in this class instead of token encoder, if I want to thrain only 1 SED head
        self.single_activation = single_activation
        self.multi_activation = multi_activation
    
    def forward(
            self, 
            x, 
            multitarget_mask = None,
            attention_mask:torch.Tensor = None,
            return_NoF:bool = False,
            pool:bool = True,
        ):
        """
        Pass padded vision embeddings in classifier with implementing additional logic:
        * multi target model split(multitarget_mask)
        * removes NoF token (return_NoF = False)
        * returns prediction token wise (pool = False)

        the classifier object's attribute seq_mode determines, whether to return list of token-wise predictions
        or padded tensor of token-wise predictions
        Args:
            x: features from vision encoder.
            multitarget_mask: list of booleans, where True means to use multitarget model.
            return_NoF: whether to return NoF token.
            pool: wheter to return output sequence wise or token wise.
        """

        #B, T_max, Channels -> B, empty/T_max, Class_digit+NoF
        #we have rotary embeddings, so we can't pass empty batches anymore
        if torch.all(multitarget_mask) == True:
            output = self.multi_target_model(x, return_NoF, attention_mask, pool)
            output = self.multi_activation(output)
        elif torch.any(multitarget_mask) == True:
            multi_output = self.multi_target_model(x[multitarget_mask], return_NoF, attention_mask[multitarget_mask], pool)
            multi_output = self.multi_activation(multi_output)
            single_output = self.single_target_model(x[~multitarget_mask], return_NoF, attention_mask[~multitarget_mask], pool)
            single_output = self.single_activation(single_output)
            output = torch.cat([multi_output, single_output], dim = 0)
        else:
            output = self.single_target_model(x, return_NoF, attention_mask, pool)
            output = self.single_activation(output)
    
        if not self.seq_mode:

            masked_output = []
            last_indx = torch.sum(attention_mask, dim = 1).int()
            
            #list of token predictions for each instance
            for i, out in enumerate(output):
                masked_output.append(out[:last_indx[i]].cpu().detach())

            output = masked_output
            
        return output


class CLEFModel(nn.Module):
    def __init__(
        self, 
        encoder = None, 
        classifier = None,
        padding_value = 0,
        return_NoF = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.padding_value = padding_value
        self.return_NoF = return_NoF

    def pad_list_to_tensor(self, tensor_list:list, padding_value:float = None):
        """
        takes a list of embeddings [(T_i, C), ...] and creates out of them 1 padded tensor
        """
        if padding_value is None:
            padding_value = self.padding_value
        #in batch maximum for padding (in tokens)
        max_len = np.max([s.shape[0] for s in tensor_list])
        
        batch_list = []
        attention_mask_list = []
        for tensor in tensor_list:
            #create attention mask, to prevent usage of padded values in the future
            attention_mask = torch.ones(tensor.shape[0])
            attention_mask = torch.nn.functional.pad(
                attention_mask, 
                (0, max_len - tensor.shape[0]), #token wise padding
                mode ='constant',
                value = 0
            )
            
            attention_mask_list.append(attention_mask)
            
            tensor = torch.nn.functional.pad(
                            tensor, 
                            (0,0,0, max_len - tensor.shape[0]), #token wise padding
                            mode ='constant', 
                            value = padding_value #equivavelnt of zero
                        )
        
            batch_list.append(tensor)
        
        batch_tensor = torch.stack(batch_list, dim=0) # B, max(T_b), C
        attention_mask = torch.stack(attention_mask_list, dim=0) # B, max(T_b)
        
        return batch_tensor, attention_mask

    def forward(
        self, 
        x, 
        multitarget_mask,
        pool,
    ):
        """
        Pass list of sequences into vision encoder and classifier models.
        
        seq-wise:
        Unioun list of tensor into huge one, to pass it through encoder,
        The received features are padded, to create once again 1 huge tensor for classifier
        """
        #list -> predict seq-wise
        if not isinstance(x, torch.Tensor):
            
            #huge tensor for feature encoding 
            x_feature_tensor = torch.cat(x, dim = -1).permute(-1, 0, 1, 2).to(device) # (T*B, C, H, W)
            try:
                x_feature_tensor = self.encoder(x_feature_tensor) #(T*B, C)
            except:
                raise ValueError(f"Most likely cuda memmory allocation, tensor shape: {x_feature_tensor.shape}")
            
            #split back into seqs 
            prev_len = 0
            embed = [] # (B, T_b, C)
            for sub_x in x:
                cur_len = sub_x.shape[-1]
                embed.append(x_feature_tensor[prev_len : prev_len+cur_len])# (T_b, C)
                prev_len = prev_len+cur_len
                
            #pad sequnces
            embed, attention_mask = self.pad_list_to_tensor(embed)
        else:        
            embed = self.encoder(x.to(device))
            attention_mask = torch.ones((embed.shape(0), embed.shape(1)))

        probs = self.classifier(
            x = embed, 
            attention_mask = attention_mask,
            multitarget_mask = multitarget_mask,
            return_NoF = self.return_NoF,
            pool = pool
        )
        return probs


class ETransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_head,
        d_embed,
        d_c,
        d_c1,
        d_rotate,
    ):
        super().__init__()
        self.rms_n = nn.RMSNorm(embedding_size)
        self.att = MultiHeadLatentAttention(
            d_model = embedding_size, #after attention embedding_size
            num_head = num_head, 
            d_embed = d_embed, #input to attention embedding_size
            d_c = d_c, 
            d_c1 = d_c1, 
            d_rotate = d_rotate, 
            dropout=0.1, 
            bias=True,
        )
        self.liner = nn.Linear(embedding_size, embedding_size)
        self.lin_act = nn.SiLU()

    def forward(self, x, att_mask):
        res = x
        x = self.rms_n(x)
        x = self.att(sequence = x, att_mask = att_mask)
        x += res
        res = x
        x = self.rms_n(x)
        x = self.liner(x)
        x = self.lin_act(x)
        x += res
        return x


class TokenEncoder(nn.Module):
    def __init__(
        self, 
        embedding_size,
        class_num,
        num_head,
        d_c,
        d_c1,
        d_rotate,
        pool_type:str = 'cls',
    ):
        super().__init__()
        #self.vembed2embed = nn.Linear(vembedding_size, embedding_size)
        self.output_size = class_num+1
        self.embedding_size = embedding_size
        self.num_head = num_head
        self.embed2embed = ETransformerBlock(
            self.embedding_size,
            num_head,
            self.embedding_size,
            d_c,
            d_c1,
            d_rotate,
        )
        self.embed2class = nn.Linear(embedding_size, self.output_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.pool_type = pool_type
        
    def forward(
        self,
        x,
        return_NoF,
        attention_mask:torch.Tensor = None,
        pool:bool = True,
    ):
        
        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1) #B, T, E -> B, T+1, E
        #todo attention mask expand
        attention_mask = torch.cat(
            (torch.ones(attention_mask.shape[0]).unsqueeze(-1), attention_mask), 
            dim = -1
        ) #B, T -> B, T+1 {1, 0}
        
        q_mask = attention_mask.unsqueeze(2) 
        k_mask = attention_mask.unsqueeze(1)
        attention_matrix = q_mask * k_mask # (B, T+1, T+1)
        attention_matrix -= 1 #{0 ,-1}
        attention_matrix *= 1e9 #{0, -inf}
        attention_matrix = attention_matrix.unsqueeze(1).expand(-1, self.num_head, -1, -1).to(device) #(B, H, T+1, T+1)
        output = self.embed2embed(x, att_mask = attention_matrix) #B, T+1, E -> B, T+1, E #todo probably bolleans to int in att_mask
        #output = x #for simplicity in test setup

        if pool:
            #or in model cls_token pool
            if self.pool_type == 'cls':
                output = output[:, 0, :] #(B, 1, E)
                
            #or token_avg pool
            elif self.pool_type == 'mean':
                
                output = output[:, 1:, :].to(device)
                attention_mask = attention_mask[:, 1:].to(device)
                last_indx = torch.sum(attention_mask, dim = 1).int().unsqueeze(-1).to(device) # (B, 1)
                
                #handle empty batch
                if output.shape[0] == 0:
                    output = torch.empty(0, self.embedding_size, dtype=output.dtype, device=output.device)
                else:
                    #because 'output' contains padded tokens (with zeros) calculated stats will also include them
                    #to play it safe, I want to multiply 'output' on 'att_mask'
                    output = (output * attention_mask.unsqueeze(-1)).sum(dim = 1) / last_indx #B, T, E -> B, E
            else:
                raise ValueError(f"pool_type {self.pool_type} isn't implemented yet")
        else:
            output = output[:, 1:, :]

        output = self.embed2class(output) #B, ?, E -> B, ?, C+1
    
        if not return_NoF:
            #last digit is NoF
            output = output[..., :-1]
                
        return output


w = 4
w_b = 32
np.array([[int(i*w**(1+i/w_b)), int((i+1)*w**(1+i/w_b))]for i in range(12)])


import os 
import pandas as pd
import numpy as np
import glob
#=======csv-serving

ref_df = pd.DataFrame()
directory = "/kaggle/input/birdclef-2025/test_soundscapes/"
filename = glob.glob(os.path.join(directory, "*ogg"))
if len(filename) == 0:
    print('no-test-dir')
    directory = "/kaggle/input/birdclef-2025/train_soundscapes"
    filename = glob.glob(os.path.join(directory, "*ogg"))[:3]
    
filename = [os.path.basename(path) for path in filename]
dummy_primaries = ['1139490'] * len(filename)
dummy_sec = ["['']"] * len(filename)
ref_df['filename'] = filename
ref_df['primary_label'] = dummy_primaries
ref_df['secondary_labels'] = dummy_sec
#=======csv-serving


#=======model loading
with open("/kaggle/input/cltest/other/default/9/label_encoder.pkl", 'rb') as f: #le path
    le = pickle.load(f)
    
encoder = Encoder('efficientnet_b3a', original_weights = False)
config = AudioConfig(
    n_fft = 400,
    win_length = 400,
    hop_length = 400 // 2,
    slicing_func=encoder.stripe_w_overlap,
    n_mels = 320,
    stripe_width = 320,
    stripe_overlap = 200,
    prior_cut_sec = 60,
)
encoder.set_audiopreprocessing(config)
single_target_head = TokenEncoder(
    embedding_size = encoder.backbone.num_features,
    class_num = len(le.classes_),
    num_head = 4,
    d_c = 256,
    d_c1 = 256,
    d_rotate = 16,
    pool_type = 'cls',
)
classifier = Classifier(
    seq_mode = True,
    single_head = single_target_head,
    #multi_head = multi_target_head, #todo
    multi_head = single_target_head, #todo same model is trained in different modes
    single_activation = nn.Softmax(dim = -1), #dim = -1, because tokenwise softmax 
    multi_activation= nn.Sigmoid(),
)
model = CLEFModel(
    encoder = encoder,
    classifier = classifier,
    padding_value = 0,
)
model.load_state_dict(torch.load('/kaggle/input/cltest/other/default/9/dmodel_354.pt', map_location=torch.device('cpu'))) #model path
model.eval()
#=======model loading

ref_dataset = BirdDataset(
    meta_df = ref_df[:2], 
    label_encoder = le,
    config = config,
    group_mode = True,
    return_id = True,
    root_dir = directory,
)

#=======saving csv

results_df = pd.DataFrame(columns = le.classes_)
for i in range(len(filename)):
    sub_slices, _, filename = ref_dataset.preprocess_label_file(
        ref_df['filename'].iloc[i], 
        '1139490', 
        "['']",
    )
    filename = filename.split('.')[0]

    #window tricks
    w = 4
    w_b = 32

    
    for i in range(12):
        new_ids = filename + f'_{(i+1)*5}'
        #list of booleans is multitarget
        #single boolean is a pool mode
        
        prediction = model(
            [sub_slices[..., int(i*w**(1+i/w_b)): int((i+1)*w**(1+i/w_b))]],
            torch.tensor([False]),
            True
        )
        temp_df = pd.DataFrame(
            data = prediction.detach().numpy(), 
            columns = le.classes_, 
            index = [new_ids],
        )
        results_df = pd.concat([results_df, temp_df])

results_df.to_csv('submission.csv', index_label = 'row_id')

