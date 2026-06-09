!pip install \
  "/kaggle/input/fix-wheel/pyg_260_fixed/torch_scatter-2.1.2+pt26cu124-cp310-cp310-linux_x86_64.whl" \
  "/kaggle/input/fix-wheel/pyg_260_fixed/torch_sparse-0.6.18+pt26cu124-cp310-cp310-linux_x86_64.whl" \
  "/kaggle/input/fix-wheel/pyg_260_fixed/torch_cluster-1.6.3+pt26cu124-cp310-cp310-linux_x86_64.whl" \
  "/kaggle/input/fix-wheel/pyg_260_fixed/torch_spline_conv-1.2.2+pt26cu124-cp310-cp310-linux_x86_64.whl" \
  "/kaggle/input/fix-wheel/pyg_260_fixed/torch_geometric-2.6.1-py3-none-any.whl"


import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pickle
import os
import sys


config = {
    "seed": 0,
    "cutoff_date": "2020-01-01",
    "test_cutoff_date": "2022-05-01",
    "max_len": 384,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "mixed_precision": "bf16",
    "epochs": 10,
    "cos_epoch": 5,
    "loss_power_scale": 1.0,
    "model_config_path": "../working/configs/pairwise.yaml",
    "max_cycles": 1,
    "grad_clip": 0.1,
    "gradient_accumulation_steps": 1,
    "d_clamp": 30,
    "max_len_filter": 9999999,
    "structural_violation_epoch": 50,
    "balance_weight": False,
}


test_data=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
test_data.head()


from torch.utils.data import Dataset, DataLoader

class RNADataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in (self.data.loc[idx,'sequence'])]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)




        return {'sequence':sequence}

test_dataset=RNADataset(test_data)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import math # math ëª¨ë“ˆ ì�„í�¬íŠ¸ í™•ì�¸
import torch # torch ì�„í�¬íŠ¸ í™•ì�¸
import torch.nn as nn
sys.path.append('/kaggle/input/ribonanzanet2/pytorch/alpha/1')
from Network import RibonanzaNet, MultiHeadAttention # MultiHeadAttention ì�„í�¬íŠ¸ ì¶”ê°€ (SimpleStructureModuleì—�ì„œ ì‚¬ìš©)
# from torch.utils import checkpoint # checkpoint ì�„í�¬íŠ¸ (ì›�ë³¸ forward ë©”ì†Œë“œì—�ì„œ ì‚¬ìš©)
# Script B ì›�ë³¸ì—� checkpoint ì�„í�¬íŠ¸ê°€ ëª…ì‹œì �ìœ¼ë¡œ ì—†ì—ˆìœ¼ë‚˜,
# ëª¨ë�¸ì�˜ forward ë©”ì†Œë“œì—�ì„œ ì‚¬ìš©í•˜ê³  ì�ˆìœ¼ë¯€ë¡œ í•„ìš”í•©ë‹ˆë‹¤.
# ë§Œì•½ ì�´ ìŠ¤í�¬ë¦½íŠ¸ì�˜ ë‹¤ë¥¸ ë¶€ë¶„ì—�ì„œ ì�´ë¯¸ import torch.utils.checkpoint as checkpoint í–ˆë‹¤ë©´ ì¤‘ë³µ ë¶ˆí•„ìš”
try:
    from torch.utils.checkpoint import checkpoint
except ImportError:
    # PyTorch < 1.11 ì—�ì„œëŠ” ì�´ë¦„ì�´ ë‹¤ë¥¼ ìˆ˜ ì�ˆìœ¼ë‚˜, ìµœì‹  ë²„ì „ ê¸°ì¤€
    print("Warning: torch.utils.checkpoint.checkpoint import failed. Ensure PyTorch version is compatible or import manually.")
    def checkpoint(fn, *args, **kwargs): # Dummy checkpoint for environments where it might be missing
        return fn(*args)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # emb = math.log(10000) / (half_dim - 1) # ì›�ë³¸ ì½”ë“œ
        # half_dimì�´ 1ì�¼ ê²½ìš° ZeroDivisionError ë°œìƒ� ê°€ëŠ¥. Script Aì²˜ëŸ¼ ìˆ˜ì •
        denominator = (half_dim - 1) if half_dim > 1 else 1.0
        emb = math.log(10000) / denominator
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, rnet_config, config, pretrained=False): # ì—¬ê¸°ì„œ configëŠ” diffusion_configë¥¼ ì�˜ë¯¸
        rnet_config.dropout=0.1
        rnet_config.use_grad_checkpoint=True # Script B ì›�ë³¸ ì„¤ì • ìœ ì§€
        super(finetuned_RibonanzaNet, self).__init__(rnet_config)
        if pretrained:
            # ì‹¤ì œ ì‚¬ìš©ì‹œ config.pretrained_weight_pathê°€ ìœ íš¨í•œ ê²½ë¡œì�¸ì§€ í™•ì�¸ í•„ìš”
            self.load_state_dict(torch.load(config.pretrained_weight_path,map_location='cpu'))

        self.dropout=nn.Dropout(0.0)

        decoder_dim=config.decoder_dim
        # SimpleStructureModule ì •ì�˜ê°€ ì�´ ì½”ë“œ ë¸”ë¡� ì�´í›„ì—� ì˜¤ë¯€ë¡œ, ì—¬ê¸°ì„œ ì‚¬ìš© ê°€ëŠ¥
        self.structure_module=[SimpleStructureModule(d_model=decoder_dim, nhead=config.decoder_nhead,
                 dim_feedforward=decoder_dim*4, pairwise_dimension=rnet_config.pairwise_dimension, dropout=0.0) for i in range(config.decoder_num_layers)]
        self.structure_module=nn.ModuleList(self.structure_module)

        self.xyz_embedder=nn.Linear(3,decoder_dim)
        self.xyz_norm=nn.LayerNorm(decoder_dim)
        self.xyz_predictor=nn.Linear(decoder_dim,3)

        self.adaptor=nn.Sequential(nn.Linear(rnet_config.ninp,decoder_dim),nn.LayerNorm(decoder_dim))

        self.distogram_predictor=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                                nn.Linear(rnet_config.pairwise_dimension,40))

        self.time_embedder=SinusoidalPosEmb(decoder_dim)

        self.time_mlp=nn.Sequential(nn.Linear(decoder_dim,decoder_dim),
                                    nn.ReLU(),
                                    nn.Linear(decoder_dim,decoder_dim))
        self.time_norm=nn.LayerNorm(decoder_dim)

        self.distance2pairwise=nn.Linear(1,rnet_config.pairwise_dimension,bias=False)

        self.pair_mlp=nn.Sequential(nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension),
                                    nn.ReLU(),
                                    nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension))

        #hyperparameters for diffusion
        self.n_times = config.n_times # diffusion_configì—�ì„œ n_times ê°€ì ¸ì˜´

        beta_1, beta_T = config.beta_min, config.beta_max
        betas = torch.linspace(start=beta_1, end=beta_T, steps=config.n_times)
        
        # Script A/Bì�˜ ë²„í�¼ ì •ì�˜ë¥¼ í†µí•© ë°� í•„ìš”í•œ ëª¨ë“  ë²„í�¼ ì •ì�˜
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("sqrt_betas", betas.sqrt(), persistent=False)
        
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("sqrt_alphas", alphas.sqrt(), persistent=False)
        
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False) # Script AëŠ” ì—†ì§€ë§Œ, BëŠ” sqrt_alpha_bars ì‚¬ìš©
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt(), persistent=False)
        
        # Script AëŠ” sqrt_1mabar, Script BëŠ” sqrt_one_minus_alpha_bars. ì�´ë¦„ í†µì�¼ ë˜�ëŠ” ë‘˜ ë‹¤ ì •ì�˜
        # sqrt_one_minus_alpha_barsëŠ” sqrt(1-alpha_bars)ì™€ ë�™ì�¼
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt(), persistent=False)

        self.data_std=config.data_std # diffusion_configì—�ì„œ data_std ê°€ì ¸ì˜´

    def custom(self, module): # checkpointë¥¼ ìœ„í•œ ë�˜í�¼
        def custom_forward(*inputs):
            # SimpleStructureModuleì�€ ì�…ë ¥ì�„ í•˜ë‚˜ë¡œ ë°›ìœ¼ë¯€ë¡œ, inputs[0]ì�„ ì „ë‹¬
            if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
                 return module(inputs[0])
            return module(*inputs)
        return custom_forward

    def embed_pair_distance(self,inputs):
        pairwise_features,xyz=inputs
        distance_matrix=xyz[:,None,:,:]-xyz[:,:,None,:]
        # clip min ê°’ì�„ Script A ì²˜ëŸ¼ 1e-9 (ë§¤ìš° ì�‘ì�€ ê°’) ë˜�ëŠ” Script Bì�˜ 2ë¡œ ì„¤ì •
        # Script Bì�˜ clip(2, 37**2)ëŠ” ë¬¼ë¦¬ì � ì�˜ë¯¸ê°€ ì�ˆì�„ ìˆ˜ ì�ˆìœ¼ë¯€ë¡œ ìœ ì§€
        distance_matrix=(distance_matrix**2).sum(-1).clamp(min=1e-9, max=37**2).sqrt() # Script Aì�˜ clamp(min=1e-9) ì �ìš©
        distance_matrix=distance_matrix[:,:,:,None]
        pairwise_features=pairwise_features+self.distance2pairwise(distance_matrix)
        return pairwise_features

    # Script Bì�˜ ì›�ë�˜ forward ë©”ì†Œë“œëŠ” í•™ìŠµ/ë‹¨ì�¼ ìŠ¤í…� ì˜ˆì¸¡ìš©ìœ¼ë¡œ ë³´ì�„. ì¶”ë¡ ì—�ëŠ” ì§�ì ‘ ì‚¬ìš© ì•ˆ í•¨.
    # í•„ìš”í•˜ë‹¤ë©´ ìœ ì§€í•˜ë�˜, ì—¬ê¸°ì„œëŠ” ë ˆì�´í„´íŠ¸ ì¶”ì¶œì�„ ìœ„í•œ ìˆ˜ì •ì—� ì§‘ì¤‘.
    # def forward(self,src,xyz,t): ... (ì›�ë³¸ Script Bì�˜ forward) ...

    def denoise(self,sequence_features,pairwise_features,xyz_current_noise,ts_current_step): # ì�¸ì�� ì�´ë¦„ Script Aì™€ ìœ ì‚¬í•˜ê²Œ ë³€ê²½
        # xyz_current_noise: (N_SAMPLES, SeqLen, 3)
        # ts_current_step: (N_SAMPLES,)
        N = xyz_current_noise.shape[0] # N_SAMPLES

        # sequence_features, pairwise_featuresëŠ” (1, L, D) í˜•íƒœì�¼ ìˆ˜ ì�ˆìœ¼ë¯€ë¡œ Nì—� ë§�ê²Œ í™•ì�¥
        # (ì�´ë¯¸ get_embeddingsì—�ì„œ ë°°ì¹˜ 1ë¡œ ë‚˜ì™”ë‹¤ê³  ê°€ì •)
        seq_f_expanded = sequence_features.expand(N, -1, -1)
        pair_f_expanded = pairwise_features.expand(N, -1, -1, -1)

        current_pair_f = self.embed_pair_distance([pair_f_expanded, xyz_current_noise])
        adapted_seq_f = self.adaptor(seq_f_expanded) # (N, L, decoder_dim)
        
        time_encoding = self.time_embedder(ts_current_step).unsqueeze(1) # (N, 1, decoder_dim)

        # Script Aì�˜ denoise ë¡œì§� ì°¸ê³ 
        tgt = adapted_seq_f + self.xyz_embedder(xyz_current_noise) + time_encoding
        tgt = self.xyz_norm(tgt)

        tgt_after_time_mlp = tgt + self.time_mlp(tgt) # Script Aì—�ì„œëŠ” tgt + self.time_mlp(time_encoding) ì�´ì—ˆìœ¼ë‚˜, ì—¬ê¸°ì„  tgt ì‚¬ìš©
        tgt = self.time_norm(tgt_after_time_mlp) # Script Aì�˜ self.time_norm(tgt + self.time_mlp(time_encoding)) ëŒ€ì‹  ì‚¬ìš©

        for layer in self.structure_module:
            # SimpleStructureModuleì�˜ forwardëŠ” íŠœí”Œ/ë¦¬ìŠ¤íŠ¸ë¥¼ ë‹¨ì�¼ ì�¸ì��ë¡œ ë°›ì�Œ
            # (tgt, src_features, pairwise_features, xyz, mask) ìˆœì„œ
            tgt = layer((tgt, adapted_seq_f, current_pair_f, xyz_current_noise, None))

        final_tgt_features = tgt # ì�´ê²ƒì�´ ë ˆì�´í„´íŠ¸ ë²¡í„° (N_SAMPLES, SeqLen, decoder_dim)
        epsilon_pred = self.xyz_predictor(final_tgt_features) # (N_SAMPLES, SeqLen, 3)

        return epsilon_pred, final_tgt_features


    def extract(self, a, t, x_shape):
        # a: (total_timesteps,) ì˜ˆë¥¼ ë“¤ì–´ self.alphas
        # t: (N_SAMPLES,) ê°� ìƒ˜í”Œì�˜ í˜„ì�¬ íƒ€ì�„ìŠ¤í…� ì�¸ë�±ìŠ¤
        # x_shape: (N_SAMPLES, SeqLen, 3) ë…¸ì�´ì¦ˆ/ë�°ì�´í„°ì�˜ í˜•íƒœ
        device = t.device # aê°€ GPUì—� ì�ˆì�„ ìˆ˜ë�„ ì�ˆê³  CPUì—� ì�ˆì�„ ìˆ˜ë�„ ì�ˆìœ¼ë¯€ë¡œ, tì�˜ ë””ë°”ì�´ìŠ¤ ì‚¬ìš©
        a_gathered = torch.gather(a.to(device), 0, t.long()) # të¥¼ ì�¸ë�±ìŠ¤ë¡œ ì‚¬ìš©í•˜ê¸° ìœ„í•´ long íƒ€ì�…ìœ¼ë¡œ
        # view ëŒ€ì‹  reshape ì‚¬ìš©, Script A ë°©ì‹�ê³¼ ë�™ì�¼í•˜ê²Œ
        return a_gathered.view(t.size(0), *((1,) * (len(x_shape) - 1)))

    # scale_to_minus_one_to_one, reverse_scale_to_zero_to_one, make_noisyëŠ” í•™ìŠµìš©. ì¶”ë¡ ì—�ì„œëŠ” ì§�ì ‘ ì‚¬ìš© ì•ˆí•¨.
    # í•„ìš”ì‹œ ìœ ì§€.

    def denoise_at_t(self, x_t, sequence_features, pairwise_features, timesteps_batch, t_val):
        # x_t: (N_SAMPLES, SeqLen, 3)
        # sequence_features: (1, SeqLen, D_seq) ë˜�ëŠ” (N_SAMPLES, SeqLen, D_seq) - denoise ë‚´ë¶€ì—�ì„œ expandë�¨
        # pairwise_features: (1, SeqLen, SeqLen, D_pair) ë˜�ëŠ” (N_SAMPLES, ...) - denoise ë‚´ë¶€ì—�ì„œ expandë�¨
        # timesteps_batch: (N_SAMPLES,) í˜„ì�¬ ìŠ¤í…� ì�¸ë�±ìŠ¤ (ì˜ˆ: 999, 998, ...)
        # t_val: ìŠ¤ì¹¼ë�¼ ê°’ (í˜„ì�¬ ìŠ¤í…� ì�¸ë�±ìŠ¤)

        # ë…¸ì�´ì¦ˆ ì¶”ê°€: ë§ˆì§€ë§‰ ìŠ¤í…�(t_val=0)ì—�ì„œëŠ” ë…¸ì�´ì¦ˆ 0, ê·¸ ì™¸ì—�ëŠ” ë�œë�¤ ë…¸ì�´ì¦ˆ (Script A ë°©ì‹�)
        noise_for_step = torch.randn_like(x_t) if t_val > 0 else torch.zeros_like(x_t)
        
        # ìˆ˜ì •ë�œ denoise í˜¸ì¶œ: epsilon_predì™€ final_tgt_at_this_step ë°˜í™˜
        epsilon_pred, final_tgt_at_this_step = self.denoise(sequence_features, pairwise_features, x_t, timesteps_batch)
        
        # í•„ìš”í•œ ê³„ìˆ˜ë“¤ ì¶”ì¶œ (self.alphas, self.sqrt_alphas ë“±ì�€ __init__ì—�ì„œ ì˜¬ë°”ë¥´ê²Œ register_buffer ë�˜ì–´ì•¼ í•¨)
        alpha_t = self.extract(self.alphas, timesteps_batch, x_t.shape)
        sqrt_alpha_t = self.extract(self.sqrt_alphas, timesteps_batch, x_t.shape)
        sqrt_1m_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, timesteps_batch, x_t.shape) # ì�´ë¦„ ì�¼ì¹˜
        sqrt_beta_t = self.extract(self.sqrt_betas, timesteps_batch, x_t.shape)
        
        # Script Aì�˜ ë””ë…¸ì�´ì§• ê³µì‹� ì �ìš© (ìˆ˜ì¹˜ ì•ˆì •ì„±ì�„ ìœ„í•´ 1e-9 ì¶”ê°€)
        term_eps_coeff = (1.0 - alpha_t) / (sqrt_1m_alpha_bar_t + 1e-9)
        x_denoised_contribution = x_t - term_eps_coeff * epsilon_pred
        x_t_minus_1 = (1.0 / (sqrt_alpha_t + 1e-9)) * x_denoised_contribution + sqrt_beta_t * noise_for_step
        
        return x_t_minus_1, final_tgt_at_this_step


    def sample(self, src, N): # Nì�€ N_SAMPLES
        # src: (1, SeqLen) ì�…ë ¥ ì‹œí€€ìŠ¤ í† í�°
        L_actual = src.shape[1]
        x_t = torch.randn((N, L_actual, 3), device=src.device) # ì´ˆê¸° ë…¸ì�´ì¦ˆ
        
        # ì´ˆê¸° ì�„ë² ë”© (ë°°ì¹˜ í�¬ê¸° 1ë¡œ ìƒ�ì„±ë�¨)
        sequence_features, pairwise_features = self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        # sequence_features: (1, L, D_seq), pairwise_features: (1, L, L, D_pair)
        
        # distogram ê³„ì‚° (Script B ì›�ë³¸ ë¡œì§� ìœ ì§€, device ì�¼ê´€ì„± ë°� squeeze ë°©ì‹� ìˆ˜ì •)
        distogram_logits = self.distogram_predictor(pairwise_features) # (1, L, L, 40)
        
        # squeezeëŠ” ì°¨ì›�ì�´ 1ì�¸ ê²½ìš°ì—�ë§Œ ìˆ˜í–‰í•˜ë�„ë¡� í•˜ì—¬ N_SAMPLES > 1ì�¼ ë•Œ ë¬¸ì œ ë°©ì§€
        squeezed_distogram_logits = distogram_logits
        if distogram_logits.shape[0] == 1 and distogram_logits.ndim > 3: # (1, L, L, 40) ê°™ì�€ ê²½ìš°
             squeezed_distogram_logits = distogram_logits.squeeze(0) # -> (L, L, 40)

        distogram_values = squeezed_distogram_logits[:,:,2:40] * torch.arange(2, 40, device=src.device).float()
        distogram = distogram_values.sum(-1) # (L,L)

        final_step_tgt_features_for_global = None # ê¸€ë¡œë²Œ í”¼ì²˜ë¥¼ ìœ„í•œ ë³€ìˆ˜

        for t_val in range(self.n_times - 1, -1, -1): # ë†’ì�€ tì—�ì„œ ë‚®ì�€ të¡œ ì§„í–‰ (DDPM ì—­ë°©í–¥)
            timesteps_batch = torch.full((N,), t_val, device=src.device, dtype=torch.long)
            
            # denoise_at_tëŠ” ì�´ì œ ë‘� ê°’ì�„ ë°˜í™˜
            # sequence_features, pairwise_featuresëŠ” ì—¬ê¸°ì„œ (1,L,D) í˜•íƒœë¡œ ì „ë‹¬ë�˜ì–´ë�„
            # denoise_at_t -> denoise ë‚´ë¶€ì—�ì„œ N_SAMPLESì—� ë§�ê²Œ expandë�¨.
            x_t, current_tgt_features = self.denoise_at_t(x_t, sequence_features, pairwise_features, timesteps_batch, t_val)
            
            if t_val == 0: # ë§ˆì§€ë§‰ ìŠ¤í…�ì�˜ í”¼ì²˜ë¥¼ ì €ì�¥
                final_step_tgt_features_for_global = current_tgt_features # (N, L, DecoderDim)
        
        x_0 = x_t * self.data_std # ìµœì¢… ì¢Œí‘œ ìŠ¤ì¼€ì�¼ë§�

        global_features = None
        if final_step_tgt_features_for_global is not None:
            # (N, SeqLen, DecoderDim) -> (N, DecoderDim) ì‹œí€€ìŠ¤ ê¸¸ì�´ì—� ëŒ€í•´ í�‰ê· 
            global_features = torch.mean(final_step_tgt_features_for_global, dim=1)
        else:
            print("Warning: sample ë©”ì„œë“œì—�ì„œ final_step_tgt_features_for_globalì�´ ì„¤ì •ë�˜ì§€ ì•Šì•˜ìŠµë‹ˆë‹¤.")

        return x_0, distogram, global_features # ê¸€ë¡œë²Œ í”¼ì²˜ ì¶”ê°€ ë°˜í™˜

class SimpleStructureModule(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward, pairwise_dimension, dropout=0.1, # Script B ì›�ë³¸ dropout=0.1
                 ):
        super(SimpleStructureModule, self).__init__()
        # MultiHeadAttention í�´ë�˜ìŠ¤ê°€ from Network import * ë¡œ ê°€ì ¸ì™€ì¡Œë‹¤ê³  ê°€ì •
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) # FFN ì¤‘ê°„ì�˜ ë“œë¡­ì•„ì›ƒ
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) # ì–´í…�ì…˜ í›„ ë“œë¡­ì•„ì›ƒ
        self.dropout2 = nn.Dropout(dropout) # FFN í›„ ë“œë¡­ì•„ì›ƒ

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)

        self.activation = nn.GELU()

    def custom(self, module): # ì�´ custom ë©”ì†Œë“œëŠ” gradient checkpointingì�„ ìœ„í•´ ì‚¬ìš©ë�œ ê²ƒìœ¼ë¡œ ë³´ì�„
        def custom_forward(*inputs): # ì¶”ë¡  ì‹œì—�ëŠ” checkpointë¥¼ ì‚¬ìš©í•˜ì§€ ì•Šìœ¼ë¯€ë¡œ ì§�ì ‘ í˜¸ì¶œê³¼ ë�™ì�¼
            # SimpleStructureModuleì�˜ forwardëŠ” íŠœí”Œ/ë¦¬ìŠ¤íŠ¸ í•˜ë‚˜ë¥¼ ë°›ìœ¼ë¯€ë¡œ,
            # inputsê°€ ( (tgt, src, ...), ) í˜•íƒœì�¼ ìˆ˜ ì�ˆì�Œ.
            if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
                return module(inputs[0])
            # ë˜�ëŠ” *inputsê°€ ì�´ë¯¸ í’€ì–´ì§„ (tgt, src, ...) í˜•íƒœì�¼ ìˆ˜ ì�ˆì�Œ.
            # ì�´ ê²½ìš°ì—” return module(*inputs) ê°€ ë§�ì§€ë§Œ, finetuned_RibonanzaNetì�˜ denoiseì—�ì„œ
            # layer((...)) í˜•íƒœë¡œ í˜¸ì¶œí•˜ë¯€ë¡œ, module(inputs[0])ì�´ ë�” ì �ì ˆí•´ ë³´ì�„.
            # í˜¸ì¶œ ë°©ì‹�ì—� ë”°ë�¼ ì¡°ì • í•„ìš”. ì—¬ê¸°ì„œëŠ” layer( (internal_args_tuple) ) ë¡œ ê°€ì •.
            return module(inputs[0]) # í˜¹ì�€ module(*inputs) - í˜¸ì¶œ ë°©ì‹� í™•ì�¸ í•„ìš”

        return custom_forward

    def forward(self, input_tuple): # ì�¸ì�� ì�´ë¦„ì�„ input_tupleë¡œ ëª…í™•í�ˆ í•¨
        # input_tuple: (tgt, adapted_seq_f, current_pair_f, xyz_current_noise, None)
        tgt , src_features, pairwise_features, xyz, src_mask = input_tuple # pred_t ëŒ€ì‹  xyz, src ëŒ€ì‹  src_features ì‚¬ìš©
        
        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)
        # src_maskëŠ” ê¸°ë³¸ì �ìœ¼ë¡œ Noneìœ¼ë¡œ ì „ë‹¬ë�¨. MultiHeadAttentionì—�ì„œ ì²˜ë¦¬.

        res=tgt
        # self.self_attnì�˜ ë„¤ ë²ˆì§¸ ì�¸ì��ëŠ” key_padding_mask (src_maskì™€ ìœ ì‚¬í•œ ì—­í• ì�´ì§€ë§Œ í˜•íƒœ ë‹¤ë¦„) ë˜�ëŠ” attn_mask
        # Script Aì—�ì„œëŠ” biasë¡œ ì‚¬ìš©ë�œ pairwise_biasë¥¼ attn_mask (mask ì�¸ì��)ë¡œ ì „ë‹¬.
        # src_maskëŠ” key_padding_mask ì—­í• .
        # MultiHeadAttention(query, key, value, mask=attn_mask, src_mask=key_padding_mask)
        # Script A: self_attn(tgt, tgt, tgt, mask=bias, src_mask=mask)
        # ì—¬ê¸°ì„œ biasëŠ” pairwise_bias, maskëŠ” key_padding_mask.
        # ì—¬ê¸°ì„œëŠ” src_maskê°€ key_padding_mask ì—­í• .
        tgt, attention_weights = self.self_attn(tgt, tgt, tgt, mask=pairwise_bias, src_mask=src_mask)
        tgt = res + self.dropout1(tgt)
        tgt = self.norm1(tgt)

        res=tgt
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = res + self.dropout2(tgt)
        tgt = self.norm2(tgt)

        return tgt


import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)


diffusion_config=load_config_from_yaml("/kaggle/input/ribonanzanet2-ddpm-v2/diffusion_config.yaml")
rnet_config=load_config_from_yaml("/kaggle/input/ribonanzanet2/pytorch/alpha/1/pairwise.yaml")

model=finetuned_RibonanzaNet(rnet_config,diffusion_config).cuda()



state_dict=torch.load("/kaggle/input/ribonanzanet2-ddpm-v2/RibonanzaNet-DDPM-v2.pt",map_location='cpu')

#get rid of module. from ddp state dict
new_state_dict={}

for key in state_dict:
    new_state_dict[key[7:]]=state_dict[key]

model.load_state_dict(new_state_dict)


from tqdm import tqdm
import numpy as np # í˜¹ì‹œ NumPyê°€ import ì•ˆ ë�˜ì–´ì�ˆì�„ ê²½ìš°ë¥¼ ìœ„í•´ ì¶”ê°€

model.eval()
# preds=[] # ê¸°ì¡´ ì�´ë¦„ ëŒ€ì‹  ëª…í™•í•˜ê²Œ ë³€ê²½
preds_xyz_all_rnas = []       # 3D ì¢Œí‘œ (xyz) ì €ì�¥ìš© ë¦¬ìŠ¤íŠ¸
# preds_distograms_all_rnas = [] # distogram ì €ì�¥ìš© ë¦¬ìŠ¤íŠ¸ (í•„ìš”í•˜ë‹¤ë©´)
preds_global_features_all_rnas = [] # ê¸€ë¡œë²Œ í”¼ì²˜ ì €ì�¥ìš© ë¦¬ìŠ¤íŠ¸ (ìƒˆë¡œ ì¶”ê°€)

for i in tqdm(range(len(test_dataset))):
    src = test_dataset[i]['sequence'].long()
    src = src.unsqueeze(0).cuda()
    # target_id = test_data.loc[i,'target_id'] # ì�´ ë³€ìˆ˜ëŠ” í˜„ì�¬ ë£¨í”„ ë‚´ì—�ì„œ ì§�ì ‘ ì‚¬ìš©ë�˜ì§€ëŠ” ì•Šì�Œ

    # tmp=[] # ì‚¬ìš©ë�˜ì§€ ì•Šì�Œ
    # predicted_dm=[] # ì‚¬ìš©ë�˜ì§€ ì•Šì�Œ
    # for _ in range(5): # ë£¨í”„ ë¶ˆí•„ìš”, sample ë©”ì†Œë“œê°€ Nê°œì�˜ ìƒ˜í”Œì�„ í•œ ë²ˆì—� ìƒ�ì„±

    with torch.no_grad():
        # model.sampleì�˜ ë°˜í™˜ê°’ì�´ 3ê°œë¡œ ë³€ê²½ë�¨
        # xyz_samples: (N_SAMPLES, SeqLen, 3)
        # distogram_pred: (SeqLen, SeqLen) - sample ë©”ì†Œë“œ êµ¬í˜„ì—� ë”°ë�¼ ë‹¤ë¥¼ ìˆ˜ ì�ˆì�Œ
        # global_features_pred: (N_SAMPLES, FeatureDim)
        xyz_samples, distogram_pred, global_features_pred = model.sample(src, 5) # N_SAMPLES=5ë¡œ ê³ ì •

    preds_xyz_all_rnas.append(xyz_samples.cpu().numpy())
    # preds_distograms_all_rnas.append(distogram_pred.cpu().numpy()) # distogramë�„ ì €ì�¥í•˜ë ¤ë©´ ì£¼ì„� í•´ì œ

    if global_features_pred is not None:
        preds_global_features_all_rnas.append(global_features_pred.cpu().numpy())
    else:
        # ê¸€ë¡œë²Œ í”¼ì²˜ê°€ Noneìœ¼ë¡œ ë°˜í™˜ë�  ê²½ìš°ë¥¼ ëŒ€ë¹„ (ì˜ˆ: sample ë©”ì†Œë“œ ë‚´ì—�ì„œ ìƒ�ì„± ì‹¤íŒ¨ ì‹œ)
        # ë˜�ëŠ” ê°� RNA ìƒ˜í”Œë³„ë¡œ Noneì�„ ì¶”ê°€í•˜ê±°ë‚˜, ë¹ˆ NumPy ë°°ì—´ì�„ ì¶”ê°€í•  ìˆ˜ ì�ˆìŠµë‹ˆë‹¤.
        # ì—¬ê¸°ì„œëŠ” (N_SAMPLES, 0) í˜•íƒœì�˜ ë¹ˆ ë°°ì—´ì�„ ì¶”ê°€í•˜ì—¬ ì°¨ì›� ìˆ˜ë¥¼ ìœ ì§€í•˜ë�„ë¡� í•¨ (ì‹¤ì œë¡œëŠ” Noneì�´ ë�” ì �ì ˆí•  ìˆ˜ ì�ˆì�Œ)
        # ë˜�ëŠ” ì•„ë�˜ ì €ì�¥ ë¡œì§�ì—�ì„œ Noneì�„ ì²˜ë¦¬í•˜ë�„ë¡� í•¨
        preds_global_features_all_rnas.append(None) # ë˜�ëŠ” np.empty((5, 0)) ë“± ìƒ�í™©ì—� ë§�ê²Œ


import json
import torch
import numpy as np
from tqdm.auto import tqdm # tqdm ì¶”ê°€

# --- ì�´ì „ì—� ì •ì�˜ë�˜ì—ˆê±°ë‚˜ ë¡œë“œë�˜ì—ˆë‹¤ê³  ê°€ì •í•˜ëŠ” ë³€ìˆ˜ë“¤ ---
# preds_xyz_all_rnas: DDPM ì¶”ë¡  ê²°ê³¼ 3D ì¢Œí‘œ ë¦¬ìŠ¤íŠ¸ (ê°� ìš”ì†ŒëŠ” NumPy ë°°ì—´ (5, SeqLen, 3))
# preds_global_features_all_rnas: DDPM ì¶”ë¡  ê²°ê³¼ ê¸€ë¡œë²Œ í”¼ì²˜ ë¦¬ìŠ¤íŠ¸ (ê°� ìš”ì†ŒëŠ” NumPy ë°°ì—´ (5, FeatureDim) ë˜�ëŠ” None)
# test_dataset: RNADataset ì�¸ìŠ¤í„´ìŠ¤ (src_tokens ê°€ì ¸ì˜¤ê¸°ìš©)
# test_data: Pandas DataFrame (ì›�ë³¸ test_sequences.csv ë¡œë“œí•œ ê²ƒ. original_rna_id, rna_sequence_str ê°€ì ¸ì˜¤ê¸°ìš©)
# DEVICE: torch.device ì„¤ì • (ì˜ˆ: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# -------------------------------------------------------------

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ìŠ¤ì¼€ì�¼ëŸ¬ íŒŒë�¼ë¯¸í„° ë¡œë“œ í•¨ìˆ˜ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_scaler_params(scaler_json_path):
    """ ì €ì�¥ë�œ ìŠ¤ì¼€ì�¼ëŸ¬ íŒŒë�¼ë¯¸í„°(í�‰ê· , í‘œì¤€í�¸ì°¨)ë¥¼ ë¡œë“œí•˜ëŠ” í•¨ìˆ˜ """
    try:
        with open(scaler_json_path, 'r') as f:
            scaler_params = json.load(f)
        means = {k: float(v) for k, v in scaler_params['means'].items()}
        stds = {k: float(v) for k, v in scaler_params['stds'].items()}
        print(f"Scaler params loaded from {scaler_json_path}")
        return means, stds
    except FileNotFoundError:
        print(f"Error: Scaler params file not found at {scaler_json_path}. Using default (mean 0, std 1).")
        return {'x': 0., 'y': 0., 'z': 0.}, {'x': 1., 'y': 1., 'z': 1.}
    except Exception as e:
        print(f"Error loading scaler params: {e}. Using default (mean 0, std 1).")
        return {'x': 0., 'y': 0., 'z': 0.}, {'x': 1., 'y': 1., 'z': 1.}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ í…�ì„œ ì¢Œí‘œ ì²˜ë¦¬ í•¨ìˆ˜ (ì„¼í„°ë§�, í‘œì¤€í™”) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def center_coordinates_tensor(coords_tensor):
    """ 3D ì¢Œí‘œ í…�ì„œë¥¼ ì„¼í„°ë§�í•©ë‹ˆë‹¤. NaNì�„ ì œì™¸í•˜ê³  í�‰ê· ì�„ ê³„ì‚°í•©ë‹ˆë‹¤. """
    if coords_tensor.ndim == 2: # (SeqLen, 3)
        coords_tensor_batched = coords_tensor.unsqueeze(0) # (1, SeqLen, 3)ìœ¼ë¡œ ë§Œë“¦
    else: # (N_SAMPLES, SeqLen, 3)
        coords_tensor_batched = coords_tensor

    centered_coords_list = []
    for i in range(coords_tensor_batched.shape[0]): # ê°� ìƒ˜í”Œì—� ëŒ€í•´ ì²˜ë¦¬
        sample_coords = coords_tensor_batched[i] # (SeqLen, 3)
        valid_mask = ~torch.isnan(sample_coords).any(dim=1)
        if valid_mask.sum() > 0:
            mean_for_centering = sample_coords[valid_mask].mean(dim=0, keepdim=True) # (1, 3)
        else:
            mean_for_centering = torch.zeros((1, 3), dtype=sample_coords.dtype, device=sample_coords.device)
        centered_coords_list.append(sample_coords - mean_for_centering)
    
    if not centered_coords_list: # ëª¨ë“  ìƒ˜í”Œì�´ ìœ íš¨í•˜ì§€ ì•Šì�€ ê·¹ë‹¨ì � ê²½ìš°
         return torch.full_like(coords_tensor_batched, float('nan'))

    output_tensor = torch.stack(centered_coords_list)
    if coords_tensor.ndim == 2 and output_tensor.shape[0] == 1 : # ì›�ë�˜ ì°¨ì›�ìœ¼ë¡œ ë³µì›�
        output_tensor = output_tensor.squeeze(0)
    return output_tensor

def standardize_coordinates_tensor(coords_tensor, means_dict, stds_dict, epsilon=1e-8):
    """ 3D ì¢Œí‘œ í…�ì„œë¥¼ Z-score í‘œì¤€í™”í•©ë‹ˆë‹¤. """
    if coords_tensor.ndim == 2: # (SeqLen, 3)
        single_sample = True
        coords_tensor_batched = coords_tensor.unsqueeze(0) # (1, SeqLen, 3)ìœ¼ë¡œ ë§Œë“¦
    else: # (N_SAMPLES, SeqLen, 3)
        single_sample = False
        coords_tensor_batched = coords_tensor
    
    device = coords_tensor_batched.device
    # means_dictì™€ stds_dictì�˜ íƒ€ì�…ì�´ floatì�„ì�„ load_scaler_paramsì—�ì„œ ë³´ì�¥
    means_tensor = torch.tensor([[means_dict['x'], means_dict['y'], means_dict['z']]], dtype=torch.float32, device=device) # (1, 1, 3)
    stds_tensor = torch.tensor([[stds_dict['x'], stds_dict['y'], stds_dict['z']]], dtype=torch.float32, device=device) # (1, 1, 3)

    standardized_tensor = (coords_tensor_batched - means_tensor) / (stds_tensor + epsilon)
    
    if single_sample:
        standardized_tensor = standardized_tensor.squeeze(0) # (SeqLen, 3)
    return standardized_tensor

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ë�°ì�´í„° ì „ì²˜ë¦¬ ì‹¤í–‰ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# <<<< ì¤‘ìš”: ì‹¤ì œ ìŠ¤ì¼€ì�¼ëŸ¬ íŒŒë�¼ë¯¸í„° íŒŒì�¼ ê²½ë¡œë¡œ ìˆ˜ì •í•˜ì„¸ìš” >>>>
SCALER_PARAMS_PATH = "/kaggle/input/data-for-egnn/coordinate_scaler_params_gb_feature.v2.json" # ì˜ˆì‹œ ê²½ë¡œ
loaded_means, loaded_stds = load_scaler_params(SCALER_PARAMS_PATH)

# ìµœì¢…ì �ìœ¼ë¡œ ê·¸ë�˜í”„ë¡œ ë³€í™˜í•  ë�°ì�´í„°ë¥¼ ë‹´ì�„ ë¦¬ìŠ¤íŠ¸
processed_data_for_graph_conversion = []

print("\nDDPM ì¶”ë¡  ê²°ê³¼ì—� ëŒ€í•œ ë�°ì�´í„° ì „ì²˜ë¦¬ (ì„¼í„°ë§� ë°� í‘œì¤€í™”)ë¥¼ ì‹œì�‘í•©ë‹ˆë‹¤...")
# len(test_data)ëŠ” DDPM ì¶”ë¡  ë£¨í”„ì—�ì„œ ì‚¬ìš©í•œ RNA ê°œìˆ˜ì™€ ë�™ì�¼í•´ì•¼ í•¨
for i in tqdm(range(len(test_data)), desc="ì„¼í„°ë§� ë°� í‘œì¤€í™” ì¤‘"):
    # test_datasetìœ¼ë¡œë¶€í„° src_tokens ê°€ì ¸ì˜¤ê¸°
    # DDPM ì¶”ë¡  ì‹œ ì‚¬ìš©í•œ srcì™€ ë�™ì�¼í•œ ê²ƒì�„ ì‚¬ìš©í•´ì•¼ í•¨
    src_tokens_for_entry = test_dataset[i]['sequence'].long() # (SeqLen,)

    # test_data DataFrameìœ¼ë¡œë¶€í„° ID ë°� ì‹œí€€ìŠ¤ ë¬¸ì��ì—´ ê°€ì ¸ì˜¤ê¸°
    original_rna_id = test_data.loc[i, 'target_id']
    rna_sequence_str = test_data.loc[i, 'sequence']

    # DDPM ì¶”ë¡  ê²°ê³¼ (NumPy ë°°ì—´)ë¥¼ PyTorch í…�ì„œë¡œ ë³€í™˜ ë°� DEVICEë¡œ ì�´ë�™
    xyz_samples_raw_np = preds_xyz_all_rnas[i] # (5, SeqLen, 3) NumPy
    xyz_samples_raw = torch.from_numpy(xyz_samples_raw_np).float().to(DEVICE)

    global_features_np = preds_global_features_all_rnas[i] # (5, FeatureDim) NumPy ë˜�ëŠ” None
    global_features_pred_tensor = None
    if global_features_np is not None and isinstance(global_features_np, np.ndarray) and global_features_np.size > 0:
        global_features_pred_tensor = torch.from_numpy(global_features_np).float().to(DEVICE)

    # 1. ì„¼í„°ë§� ì �ìš© (ê°� ìƒ˜í”Œì—� ëŒ€í•´)
    xyz_samples_centered = center_coordinates_tensor(xyz_samples_raw) # (5, SeqLen, 3) Tensor on DEVICE

    # 2. Z-score í‘œì¤€í™” ì �ìš© (ê°� ìƒ˜í”Œì—� ëŒ€í•´)
    xyz_samples_standardized = standardize_coordinates_tensor(xyz_samples_centered, loaded_means, loaded_stds) # (5, SeqLen, 3) Tensor on DEVICE

    # ê°� ìƒ˜í”Œë³„ë¡œ ì²˜ë¦¬ë�œ ë�°ì�´í„°ë¥¼ ë”•ì…”ë„ˆë¦¬ í˜•íƒœë¡œ ì €ì�¥
    for sample_idx in range(xyz_samples_standardized.shape[0]): # 5ë²ˆ ë°˜ë³µ
        current_xyz_std_list = xyz_samples_standardized[sample_idx].cpu().tolist()

        current_global_feature_list = None
        if global_features_pred_tensor is not None and sample_idx < global_features_pred_tensor.shape[0]:
            current_global_feature_list = global_features_pred_tensor[sample_idx].cpu().tolist()

        processed_entry = {
            "id": f"{original_rna_id}_{sample_idx + 1}",
            "sequence": rna_sequence_str,
            "coords_processed": current_xyz_std_list, # ì„¼í„°ë§� ë°� í‘œì¤€í™”ë�œ ì¢Œí‘œ
            "global_feature": current_global_feature_list,
            "src_tokens": src_tokens_for_entry.cpu().tolist() # ì›�ë³¸ í† í�°
        }
        processed_data_for_graph_conversion.append(processed_entry)

print(f"ì´� {len(processed_data_for_graph_conversion)}ê°œì�˜ ì²˜ë¦¬ë�œ entry ìƒ�ì„± ì™„ë£Œ.")
print("ì�´ì œ `processed_data_for_graph_conversion` ë¦¬ìŠ¤íŠ¸ë¥¼ ì‚¬ìš©í•˜ì—¬ ê·¸ë�˜í”„ ë³€í™˜ ë°� DataLoader ìƒ�ì„±ì�„ ì§„í–‰í•  ìˆ˜ ì�ˆìŠµë‹ˆë‹¤.")

# # (ì„ íƒ� ì‚¬í•­) ì¤‘ê°„ ê²°ê³¼ í™•ì�¸ ë˜�ëŠ” ì €ì�¥ - ë””ë²„ê¹…ìš©
# if processed_data_for_graph_conversion:
#     print("\nì²« ë²ˆì§¸ ì²˜ë¦¬ë�œ entry ì˜ˆì‹œ:")
#     print(json.dumps(processed_data_for_graph_conversion[0], indent=2))
#
#     # output_json_filepath_debug = "debug_predictions_processed_latent.json"
#     # with open(output_json_filepath_debug, 'w') as f:
#     #     json.dump(processed_data_for_graph_conversion, f, indent=2)
#     # print(f"ë””ë²„ê¹…ìš© ì²˜ë¦¬ ë�°ì�´í„°ê°€ '{output_json_filepath_debug}'ì—� ì €ì�¥ë�˜ì—ˆìŠµë‹ˆë‹¤.")


#score val
import pandas as pd
import pandas.api.types
import os
import re

# Function to parse TMscore output
def parse_tmscore_output(output):
    result = {}

    # Extract TM-score based on length of reference structure (second)
    tm_score_match = re.findall(r"TM-score=\s+([\d.]+)", output)[1]
    result['TM-score'] = float(tm_score_match) if tm_score_match else None

    return result

def write_pdb_line(atom_name, atom_serial, residue_name, chain_id, residue_num, x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P'):
    """
    Writes a single line of PDB format based on provided atom information. 
    
    Args:
        atom_name (str): Name of the atom (e.g., "N", "CA").
        atom_serial (int): Atom serial number.
        residue_name (str): Residue name (e.g., "ALA"). 
        chain_id (str): Chain identifier. 
        residue_num (int): Residue number. 
        x_coord (float): X coordinate.
        y_coord (float): Y coordinate.
        z_coord (float): Z coordinate.
        occupancy (float, optional): Occupancy value (default: 1.0). 
        b_factor (float, optional): B-factor value (default: 0.0). 
    
    Returns:
        str: A single line of PDB string.
    """
    line = f"ATOM  {atom_serial:>5d}  {atom_name:<5s} {residue_name:<3s} {residue_num:>3d}    {x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           {atom_type}\n"
    return line

def write2pdb(df, xyz_id, pdb_path):
    resolved_cnt=0
    with open(pdb_path, "w") as pdb_file:
        for _, row in df.iterrows():
            x_coord=row[f"x_{xyz_id}"]
            y_coord=row[f"y_{xyz_id}"]
            z_coord=row[f"z_{xyz_id}"]

            if x_coord>-1e17 and y_coord>-1e17 and z_coord>-1e17:
            #if True:
                resolved_cnt+=1
                pdb_line = write_pdb_line(
                    atom_name="C1'", 
                    atom_serial=int(row["resid"]), 
                    residue_name=row['resname'], 
                    chain_id='0', 
                    residue_num=int(row["resid"]), 
                    x_coord=x_coord, 
                    y_coord=y_coord, 
                    z_coord=z_coord,
                    atom_type="C"
                )
                pdb_file.write(pdb_line)
    return resolved_cnt


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Computes the TM-score between predicted and native RNA structures using USalign.

    This function evaluates the structural similarity of RNA predictions to native structures
    by computing the TM-score. It uses USalign, a structural alignment tool, to compare
    the predicted structures with the native structures.

    Workflow:
    1. Copies the USalign binary to the working directory and grants execution permissions.
    2. Extracts the `pdb_id` from the `ID` column of both the solution and submission DataFrames.
    3. Iterates over each unique `pdb_id`, grouping the native and predicted structures.
    4. Writes PDB files for native and predicted structures.
    5. Runs USalign on each predicted-native pair and extracts the TM-score.
    6. Computes the highest TM-score per target and returns aggregated results.

    Args:
        solution (pd.DataFrame): A DataFrame containing the native RNA structures.
        submission (pd.DataFrame): A DataFrame containing the predicted RNA structures.
        row_id_column_name (str): The name of the column containing unique row identifiers.

    Returns:
        tuple:
            - results (list): The highest TM-score for each `pdb_id`.
            - results_per_sub (list): TM-scores for each predicted-native pair.
            - outputs (list): Raw output logs from USalign for debugging.
    '''

    os.system("cp /kaggle/input/usalign/USalign /kaggle/working/")
    os.system("sudo chmod u+x /kaggle/working//USalign")


    # Extract pdb_id from ID (pdb_resid)
    solution["pdb_id"] = solution["ID"].apply(lambda x: x.split("_")[0])
    submission["pdb_id"] = submission["ID"].apply(lambda x: x.split("_")[0])

    #fix pdb_ids comment out later
    # solution.loc[solution['pdb_id']=="R1138v1",'pdb_id']='R1138'
    # solution.loc[solution['pdb_id']=="R1117",'pdb_id']='R1117v2'
    
    results=[]
    outputs=[]
    results_per_sub=[]
    # Iterate through each pdb_id and generate PDB files for both clean and corrupted data
    for pdb_id, group_native in solution.groupby("pdb_id"):
        group_predicted = submission[submission["pdb_id"] == pdb_id]
        #print(group_native,group_predicted)
        # Define output file paths
        # clean_pdb_path = os.path.join(output_folder, f"{pdb_id}_C3_clean.pdb")
        # corrupted_pdb_path = os.path.join(output_folder, f"{pdb_id}_C3_corrupted.pdb")
        native_pdb=f'native.pdb'
        predicted_pdb=f'predicted.pdb'

        all_scores=[]
        for pred_cnt in range(1,6):
            tmp=[]
            for native_cnt in range(1,41):
                # Write solution PDB
                resolved_cnt=write2pdb(group_native, native_cnt, native_pdb)
                
                # Write predicted PDB
                _=write2pdb(group_predicted, pred_cnt, predicted_pdb)

                if resolved_cnt>0:
                    command = f"/kaggle/working/USalign {predicted_pdb} {native_pdb} -atom \" C1'\""
                    output = os.popen(command).read()
                    outputs.append(output)
                    parsed_data = parse_tmscore_output(output)
                    tmp.append(parsed_data['TM-score'])
                    
            all_scores.append(max(tmp))
        # print(output)
        # stop
        print(pdb_id)
        print(all_scores)
        results_per_sub.append(all_scores)
        results.append(max(all_scores))
    
    print(results)
    #return sum(results)/len(results), outputs
    return results, results_per_sub, outputs
    #return outputs

if 'R1107' in set(test_data['target_id']):
    solution=pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
    submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')  # <<< Update to your submission path

    
    scores,results_per_sub,outputs=score(solution,submission,'ID')
    print(np.mean(scores))


# --- ì�´ì „ Script Bì�˜ ì¶”ë¡  ë°� ì „ì²˜ë¦¬ ë£¨í”„ ì�´í›„ ---
# processed_data_for_graph_conversion ë¦¬ìŠ¤íŠ¸ê°€ ì¤€ë¹„ë�œ ìƒ�íƒœë�¼ê³  ê°€ì •
# ì˜ˆì‹œ:
# processed_data_for_graph_conversion = [
#  { "id": "RNA_A_1", "sequence": "AUCG...", "coords_processed": [[...],[...]],
#    "global_feature": [...], "src_tokens": [...] },
#  ...
# ]
# --------------------------------------------------

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch as GeomBatch
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import List, Dict, Tuple
import random
import math
from tqdm.auto import tqdm # tqdm ì¶”ê°€

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 0. ìƒ�ìˆ˜ (í•„ìš”ì‹œ ê°’ ì¡°ì •) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
NT2I       = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': 0} # 'N'ì�„ 'A'ì™€ ë�™ì�¼í•˜ê²Œ ì²˜ë¦¬
KNN_K      = 10  # k-NNì�˜ k ê°’
EDGE_DIM   = 15  # 4(oh_i) + 4(oh_j) + 1(bb) + 3(delta) + 1(dist) + 1(ri) + 1(rj)
EXP_CF_DIM = 768 # ì˜ˆìƒ�ë�˜ëŠ” conditioning_feature(global_feature) ì°¨ì›�
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 1. Edge builder (k-NN) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_edges_knn(
    pos_tensor: torch.Tensor, # k-NN êµ¬ì„±ì—� ì‚¬ìš©í•  ì¢Œí‘œ (í‘œì¤€í™”ë�œ ì¢Œí‘œ ë˜�ëŠ” ì„¼í„°ë§�ë�œ ì¢Œí‘œ)
    seq_str: str,
    resid_list: List[int],    # 1-based resid ë¦¬ìŠ¤íŠ¸
    k: int = KNN_K
) -> Tuple[torch.Tensor, torch.Tensor]:
    L = pos_tensor.size(0)
    if L == 0:
        return torch.empty((2, 0), dtype=torch.long, device=pos_tensor.device), \
               torch.empty((0, EDGE_DIM), dtype=torch.float32, device=pos_tensor.device)

    dmat = torch.cdist(pos_tensor, pos_tensor) # ê±°ë¦¬ í–‰ë ¬
    pairs = set()

    # k-ìµœê·¼ì ‘ ì�´ì›ƒ (ì��ê¸° ì��ì‹  ì œì™¸)
    for i_node_idx in range(L):
        neighbor_indices = torch.arange(L, device=pos_tensor.device)
        neighbor_indices = neighbor_indices[neighbor_indices != i_node_idx]
        if len(neighbor_indices) == 0:
            continue
        distances_to_i = dmat[i_node_idx, neighbor_indices]
        sorted_neighbor_indices_by_dist = neighbor_indices[torch.argsort(distances_to_i)]
        effective_k = min(k, len(sorted_neighbor_indices_by_dist))
        for j_local_idx in range(effective_k):
            pairs.add((i_node_idx, sorted_neighbor_indices_by_dist[j_local_idx].item()))

    # Backbone (i â†” i+1, ì–‘ë°©í–¥) ì—£ì§€ ì¶”ê°€
    for i_node_idx in range(L - 1):
        pairs.add((i_node_idx, i_node_idx + 1))
        pairs.add((i_node_idx + 1, i_node_idx))

    send_nodes, receive_nodes, edge_features_list = [], [], []
    length_minus_one_normalized = max(L - 1, 1) # ì •ê·œí™”ë¥¼ ìœ„í•œ ë¶„ëª¨ (0ìœ¼ë¡œ ë‚˜ëˆ„ê¸° ë°©ì§€)

    for i_node, j_node in pairs:
        dev = pos_tensor.device
        #send_nodes.append(i_node)
        #receive_nodes.append(j_node)
        base_i_char = seq_str[i_node]
        base_j_char = seq_str[j_node]
        # NT2I.getì�˜ ë‘� ë²ˆì§¸ ì�¸ì��ëŠ” í‚¤ê°€ ì—†ì�„ ë•Œ ë°˜í™˜í•  ê¸°ë³¸ê°’ (ì—¬ê¸°ì„œëŠ” 'N'ì�˜ ì�¸ë�±ìŠ¤)
        one_hot_i = F.one_hot(torch.tensor(NT2I.get(base_i_char.upper(), NT2I['N']), device=dev), num_classes=4).float()
        one_hot_j = F.one_hot(torch.tensor(NT2I.get(base_j_char.upper(), NT2I['N']), device=dev), num_classes=4).float()
        is_backbone_edge = torch.tensor([1.0 if abs(i_node - j_node) == 1 else 0.0], device=dev)
        delta_coords = pos_tensor[i_node] - pos_tensor[j_node] # ë°©í–¥ ë²¡í„°
        distance_val = torch.norm(delta_coords, p=2).unsqueeze(0) # L2 norm ê±°ë¦¬
        # resid_listëŠ” 1-based index
        resid_i_normalized = torch.tensor([(resid_list[i_node] - 1) / length_minus_one_normalized], device=dev)
        resid_j_normalized = torch.tensor([(resid_list[j_node] - 1) / length_minus_one_normalized], device=dev)
        edge_features_list.append(torch.cat([one_hot_i, one_hot_j, is_backbone_edge,
                                             delta_coords, distance_val, resid_i_normalized, resid_j_normalized]))

    edge_index_tensor = (torch.tensor([send_nodes, receive_nodes], dtype=torch.long, device=pos_tensor.device)
                         if pairs else torch.empty((2, 0), dtype=torch.long, device=pos_tensor.device))
    edge_attr_tensor = (torch.stack(edge_features_list).to(device=pos_tensor.device, dtype=torch.float32) if edge_features_list
                        else torch.empty((0, EDGE_DIM), dtype=torch.float32, device=pos_tensor.device))
    return edge_index_tensor, edge_attr_tensor

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 2. processed_entry â†’ PyG Data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def processed_entry_to_pyg_data(entry: Dict) -> Data:
    """ Script Bì�˜ processed_entry ë”•ì…”ë„ˆë¦¬ë¥¼ PyTorch Geometric Data ê°�ì²´ë¡œ ë³€í™˜í•©ë‹ˆë‹¤."""
    seq_str      = entry["sequence"]
    entry_id     = entry["id"]
    # coords_processedëŠ” ì�´ë¯¸ NumPy ë°°ì—´ì—�ì„œ tolist()ë�œ ë¦¬ìŠ¤íŠ¸ ìƒ�íƒœ
    pos_std_list = entry["coords_processed"] # í‘œì¤€í™”ë�œ ì¢Œí‘œ (EGNN ì�…ë ¥ìš©)
    global_feature_list = entry["global_feature"]
    # src_tokens_list = entry["src_tokens"] # í•„ìš”ì‹œ ì‚¬ìš©

    L = len(seq_str)
    if not (L > 0 and isinstance(pos_std_list, list) and len(pos_std_list) == L):
         raise ValueError(f"ID {entry_id}: ì‹œí€€ìŠ¤(L={L})ì™€ ì¢Œí‘œ(L={len(pos_std_list) if pos_std_list else 'None'}) ê¸¸ì�´ ë¶ˆì�¼ì¹˜ ë˜�ëŠ” ì¢Œí‘œ íƒ€ì�… ì˜¤ë¥˜.")

    # PyG Data ê°�ì²´ ìƒ�ì„± ì‹œ ëª¨ë“  í…�ì„œëŠ” ë�™ì�¼í•œ ë””ë°”ì�´ìŠ¤ì—� ì�ˆì–´ì•¼ í•¨ (DEVICEë¡œ í†µì�¼)
    pos_tensor_for_graph = torch.tensor(pos_std_list, dtype=torch.float32, device=DEVICE)

    resname_list = [s_char for s_char in seq_str] # ì˜ˆ: ['A', 'U', 'G', ...]
    resid_list   = list(range(1, L + 1))      # ì˜ˆ: [1, 2, ..., L] (1-based)

    # k-NN ê·¸ë�˜í”„ ì—£ì§€ ë°� ì—£ì§€ íŠ¹ì§• ìƒ�ì„± (ìœ„ì—�ì„œ ì •ì�˜í•œ í•¨ìˆ˜ ì‚¬ìš©)
    # ì—¬ê¸°ì„œëŠ” í‘œì¤€í™”ë�œ ì¢Œí‘œ(pos_tensor_for_graph)ë¡œ k-NNì�„ ë§Œë“­ë‹ˆë‹¤.
    edge_i, edge_a = build_edges_knn(pos_tensor_for_graph, seq_str, resid_list)

    # ê¸€ë¡œë²Œ í”¼ì²˜ (conditioning_feature) ì²˜ë¦¬
    cf_tensor = torch.zeros(EXP_CF_DIM, device=DEVICE) # ê¸°ë³¸ê°’ì�€ 0ìœ¼ë¡œ ì±„ì›Œì§„ í…�ì„œ
    if global_feature_list is not None:
        temp_cf_tensor = torch.tensor(global_feature_list, dtype=torch.float32, device=DEVICE)
        temp_cf_tensor = temp_cf_tensor.view(-1) # 1Dë¡œ ë§Œë“¦
        if temp_cf_tensor.shape[0] == EXP_CF_DIM:
            cf_tensor = temp_cf_tensor
        elif temp_cf_tensor.shape[0] < EXP_CF_DIM:
            padding = torch.zeros(EXP_CF_DIM - temp_cf_tensor.shape[0], device=DEVICE)
            cf_tensor = torch.cat([temp_cf_tensor, padding])
        else: # EXP_CF_DIMë³´ë‹¤ ê¸´ ê²½ìš° ì��ë¥´ê¸°
            cf_tensor = temp_cf_tensor[:EXP_CF_DIM]
    # cf_tensorëŠ” (EXP_CF_DIM,) í˜•íƒœ. collate ì‹œ (BatchSize, 1, EXP_CF_DIM)ì�´ ë�  ìˆ˜ ì�ˆìœ¼ë¯€ë¡œ Datasetì—�ì„œ (1, Dim)ìœ¼ë¡œ ë§Œë“¦

    # ë…¸ë“œ íŠ¹ì§• 'x' ìƒ�ì„±: One-hot(resname) + normalized resid
    one_hot_node_features = torch.stack([
        F.one_hot(torch.tensor(NT2I.get(ch.upper(), NT2I['N'])), num_classes=4)
        for ch in resname_list
    ]).float().to(DEVICE)
    resid_norm_node_features = ((torch.tensor(resid_list, dtype=torch.float32, device=DEVICE) - 1)
                             / max(L - 1, 1)).unsqueeze(1)
    node_x_features = torch.cat([one_hot_node_features, resid_norm_node_features], dim=1) # (L, 5)

    # PyG Data ê°�ì²´ ìƒ�ì„±
    # EGNN ì¶”ë¡ ë§Œ í•˜ëŠ” ê²½ìš°, íƒ€ê²Ÿ yëŠ” í•„ìš” ì—†ì�„ ìˆ˜ ì�ˆìŠµë‹ˆë‹¤.
    data = Data(
        x=node_x_features,                # (L, 5) ë…¸ë“œ íŠ¹ì§•
        pos=pos_tensor_for_graph,         # (L, 3) í‘œì¤€í™”ë�œ ì¢Œí‘œ (EGNN ì�…ë ¥)
        edge_index=edge_i,                # (2, NumEdges)
        edge_attr=edge_a,                 # (NumEdges, EDGE_DIM)
        conditioning_feature=cf_tensor.unsqueeze(0), # (1, EXP_CF_DIM) í˜•íƒœë¡œ ì €ì�¥ (collate ìš©ì�´)
        id=entry_id,
        sequence_str=seq_str,
        num_nodes = L # num_nodes ëª…ì‹œì � ì¶”ê°€
        # resname, residëŠ” í•„ìš”ì‹œ ì¶”ê°€ ì €ì�¥ ê°€ëŠ¥ (ì�´ë¯¸ x ìƒ�ì„±ì—� ì‚¬ìš©ë�¨)
    )
    return data

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ê·¸ë�˜í”„ ë�°ì�´í„° ìƒ�ì„± ì‹¤í–‰ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\nì²˜ë¦¬ë�œ entry ë¦¬ìŠ¤íŠ¸ë¥¼ PyG Data ê°�ì²´ë¡œ ë³€í™˜í•©ë‹ˆë‹¤ (ì´� {len(processed_data_for_graph_conversion)}ê°œ)...")
pyg_data_list_for_inference = []
skipped_conversion_count = 0
for entry_dict_item in tqdm(processed_data_for_graph_conversion, desc="PyG Data ê°�ì²´ ë³€í™˜ ì¤‘"):
    try:
        pyg_data_item = processed_entry_to_pyg_data(entry_dict_item)
        pyg_data_list_for_inference.append(pyg_data_item)
    except ValueError as ve:
        # print(f"Warning: ID {entry_dict_item.get('id', '?')} ë³€í™˜ ì˜¤ë¥˜ (ValueError): {ve}. ê±´ë„ˆëœ�ë‹ˆë‹¤.")
        skipped_conversion_count += 1
    except Exception as e_conv:
        # print(f"Warning: ID {entry_dict_item.get('id', '?')} ë³€í™˜ ì¤‘ ì•Œ ìˆ˜ ì—†ëŠ” ì˜¤ë¥˜: {e_conv}. ê±´ë„ˆëœ�ë‹ˆë‹¤.")
        skipped_conversion_count += 1

if skipped_conversion_count > 0:
    print(f"Warning: ì´� {skipped_conversion_count}ê°œì�˜ entryê°€ PyG Data ê°�ì²´ ë³€í™˜ì—� ì‹¤íŒ¨í•˜ì—¬ ê±´ë„ˆë›°ì—ˆìŠµë‹ˆë‹¤.")
print(f"ì´� {len(pyg_data_list_for_inference)}ê°œì�˜ PyG Data ê°�ì²´ ìƒ�ì„± ì™„ë£Œ.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ 3. ë‹¨ìˆœí™”ë�œ Dataset, Sampler, DataLoader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 3.1) Dataset (ë©”ëª¨ë¦¬ ë‚´ ê·¸ë�˜í”„ ë¦¬ìŠ¤íŠ¸ ì‚¬ìš©)
class InMemoryGraphDataset(torch.utils.data.Dataset):
    def __init__(self, pyg_data_list: List[Data]):
        super().__init__()
        self.graphs = pyg_data_list
        # Data ê°�ì²´ ìƒ�ì„± ì‹œ conditioning_featureë¥¼ (1, EXP_CF_DIM)ìœ¼ë¡œ ì�´ë¯¸ ë§Œë“¦

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

# 3.2) BucketBatchSampler (ì„ íƒ� ì‚¬í•­, ì¶”ë¡  ì‹œ ë‹¨ìˆœ ìˆœì°¨ ì²˜ë¦¬ë�„ ê°€ëŠ¥)
#      ë©”ëª¨ë¦¬ê°€ ë§¤ìš° ë‹¤ì–‘í•œ ê¸¸ì�´ì�˜ ê·¸ë�˜í”„ë¡œ ì�¸í•´ ë¬¸ì œê°€ ë�  ê²½ìš°ì—�ë§Œ ìœ ìš©.
#      ì—¬ê¸°ì„œëŠ” ë‹¨ìˆœí™”ë¥¼ ìœ„í•´ ì‚¬ìš©í•˜ì§€ ì•Šì�Œ. í•„ìš”ì‹œ ì�´ì „ ì½”ë“œì—�ì„œ ê°€ì ¸ì™€ ì‚¬ìš©.

# 3.3) Collate Function (conditioning_feature ì²˜ë¦¬)
def collate_graphs_for_egnn(data_list: List[Data]) -> GeomBatch:
    batch = GeomBatch.from_data_list(data_list) # PyGê°€ ëŒ€ë¶€ë¶„ ì��ë�™ ì²˜ë¦¬
    # Data ê°�ì²´ ìƒ�ì„± ì‹œ conditioning_featureë¥¼ (1, EXP_CF_DIM)ìœ¼ë¡œ ë§Œë“¤ì—ˆìœ¼ë¯€ë¡œ,
    # ë°°ì¹˜ ê°�ì²´ì—�ì„œëŠ” (NumGraphsInBatch, 1, EXP_CF_DIM)ì�´ ë�¨.
    # EGNN ëª¨ë�¸ì�´ (NumGraphsInBatch, EXP_CF_DIM)ì�„ ê¸°ëŒ€í•œë‹¤ë©´ squeeze(1) í•„ìš”.
    if hasattr(batch, 'conditioning_feature') and batch.conditioning_feature is not None:
        if batch.conditioning_feature.dim() == 3 and batch.conditioning_feature.shape[1] == 1:
            batch.conditioning_feature = batch.conditioning_feature.squeeze(1)
        # ì°¨ì›� ë¶ˆì�¼ì¹˜ ì‹œ ê²½ê³  ë˜�ëŠ” ì—�ëŸ¬ ì²˜ë¦¬ ì¶”ê°€ ê°€ëŠ¥
    return batch

# 3.4) ë‹¨ìˆœí™”ë�œ DataLoader ìƒ�ì„± í•¨ìˆ˜
def make_simple_inference_loader(
    pyg_data_list: List[Data],
    batch_size: int = 32, # GPU ë©”ëª¨ë¦¬ ë°� ì¶”ë¡  íš¨ìœ¨ ê³ ë ¤í•˜ì—¬ ì„¤ì •
    num_workers: int = 0
) -> PyGDataLoader:
    if not pyg_data_list:
        print("Warning: DataLoader ìƒ�ì„±ì�„ ìœ„í•œ PyG ë�°ì�´í„° ë¦¬ìŠ¤íŠ¸ê°€ ë¹„ì–´ì�ˆìŠµë‹ˆë‹¤. Noneì�„ ë°˜í™˜í•©ë‹ˆë‹¤.")
        return None

    dataset = InMemoryGraphDataset(pyg_data_list)
    
    loader = PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # ì¶”ë¡  ì‹œì—�ëŠ” ì…”í”Œ ë¶ˆí•„ìš”
        num_workers=num_workers,
        pin_memory=False,
        #pin_memory=(DEVICE.type == 'cuda'),
        collate_fn=collate_graphs_for_egnn
    )
    print(f"ë‹¨ìˆœ ì¶”ë¡  DataLoader ìƒ�ì„± ì™„ë£Œ: {len(dataset)}ê°œ ê·¸ë�˜í”„, {len(loader)}ê°œ ë°°ì¹˜ (ë°°ì¹˜í�¬ê¸° {batch_size})")
    return loader

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ DataLoader ìƒ�ì„± ë°� í…ŒìŠ¤íŠ¸ (ì¶”ë¡ ìš©) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
inference_final_loader = None
if pyg_data_list_for_inference:
    inference_final_loader = make_simple_inference_loader(
        pyg_data_list_for_inference,
        batch_size=16 # ì˜ˆì‹œ ë°°ì¹˜ í�¬ê¸°, ì‹¤ì œ ì‚¬ìš© ì‹œ ì¡°ì •
    )

    if inference_final_loader and len(inference_final_loader) > 0:
        print("\nìƒ�ì„±ë�œ ì¶”ë¡  DataLoaderì�˜ ì²« ë²ˆì§¸ ë°°ì¹˜ ì •ë³´:")
        try:
            first_inference_batch = next(iter(inference_final_loader))
            # Data ê°�ì²´ë“¤ì�´ ì�´ë¯¸ DEVICEë¡œ ì˜®ê²¨ì¡Œìœ¼ë¯€ë¡œ, ë°°ì¹˜ëŠ” ì��ë�™ìœ¼ë¡œ í•´ë‹¹ DEVICEì—� ìƒ�ì„±ë�¨.
            # first_inference_batch = first_inference_batch.to(DEVICE) # í•„ìš”ì‹œ ëª…ì‹œì � ì�´ë�™
            print(first_inference_batch)
            print(f"  ë°°ì¹˜ ë‚´ ê·¸ë�˜í”„ ìˆ˜: {first_inference_batch.num_graphs}")
            if hasattr(first_inference_batch, 'conditioning_feature') and first_inference_batch.conditioning_feature is not None:
                print(f"  ë°°ì¹˜ conditioning_feature shape: {first_inference_batch.conditioning_feature.shape}")
            if hasattr(first_inference_batch, 'x') and first_inference_batch.x is not None:
                print(f"  ë°°ì¹˜ ë…¸ë“œ íŠ¹ì§•(x) shape: {first_inference_batch.x.shape}")
            if hasattr(first_inference_batch, 'pos') and first_inference_batch.pos is not None:
                print(f"  ë°°ì¹˜ ì¢Œí‘œ(pos) shape: {first_inference_batch.pos.shape}")
        except Exception as e_loader_test:
            print(f"DataLoader í…ŒìŠ¤íŠ¸ ì¤‘ ì˜¤ë¥˜: {e_loader_test}")
            import traceback
            traceback.print_exc()
    else:
        print("ì¶”ë¡  DataLoaderê°€ ë¹„ì–´ì�ˆê±°ë‚˜ ìƒ�ì„±ë�˜ì§€ ì•Šì•˜ìŠµë‹ˆë‹¤.")
else:
    print("PyG Data ë¦¬ìŠ¤íŠ¸ê°€ ë¹„ì–´ì�ˆì–´ DataLoaderë¥¼ ìƒ�ì„±í•  ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")


#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch_scatter import scatter_softmax, scatter_add
#import math

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Tiny Attention Block â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TinyAttn(nn.Module):
    """
    A compact attention mechanism.
    It computes scaled dot-product attention for given query, key, and value vectors,
    followed by a projection and a small MLP.
    """
    def __init__(self, dim: int, heads: int = 4, drop: float = 0.1):
        super().__init__()
        assert dim % heads == 0, f"Dimension ({dim}) must be divisible by heads ({heads})"
        self.h = heads # Number of attention heads
        self.d_head = dim // heads # Dimension of each head

        # Linear layer to project input to Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Output projection layer
        self.proj = nn.Linear(dim, dim)
        # Layer normalization
        self.ln = nn.LayerNorm(dim)
        # Small feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(), # Swish activation
            nn.Dropout(drop),
            nn.Linear(dim * 2, dim)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, h: torch.Tensor, src_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h (torch.Tensor): Input tensor of shape (E, dim), where E is number of edges/elements.
            src_idx (torch.Tensor): Source node indices for scatter_softmax, shape (E,).
                                    Used to group edges by their source node for softmax.
        Returns:
            torch.Tensor: Output tensor of shape (E, dim).
        """
        E, dim = h.shape # E = Number of elements (e.g., edges), dim = feature dimension

        # Project to Q, K, V and split
        q, k, v = self.qkv(h).chunk(3, dim=-1) # Each (E, dim)

        # Reshape for multi-head attention: (E, num_heads, head_dim)
        q = q.view(E, self.h, self.d_head)
        k = k.view(E, self.h, self.d_head)
        v = v.view(E, self.h, self.d_head)

        # Calculate scaled dot-product attention scores (logits)
        # (q * k) performs element-wise multiplication
        # .sum(dim=-1) sums across the head_dim
        logits = (q * k).sum(dim=-1) / math.sqrt(self.d_head) # Shape: (E, num_heads)

        # Apply softmax grouped by source node index to get attention weights
        attn_weights = scatter_softmax(logits, src_idx, dim=0) # Shape: (E, num_heads)

        # Apply attention weights to values
        # attn_weights.unsqueeze(-1) gives (E, num_heads, 1)
        # v is (E, num_heads, head_dim)
        # Result is (E, num_heads, head_dim)
        weighted_values = attn_weights.unsqueeze(-1) * v

        # Concatenate heads and reshape back to (E, dim)
        ctx = weighted_values.contiguous().view(E, dim)

        # Output projection, residual connection, and MLP block
        out = h + self.drop(self.proj(ctx)) # Apply projection and add residual
        out = out + self.drop(self.mlp(self.ln(out))) # Apply MLP block with residual
        return out

# â”€â”€â”€â”€â”€â”€â”€â”€â”€ DeepGate + Singleâ€‘Layer FiLM â”€â”€â”€â”€â”€â”€â”€â”€â”€
class DeepGateCrossFiLM(nn.Module):
    """
    DeepGate module with a *single* FiLM projection layer.
    It processes edge attributes through an up-stack of TinyAttn blocks,
    then modulates these features using a conditioning vector (global feature) via FiLM,
    followed by a tap-attention mechanism and a down-stack of TinyAttn blocks.
    The 'bert_dim' parameter is replaced by 'conditioning_feature_dim'.
    """

    def __init__(
        self,
        edge_dim: int = 15, # Dimension of input edge features
        up_dims: list[int] = [16, 32, 64, 128], # Dimensions for the up-stack attention blocks
        down_dims: list[int] = [128, 64, 32, 16, 8], # Dimensions for the down-stack attention blocks
        tap_dim: int = 128, # Dimension at the "tap" point (output of up-stack, input to FiLM and down-stack)
        conditioning_feature_dim: int = None, # Dimension of the new global conditioning feature
        bert_dim: int = None, # Old parameter name for backward compatibility (will be overridden by conditioning_feature_dim if both provided)
        mlp_out: int = 256, # Output dimension of the MLP that processes the conditioning feature for FiLM
        attn_heads: int = 4, # Number of attention heads in TinyAttn blocks
        attn_drop: float = 0.1, # Dropout rate in TinyAttn blocks
        eps: float = 1e-8, # Epsilon for numerical stability
        **_ignored, # Allows for backward-compatibility with existing instantiation arguments
    ):
        super().__init__()
        assert tap_dim == up_dims[-1], "tap_dim must match the last dimension in up_dims"
        assert tap_dim == down_dims[0], "tap_dim must match the first dimension in down_dims"
        for d_val in up_dims + down_dims: # Check divisibility for all attention block dimensions
            assert d_val % attn_heads == 0, f"Dimension {d_val} is not divisible by heads {attn_heads}"

        # Determine the final dimension for the conditioning feature
        final_cond_dim = 768 # Default, e.g., if no dimension is specified
        if conditioning_feature_dim is not None:
            final_cond_dim = conditioning_feature_dim
        elif bert_dim is not None: # Check for old parameter name if new one isn't provided
            final_cond_dim = bert_dim
            print(f"Warning: Argument 'bert_dim' (value: {bert_dim}) is deprecated for DeepGateCrossFiLM. "
                  f"Please use 'conditioning_feature_dim'. The dimension has been set to {final_cond_dim}.")

        self.conditioning_dim_val = final_cond_dim # Store the actual dimension used for conditioning
        self.eps = eps

        # --- Upâ€‘stack: Processes initial edge features ---
        self.in_proj = nn.Linear(edge_dim, up_dims[0]) # Initial projection of edge features
        self.up_blocks = nn.ModuleList()
        self.up_proj = nn.ModuleList() # Linear projections between up-stack blocks
        for i, d_val in enumerate(up_dims):
            self.up_blocks.append(TinyAttn(dim=d_val, heads=attn_heads, drop=attn_drop))
            if i < len(up_dims) - 1: # Add projection if not the last block
                self.up_proj.append(nn.Linear(d_val, up_dims[i + 1]))

        # --- Singleâ€‘layer MLPâ€‘FiLM: Modulates features using the conditioning vector ---
        # MLP to process the conditioning vector
        self.mlp_film = nn.Linear(self.conditioning_dim_val, mlp_out)
        self.ln_mlp_film = nn.LayerNorm(mlp_out)
        self.swish_film = nn.SiLU()

        # Linear layer to generate gamma and beta for FiLM from the processed conditioning vector
        self.to_gamma_beta = nn.Linear(mlp_out, tap_dim * 2) # tap_dim for gamma, tap_dim for beta

        # LayerNorm for the features being conditioned by FiLM
        self.ln_film_cond_target = nn.LayerNorm(tap_dim)

        # --- Tapâ€‘Attention: Computes attention weights based on FiLM-conditioned features ---
        self.Wq_tap = nn.Linear(tap_dim, tap_dim) # Query projection for tap-attention
        self.Wk_tap = nn.Linear(tap_dim, tap_dim) # Key projection for tap-attention

        # --- Downâ€‘stack: Further processes FiLM-conditioned and tap-attended features ---
        self.down_blocks = nn.ModuleList()
        self.down_proj = nn.ModuleList() # Linear projections between down-stack blocks
        for i, d_val in enumerate(down_dims):
            self.down_blocks.append(TinyAttn(dim=d_val, heads=attn_heads, drop=attn_drop))
            if i < len(down_dims) - 1: # Add projection if not the last block
                self.down_proj.append(nn.Linear(d_val, down_dims[i + 1]))

        # Final projection to get scalar gating weights (phi)
        self.to_phi = nn.Sequential(
            nn.LayerNorm(down_dims[-1]),
            nn.Linear(down_dims[-1], 1)
        )

    def forward(self, edge_attr: torch.Tensor, edge_index: torch.Tensor,
                conditioning_vec: torch.Tensor, edge_batch: torch.Tensor):
        """
        Args:
            edge_attr (torch.Tensor): Edge features, shape (num_edges, edge_dim).
            edge_index (torch.Tensor): Edge connectivity, shape (2, num_edges).
            conditioning_vec (torch.Tensor): Global conditioning vector for the batch,
                                             shape (num_graphs_in_batch, conditioning_dim_val).
            edge_batch (torch.Tensor): Maps each edge to its graph index in the batch, shape (num_edges,).
        Returns:
            torch.Tensor: Gating weights phi multiplied by tap-attention alpha, shape (num_edges,).
        """
        src_nodes = edge_index[0] # Source nodes for each edge

        # --- Upâ€‘stack ---
        h_up = self.in_proj(edge_attr) # (E, up_dims[0])
        for i, block in enumerate(self.up_blocks):
            h_up = block(h_up, src_nodes)
            if i < len(self.up_proj):
                h_up = self.up_proj[i](h_up)
        h_tapped = h_up  # Features at the tap point, shape (E, tap_dim)

        # --- FiLM Conditioning ---
        # Expand graph-level conditioning_vec to edge-level
        # conditioning_vec is (num_graphs, cond_dim), edge_batch is (E,)
        # b_expanded will be (E, cond_dim)
        b_expanded = conditioning_vec[edge_batch]

        # Process conditioning vector through MLP
        processed_b = self.swish_film(self.ln_mlp_film(self.mlp_film(b_expanded))) # (E, mlp_out)

        # Generate gamma and beta for FiLM
        gamma, beta = self.to_gamma_beta(processed_b).chunk(2, dim=-1) # Each (E, tap_dim)

        # Apply FiLM: h_film = gamma * h_tapped + beta
        # Add residual connection after FiLM and LayerNorm
        h_film_modulated = gamma * h_tapped + beta
        h_conditioned_by_film = h_tapped + self.swish_film(self.ln_film_cond_target(h_film_modulated)) # (E, tap_dim)

        # --- Tapâ€‘Attention ---
        # Use FiLM-conditioned features for Tap-Attention queries and keys
        q_tap = self.Wq_tap(h_conditioned_by_film) # (E, tap_dim)
        k_tap = self.Wk_tap(h_conditioned_by_film) # (E, tap_dim)

        # Calculate tap-attention logits
        alpha_logits = (q_tap * k_tap).sum(dim=-1) / math.sqrt(h_conditioned_by_film.size(-1)) # (E,)
        # Apply scatter_softmax to get attention weights per source node
        alpha_tap = scatter_softmax(alpha_logits, src_nodes, dim=0) # (E,)

        # --- Downâ€‘stack ---
        # Input to down-stack is the FiLM-conditioned features
        h_down = h_conditioned_by_film
        for i, block in enumerate(self.down_blocks):
            h_down = block(h_down, src_nodes)
            if i < len(self.down_proj):
                h_down = self.down_proj[i](h_down)

        # Project to scalar gating weights phi
        phi_gate_weights = self.to_phi(h_down).squeeze(-1) # (E,)

        # Final output: element-wise product of tap-attention weights and gating weights
        return alpha_tap * phi_gate_weights # (E,)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ EGNNCore â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EGNNCore(nn.Module):
    """
    Core Equivariant Graph Neural Network (EGNN) layer.
    Updates node positions based on messages passed along edges, gated by DeepGateCrossFiLM.
    """
    def __init__(self, gate_module: DeepGateCrossFiLM, eps: float = 1e-8):
        super().__init__()
        self.gate = gate_module # The gating mechanism (DeepGateCrossFiLM)
        self.eps  = eps # Epsilon for numerical stability

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, conditioning_vec: torch.Tensor,
                edge_batch: torch.Tensor):
        """
        Args:
            pos (torch.Tensor): Node positions, shape (num_nodes, 3).
            edge_index (torch.Tensor): Edge connectivity, shape (2, num_edges).
            edge_attr (torch.Tensor): Edge features, shape (num_edges, edge_dim).
            conditioning_vec (torch.Tensor): Global conditioning vector for the batch,
                                             shape (num_graphs_in_batch, conditioning_dim_val).
            edge_batch (torch.Tensor): Maps each edge to its graph index, shape (num_edges,).
        Returns:
            torch.Tensor: Updated node positions, shape (num_nodes, 3).
        """
        src_nodes, dst_nodes = edge_index # Source and destination nodes for each edge

        # Calculate gating weights using the DeepGate module
        # These weights determine the influence of each edge message
        gate_weights = self.gate(edge_attr, edge_index, conditioning_vec, edge_batch) # (num_edges,)

        # Calculate difference vectors between source and destination node positions
        pos_diff = pos[src_nodes] - pos[dst_nodes] # (num_edges, 3)

        # Normalize difference vectors to get unit direction vectors
        norm_pos_diff = pos_diff.norm(dim=-1, keepdim=True).clamp(min=self.eps) # (num_edges, 1)
        unit_pos_diff = pos_diff / norm_pos_diff # (num_edges, 3)

        # Calculate message updates for node positions
        # Messages are scaled unit vectors, weighted by gate_weights
        # gate_weights.unsqueeze(-1) makes it (num_edges, 1) for broadcasting
        messages = gate_weights.unsqueeze(-1) * unit_pos_diff # (num_edges, 3)

        # Aggregate messages for each node using scatter_add
        # Sum messages for all edges pointing to the same source node (or could be destination, depending on convention)
        # Here, messages are aggregated at the source_nodes.
        # dim_size=pos.size(0) ensures the output tensor has a size for all nodes, even isolated ones.
        delta_pos_updates = scatter_add(messages, src_nodes, dim=0, dim_size=pos.size(0)) # (num_nodes, 3)

        # Update node positions by adding the aggregated messages
        return pos + delta_pos_updates

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ EGNN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class EGNN(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) model.
    Applies EGNNCore for a specified number of steps to refine node positions.
    Edge attributes are dynamically updated based on current node positions in each step.
    """
    def __init__(self, n_steps: int = 1, **kwargs): # kwargs will catch conditioning_feature_dim or bert_dim
        super().__init__()
        # Instantiate the gating module, passing all kwargs (including conditioning_feature_dim)
        gate_module = DeepGateCrossFiLM(**kwargs)
        self.core    = EGNNCore(gate_module=gate_module, eps=gate_module.eps)
        self.n_steps = n_steps # Number of EGNN update steps
        self.eps     = gate_module.eps # Epsilon from the gate module

    def forward(self, batch):
        """
        Args:
            batch: PyTorch Geometric Batch object containing graph data.
                   Expected attributes: pos, edge_index, edge_attr, conditioning_feature, batch (for edge_batch).
        Returns:
            torch.Tensor: Final node positions after n_steps of updates.
        """
        pos           = batch.pos # Initial node positions
        edge_index    = batch.edge_index # Edge connectivity
        initial_edge_attr = batch.edge_attr # Initial edge attributes

        # Use batch.conditioning_feature instead of batch.bert
        if not hasattr(batch, 'conditioning_feature'):
            raise AttributeError("Batch object must have a 'conditioning_feature' attribute for EGNN.")
        conditioning_vec = batch.conditioning_feature # Global conditioning vector for the batch

        # `batch.batch` maps each node to its graph index in the batch.
        # `edge_batch` maps each edge to its graph index.
        # This is derived from the source node of each edge.
        edge_batch_map    = batch.batch[edge_index[0]]

        if edge_index.numel() == 0: # Handle cases with no edges (e.g., single-node graphs)
            return pos

        # Structure of edge_attr (total 15 dim assumed by default in rna_graph_knn.py):
        #   - oh_i (4), oh_j (4), backbone (1)  -> First 9 features (static_A)
        #   - delta_coords (3)                  -> Features 9, 10, 11 (dynamically updated)
        #   - distance_val (1)                  -> Feature 12 (dynamically updated)
        #   - resid_i_norm (1), resid_j_norm (1) -> Last 2 features (static_B)
        # Slicing indices for static parts of edge_attr:
        static_part_A = initial_edge_attr[:, :9]  # One-hot encodings and backbone flag
        static_part_B = initial_edge_attr[:, 13:] # Normalized residue indices

        current_pos = pos # Node positions to be updated in each step

        for step in range(self.n_steps):
            if step == 0:
                # Use initial edge attributes for the first step
                current_edge_attr = initial_edge_attr
            else:
                # Dynamically update edge attributes based on current node positions
                src_nodes, dst_nodes = edge_index
                new_delta_coords  = current_pos[src_nodes] - current_pos[dst_nodes] # (E, 3)
                new_distance_val   = new_delta_coords.norm(dim=-1, keepdim=True).clamp(min=self.eps) # (E, 1)
                # Reconstruct edge_attr with new delta and distance
                current_edge_attr = torch.cat([static_part_A, new_delta_coords, new_distance_val, static_part_B], dim=-1)

            # Apply the EGNNCore update
            current_pos = self.core(current_pos, edge_index, current_edge_attr,
                                    conditioning_vec, edge_batch_map)
        return current_pos

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ê°€ì¤‘ì¹˜ ì´ˆê¸°í™” â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def init_weights(module: nn.Module):
    """
    Initializes weights for nn.Linear and nn.LayerNorm modules.
    Uses Kaiming uniform for Linear weights and zeros for biases.
    Uses ones for LayerNorm weights and zeros for biases.
    """
    if isinstance(module, nn.Linear):
        # Check if weight exists and is not None before initialization
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5)) # Kaiming uniform for weights
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias) # Zeros for biases
    elif isinstance(module, nn.LayerNorm):
        if hasattr(module, 'weight') and module.weight is not None :
            nn.init.ones_(module.weight) # Ones for LayerNorm weights
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias) # Zeros for LayerNorm biases




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

egnn_model = EGNN(
    n_steps                 = 3,
    edge_dim                = 15,
    up_dims                 = [16, 32, 64, 128],
    down_dims               = [128, 64, 32, 16, 8],
    tap_dim                 = 128,
    conditioning_feature_dim = 768,  # 'bert_dim'ì�„ 'conditioning_feature_dim'ìœ¼ë¡œ ë³€ê²½
    mlp_out                 = 256,
    attn_heads              = 4,
    attn_drop               = 0.1,
    eps                     = 1e-8,
).to(device)


def unstandardize_coordinates(coords_std_tensor, means_dict, stds_dict):
    """
    Z-score í‘œì¤€í™”ë�œ ì¢Œí‘œ í…�ì„œë¥¼ ì›�ë�˜ ìŠ¤ì¼€ì�¼(ì„¼í„°ë§�ë�œ ìƒ�íƒœ)ë¡œ ë�˜ë�Œë¦½ë‹ˆë‹¤.
    Args:
        coords_std_tensor: Tensor of shape (SeqLen,3) or (N,SeqLen,3), Z-score í‘œì¤€í™”ë�œ ê°’
        means_dict: {'x':â€¦, 'y':â€¦, 'z':â€¦}
        stds_dict:  {'x':â€¦, 'y':â€¦, 'z':â€¦}
    Returns:
        Tensor in same shape as coords_std_tensor, but ì›�ë�˜ ìŠ¤ì¼€ì�¼(ì„¼í„°ë§�ë�œ ìƒ�íƒœ)ë¡œ ë³µì›�ë�¨.
    """
    single = coords_std_tensor.ndim == 2
    if single:
        coords = coords_std_tensor.unsqueeze(0)  # (1,SeqLen,3)
    else:
        coords = coords_std_tensor            # (N,SeqLen,3)

    device = coords.device
    means = torch.tensor([[means_dict['x'], means_dict['y'], means_dict['z']]],
                         dtype=torch.float32, device=device)  # (1,1,3)
    stds  = torch.tensor([[stds_dict['x'], stds_dict['y'], stds_dict['z']]],
                         dtype=torch.float32, device=device)  # (1,1,3)

    # ì—­ë³€í™˜: x_original = x_std * std + mean
    restored = coords * stds + means

    if single:
        return restored.squeeze(0)  # (SeqLen,3)
    return restored  # (N,SeqLen,3)



#import json # JSON ë¡œë“œ (ìŠ¤ì¼€ì�¼ëŸ¬ íŒŒë�¼ë¯¸í„°)
#import numpy as np
#import pandas as pd
#import torch
# from tqdm.auto import tqdm # ì�´ë¯¸ ìœ„ì—�ì„œ import ë�˜ì—ˆì�„ ìˆ˜ ì�ˆì�Œ




# EGNN ëª¨ë�¸ ì�¸ìŠ¤í„´ìŠ¤ ìƒ�ì„± ë°� ê°€ì¤‘ì¹˜ ë¡œë“œ
print("\nEGNN ëª¨ë�¸ì�„ ìƒ�ì„±í•˜ê³  ê°€ì¤‘ì¹˜ë¥¼ ë¡œë“œí•©ë‹ˆë‹¤...")

egnn_model.eval()
# <<<< ì¤‘ìš”: ì‹¤ì œ í•™ìŠµë�œ EGNN ê°€ì¤‘ì¹˜ íŒŒì�¼ ê²½ë¡œë¡œ ìˆ˜ì •í•˜ì„¸ìš” >>>>
egnn_model_weights_path = "/kaggle/input/weight/pytorch/default/1/3layer_knn_1dRMAE9align_svd_MAE_best_model_epoch_12.pth" # ì˜ˆì‹œ ê²½ë¡œ
if os.path.exists(egnn_model_weights_path):
    
    try:
        egnn_model.load_state_dict(torch.load(egnn_model_weights_path, map_location=DEVICE))
        print(f"EGNN ëª¨ë�¸ ê°€ì¤‘ì¹˜ë¥¼ ì„±ê³µì �ìœ¼ë¡œ ë¡œë“œí–ˆìŠµë‹ˆë‹¤: {egnn_model_weights_path}")
    except Exception as e_load:
         print(f"EGNN ëª¨ë�¸ ê°€ì¤‘ì¹˜ ë¡œë“œ ì¤‘ ì˜¤ë¥˜ ë°œìƒ� ({egnn_model_weights_path}): {e_load}. ì´ˆê¸°í™”ë�œ ê°€ì¤‘ì¹˜ë¥¼ ì‚¬ìš©í•©ë‹ˆë‹¤.")
else:
    print(f"Warning: EGNN ëª¨ë�¸ ê°€ì¤‘ì¹˜ íŒŒì�¼({egnn_model_weights_path})ì�„ ì°¾ì�„ ìˆ˜ ì—†ìŠµë‹ˆë‹¤. ì´ˆê¸°í™”ë�œ ê°€ì¤‘ì¹˜ë¡œ ì¶”ë¡ í•©ë‹ˆë‹¤.")
egnn_model.eval()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ìŠ¤ì¼€ì�¼ëŸ¬ íŒŒë�¼ë¯¸í„° ë¡œë“œ (Z-score ì—­ë³€í™˜ìš©) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# <<<< ì¤‘ìš”: ì‹¤ì œ ìŠ¤ì¼€ì�¼ëŸ¬ íŒŒë�¼ë¯¸í„° íŒŒì�¼ ê²½ë¡œë¡œ ìˆ˜ì •í•˜ì„¸ìš” >>>>
SCALER_PARAMS_PATH = "/kaggle/input/data-for-egnn/coordinate_scaler_params_gb_feature.v2.json" # ì˜ˆì‹œ ê²½ë¡œ
loaded_means, loaded_stds = load_scaler_params(SCALER_PARAMS_PATH) # ì�´ í•¨ìˆ˜ëŠ” ì�´ì „ ì½”ë“œì—� ì •ì�˜ë�˜ì–´ ì�ˆì–´ì•¼ í•¨


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ EGNN ì¶”ë¡  ë°� ê²°ê³¼ ìˆ˜ì§‘ (Z-score ì—­ë³€í™˜ë§Œ ì �ìš©) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nEGNN ëª¨ë�¸ ì¶”ë¡ ì�„ ì‹œì�‘í•©ë‹ˆë‹¤...")
# ê²°ê³¼ë¥¼ RNA IDë³„, ìƒ˜í”Œ ë²ˆí˜¸ë³„ë¡œ ì €ì�¥ (ê°’: (SeqLen, 3) NumPy ë°°ì—´ - ì„¼í„°ë§�ë�œ ìŠ¤ì¼€ì�¼, Z-score ì—­ë³€í™˜ë�¨)
rna_id_to_final_coords = {}

if inference_final_loader: # DataLoaderê°€ ì„±ê³µì �ìœ¼ë¡œ ìƒ�ì„±ë�˜ì—ˆë‹¤ë©´
    # <<<< ì¤‘ìš”: egnn_modelì�´ ì •ì�˜ë�˜ê³  ê°€ì¤‘ì¹˜ê°€ ë¡œë“œë�˜ì—ˆë‹¤ê³  ê°€ì •í•©ë‹ˆë‹¤. >>>>
    # if 'egnn_model' not in locals():
    #     print("Error: EGNN ëª¨ë�¸('egnn_model')ì�´ ì •ì�˜ë�˜ì§€ ì•Šì•˜ìŠµë‹ˆë‹¤. ì�„ì‹œ ë�”ë¯¸ ëª¨ë�¸ì�„ ì‚¬ìš©í•©ë‹ˆë‹¤.")
    #     # ì�„ì‹œ ë�”ë¯¸ EGNN ëª¨ë�¸ (ì‹¤ì œ ëª¨ë�¸ë¡œ êµ�ì²´ í•„ìˆ˜)
    #     class DummyEGNN(torch.nn.Module):
    #         def __init__(self): super().__init__(); self.fc = torch.nn.Linear(3,3)
    #         def forward(self, data): return data.pos + torch.randn_like(data.pos) * 0.01 # ì�…ë ¥ posì—� ì•½ê°„ì�˜ ë…¸ì�´ì¦ˆ
    #     egnn_model = DummyEGNN().to(DEVICE)
    # egnn_model.eval()


    with torch.no_grad():
        for batch_graph_data in tqdm(inference_final_loader, desc="EGNN ëª¨ë�¸ ì¶”ë¡  ì¤‘"):
            batch_graph_data = batch_graph_data.to(DEVICE)

            # EGNN ëª¨ë�¸ ì¶”ë¡  (ì¶œë ¥ì�€ "ì„¼í„°ë§�ë�œ ìƒ�íƒœì—�ì„œ Z-score í‘œì¤€í™”ë�œ" ì¢Œí‘œë�¼ê³  ê°€ì •)
            corrected_pos_batch_std_centered = egnn_model(batch_graph_data) # (TotalNodesInBatch, 3)

            data_list_from_batch = batch_graph_data.to_data_list()
            current_node_idx_in_batch = 0
            for single_graph_data_from_batch in data_list_from_batch:
                num_nodes = single_graph_data_from_batch.num_nodes

                pred_coords_std_centered_single = corrected_pos_batch_std_centered[
                    current_node_idx_in_batch : current_node_idx_in_batch + num_nodes
                ]
                current_node_idx_in_batch += num_nodes

                graph_id_full = single_graph_data_from_batch.id # "originalRNAid_sampleIdx"
                original_rna_id_key, sample_idx_str = graph_id_full.rsplit('_', 1)

                # Z-score ì—­ë³€í™˜ë§Œ ìˆ˜í–‰ (ê²°ê³¼ëŠ” ì—¬ì „í�ˆ ì„¼í„°ë§�ë�œ ìƒ�íƒœì�˜ ì›�ë�˜ ìŠ¤ì¼€ì�¼)
                pred_coords_centered_original_scale = unstandardize_coordinates( # ì�´ í•¨ìˆ˜ëŠ” ì�´ì „ ì½”ë“œì—� ì •ì�˜ë�¨
                    pred_coords_std_centered_single, loaded_means, loaded_stds
                )

                if original_rna_id_key not in rna_id_to_final_coords:
                    rna_id_to_final_coords[original_rna_id_key] = [None] * 5 # 5ê°œ ìƒ˜í”Œ ê³µê°„
                try:
                    sample_idx_one_based = int(sample_idx_str)
                    if 1 <= sample_idx_one_based <= 5:
                        rna_id_to_final_coords[original_rna_id_key][sample_idx_one_based - 1] = pred_coords_centered_original_scale.cpu().numpy()
                    else:
                        print(f"Warning: ID {graph_id_full}ì�˜ sample_idx({sample_idx_one_based})ê°€ ìœ íš¨ ë²”ìœ„ë¥¼ ë²—ì–´ë‚¨.")
                except ValueError:
                     print(f"Warning: ID {graph_id_full}ì�˜ sample_idx_str ('{sample_idx_str}') ë³€í™˜ ë¶ˆê°€.")
else:
    print("ìƒ�ì„±ë�œ DataLoaderê°€ ì—†ì–´ EGNN ì¶”ë¡ ì�„ ê±´ë„ˆëœ�ë‹ˆë‹¤.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ ìµœì¢… Submission CSV íŒŒì�¼ ìƒ�ì„± â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("\nìµœì¢… Submission CSV íŒŒì�¼ì�„ ìƒ�ì„±í•©ë‹ˆë‹¤...")
final_submission_rows = []

# test_dataëŠ” ì›�ë³¸ test_sequences.csvë¥¼ ë¡œë“œí•œ Pandas DataFrameì�´ì–´ì•¼ í•©ë‹ˆë‹¤.
# ì�´ì „ì—� Script Bì—�ì„œ test_dataë¥¼ ë¡œë“œí–ˆìœ¼ë¯€ë¡œ í•´ë‹¹ ë³€ìˆ˜ë¥¼ ì‚¬ìš©í•©ë‹ˆë‹¤.
if 'test_data' not in locals() or not isinstance(test_data, pd.DataFrame) or test_data.empty:
    print("Error: 'test_data' DataFrameì�´ ë¹„ì–´ì�ˆê±°ë‚˜ ì •ì�˜ë�˜ì§€ ì•Šì•˜ìŠµë‹ˆë‹¤. Submission CSVë¥¼ ìƒ�ì„±í•  ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")
    print("'/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'ì—�ì„œ test_dataë¥¼ ë¡œë“œí•˜ë ¤ê³  ì‹œë�„í•©ë‹ˆë‹¤.")
    test_data_path_example = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
    if os.path.exists(test_data_path_example):
        test_data = pd.read_csv(test_data_path_example)
        print(f"'{test_data_path_example}'ì—�ì„œ test_dataë¥¼ ì„±ê³µì �ìœ¼ë¡œ ë¡œë“œí–ˆìŠµë‹ˆë‹¤.")
    else:
        print(f"Error: '{test_data_path_example}'ë¥¼ ì°¾ì�„ ìˆ˜ ì—†ì–´ test_dataë¥¼ ë¡œë“œí•  ìˆ˜ ì—†ìŠµë‹ˆë‹¤. CSV ìƒ�ì„±ì�„ ì¤‘ë‹¨í•©ë‹ˆë‹¤.")
        test_data = pd.DataFrame() # ë¹ˆ DataFrameìœ¼ë¡œ ì„¤ì •í•˜ì—¬ ì•„ë�˜ ë£¨í”„ë¥¼ ê±´ë„ˆë›°ë�„ë¡� í•¨

if not test_data.empty:
    for i in range(len(test_data)):
        target_id_from_csv = test_data.loc[i, 'target_id']
        sequence_from_csv = test_data.loc[i, 'sequence']
        seq_len = len(sequence_from_csv)

        predicted_coord_sets_for_this_rna = rna_id_to_final_coords.get(target_id_from_csv)

        if predicted_coord_sets_for_this_rna is None:
            print(f"Warning: RNA ID {target_id_from_csv} EGNN ì˜ˆì¸¡ ì—†ì�Œ. NaN ì¢Œí‘œ ì‚¬ìš©.")
            predicted_coord_sets_for_this_rna = [np.full((seq_len, 3), np.nan)] * 5

        for j_residue_idx in range(seq_len):
            row_for_csv = [
                f"{target_id_from_csv}_{j_residue_idx + 1}",
                sequence_from_csv[j_residue_idx],
                j_residue_idx + 1
            ]
            for k_sample_idx in range(5): # 5ê°œ ì˜ˆì¸¡ ìƒ˜í”Œì—� ëŒ€í•´
                coords_for_this_sample = None
                # predicted_coord_sets_for_this_rnaì�˜ ê¸¸ì�´ê°€ 5ì�´ê³ , ê°� ìš”ì†Œê°€ ë°°ì—´ ë˜�ëŠ” Noneì�¼ ìˆ˜ ì�ˆì�Œ
                if k_sample_idx < len(predicted_coord_sets_for_this_rna) and \
                   predicted_coord_sets_for_this_rna[k_sample_idx] is not None:
                    coords_for_this_sample = predicted_coord_sets_for_this_rna[k_sample_idx]

                if coords_for_this_sample is not None and \
                   isinstance(coords_for_this_sample, np.ndarray) and \
                   coords_for_this_sample.shape == (seq_len, 3) and \
                   j_residue_idx < coords_for_this_sample.shape[0] and \
                   not np.isnan(coords_for_this_sample[j_residue_idx]).any(): # ìœ íš¨í•œ ì¢Œí‘œì�¸ì§€ í™•ì�¸
                    row_for_csv.extend(coords_for_this_sample[j_residue_idx])
                else:
                    row_for_csv.extend([np.nan, np.nan, np.nan])
            final_submission_rows.append(row_for_csv)

    submission_columns = ['ID', 'resname', 'resid']
    for i_sample_num in range(1, 6): # 1ë¶€í„° 5ê¹Œì§€
        submission_columns.extend([f"x_{i_sample_num}", f"y_{i_sample_num}", f"z_{i_sample_num}"])

    submission_df_final = pd.DataFrame(final_submission_rows, columns=submission_columns)
    submission_output_path = 'submission.csv' # ìµœì¢… íŒŒì�¼ ì�´ë¦„
    submission_df_final.to_csv(submission_output_path, index=False)
    print(f"\nâœ… ìµœì¢… Submission íŒŒì�¼ ìƒ�ì„± ì™„ë£Œ: {submission_output_path}")
    print("Submission íŒŒì�¼ ìƒ˜í”Œ:")
    print(submission_df_final.head())
else:
    print("test_data DataFrameì�´ ë¹„ì–´ì�ˆê±°ë‚˜ ë¡œë“œë�˜ì§€ ì•Šì•„ Submission CSVë¥¼ ìƒ�ì„±í•  ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")


submission


import numpy as np

def kabsch_align(P, Q):
    """
    Kabsch ì•Œê³ ë¦¬ì¦˜ìœ¼ë¡œ P (ì˜ˆì¸¡ ì¢Œí‘œ)ë¥¼ Q (ì‹¤ì œ ì¢Œí‘œ)ì—� ì •ë ¬í•©ë‹ˆë‹¤.

    Args:
        P (np.ndarray): ì˜ˆì¸¡ ì¢Œí‘œ, shape (N, 3)
        Q (np.ndarray): ì‹¤ì œ ì¢Œí‘œ, shape (N, 3)

    Returns:
        np.ndarray: Qì—� ì •ë ¬ë�œ P ì¢Œí‘œ, shape (N, 3)
    """
    assert P.shape == Q.shape, "Pì™€ QëŠ” ë�™ì�¼í•œ shapeì�´ì–´ì•¼ í•©ë‹ˆë‹¤."
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    C = np.dot(P_cent.T, Q_cent)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(V, D), Wt)
    P_aligned = np.dot(P_cent, U) + Q.mean(axis=0)
    return P_aligned



def visualize_ground_truth_coords_with_backbone_only(
    rna_id,
    validation_csv_path="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
):
    """
    íŠ¹ì • RNA IDì�˜ ì‹¤ì œ êµ¬ì¡°ë¥¼ ë°±ë³¸ ì„ ê³¼ í•¨ê»˜ ì‹œê°�í™”í•©ë‹ˆë‹¤.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    df = pd.read_csv(validation_csv_path)
    df_rna = df[df["ID"].str.startswith(rna_id)]
    if df_rna.empty:
        print(f"â�Œ RNA ID {rna_id}ì—� ëŒ€í•œ ground truth ë�°ì�´í„°ê°€ ì—†ìŠµë‹ˆë‹¤.")
        return

    df_rna = df_rna.sort_values("resid")
    coords = df_rna[['x_1', 'y_1', 'z_1']].dropna().values
    if coords.shape[0] < 2:
        print("âš ï¸� ì¢Œí‘œ ìˆ˜ ë¶€ì¡±ìœ¼ë¡œ ë°±ë³¸ ì‹œê°�í™” ë¶ˆê°€.")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color='red', linewidth=1.5, label='Ground Truth Backbone')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='red', s=20, alpha=0.7)

    ax.set_title(f"Ground Truth RNA Structure with Backbone\n{rna_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()



def visualize_predicted_coords_with_backbone_aligned(
    rna_id,
    sample_idx=0,
    validation_csv_path="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
):
    """
    íŠ¹ì • RNA IDì�˜ ì˜ˆì¸¡ ì¢Œí‘œë¥¼ Kabsch ì •ë ¬ í›„ ë°±ë³¸ í�¬í•¨ ì‹œê°�í™”í•©ë‹ˆë‹¤.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # ì‹¤ì œ ì¢Œí‘œ ë¶ˆëŸ¬ì˜¤ê¸°
    df = pd.read_csv(validation_csv_path)
    df_rna = df[df["ID"].str.startswith(rna_id)]
    if df_rna.empty:
        print(f"â�Œ RNA ID {rna_id}ì—� ëŒ€í•œ validation ì¢Œí‘œê°€ ì—†ìŠµë‹ˆë‹¤.")
        return

    df_rna = df_rna.sort_values("resid")
    true_coords = df_rna[['x_1', 'y_1', 'z_1']].dropna().values

    # ì˜ˆì¸¡ ì¢Œí‘œ ê°€ì ¸ì˜¤ê¸°
    pred_coords = rna_id_to_final_coords.get(rna_id, [None]*5)[sample_idx]
    if pred_coords is None or pred_coords.shape != true_coords.shape:
        print(f"â�Œ ì˜ˆì¸¡ ì¢Œí‘œê°€ ì—†ê±°ë‚˜ shape ë¶ˆì�¼ì¹˜: {rna_id}")
        return

    # Kabsch ì •ë ¬ ì �ìš©
    aligned_pred = kabsch_align(pred_coords, true_coords)

    # ì‹œê°�í™”
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
            color='blue', linewidth=1.5, label=f'Predicted (Aligned)')
    ax.scatter(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
               color='blue', s=20, alpha=0.7)

    ax.set_title(f"Predicted RNA Structure with Backbone\n{rna_id} (Sample {sample_idx + 1})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()



# ì‹¤ì œ êµ¬ì¡° ì‹œê°�í™”
visualize_ground_truth_coords_with_backbone_only("R1107")

# ì˜ˆì¸¡ êµ¬ì¡° (Kabsch ì •ë ¬ í�¬í•¨)
visualize_predicted_coords_with_backbone_aligned("R1107", sample_idx=0)



def visualize_kabsch_aligned_prediction_vs_ground_truth(
    rna_id,
    sample_idx=0,
    validation_csv_path="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
):
    """
    Kabsch ì •ë ¬ì�„ ì �ìš©í•œ ì˜ˆì¸¡ êµ¬ì¡° vs ì‹¤ì œ êµ¬ì¡°ë¥¼ ë°±ë³¸ ì„  í�¬í•¨í•´ ì‹œê°�í™”í•©ë‹ˆë‹¤.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    df = pd.read_csv(validation_csv_path)
    df_rna = df[df["ID"].str.startswith(rna_id)]
    if df_rna.empty:
        print(f"â�Œ RNA ID {rna_id}ì—� ëŒ€í•œ validation ì¢Œí‘œê°€ ì—†ìŠµë‹ˆë‹¤.")
        return

    df_rna = df_rna.sort_values("resid")
    true_coords = df_rna[['x_1', 'y_1', 'z_1']].dropna().values

    pred_coords = rna_id_to_final_coords.get(rna_id, [None]*5)[sample_idx]
    if pred_coords is None or pred_coords.shape != true_coords.shape:
        print(f"â�Œ ì˜ˆì¸¡ ì¢Œí‘œê°€ ì—†ê±°ë‚˜ shapeì�´ ì�¼ì¹˜í•˜ì§€ ì•ŠìŠµë‹ˆë‹¤: {rna_id}")
        return

    # ğŸ‘‰ Kabsch ì •ë ¬
    aligned_pred = kabsch_align(pred_coords, true_coords)

    # ì‹œê°�í™”
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
            color='red', linewidth=1.5, label='Ground Truth Backbone')
    ax.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
               color='red', s=20, alpha=0.7)

    ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
            color='blue', linewidth=1.5, label=f'Predicted (Kabsch-aligned)')
    ax.scatter(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
               color='blue', s=20, alpha=0.7)

    ax.set_title(f"Kabsch-aligned RNA Structure Comparison\n{rna_id} (Sample {sample_idx + 1})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()



visualize_kabsch_aligned_prediction_vs_ground_truth("R1107", sample_idx=0)



def visualize_predicted_coords_with_labels_and_arrows(
    rna_id,
    sample_idx=0,
    validation_csv_path="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
):
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # validation ì‹¤ì œ ì¢Œí‘œ ë¡œë”© (ì •ë ¬ìš©)
    df = pd.read_csv(validation_csv_path)
    df_rna = df[df["ID"].str.startswith(rna_id)].sort_values("resid")
    true_coords = df_rna[['x_1', 'y_1', 'z_1']].dropna().values

    pred_coords = rna_id_to_final_coords.get(rna_id, [None]*5)[sample_idx]
    if pred_coords is None or pred_coords.shape != true_coords.shape:
        print(f"â�Œ ì¢Œí‘œ ë¶ˆì�¼ì¹˜ ë˜�ëŠ” ì—†ì�Œ: {rna_id}")
        return

    # Kabsch ì •ë ¬
    aligned_pred = kabsch_align(pred_coords, true_coords)
    N = aligned_pred.shape[0]

    # ì‹œê°�í™”
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ë°±ë³¸ ì„  + ì �
    ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
            color='blue', linewidth=1.2, label='Predicted Backbone')
    ax.scatter(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
               color='blue', s=20, alpha=0.8)

    # ë²ˆí˜¸ ë�¼ë²¨ ì¶”ê°€
    for i in range(N):
        x, y, z = aligned_pred[i]
        ax.text(x, y, z, f"{i+1}", size=6, color='black', alpha=0.8)

    # í™”ì‚´í‘œ ë°©í–¥ í‘œì‹œ (quiver ì‚¬ìš©)
    for i in range(N - 1):
        start = aligned_pred[i]
        direction = aligned_pred[i + 1] - aligned_pred[i]
        ax.quiver(
            start[0], start[1], start[2],
            direction[0], direction[1], direction[2],
            color='blue', linewidth=0.5, arrow_length_ratio=0.5, alpha=1.0
        )

    ax.set_title(f"Predicted RNA Structure with Backbone and Direction\n{rna_id} (Sample {sample_idx + 1})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()



visualize_predicted_coords_with_labels_and_arrows("R1107", sample_idx=0)



def visualize_ground_truth_coords_with_labels_and_arrows(
    rna_id,
    validation_csv_path="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
):
    """
    ì‹¤ì œ RNA êµ¬ì¡°ë¥¼ ë°±ë³¸ ì„ , residue ë²ˆí˜¸ ë�¼ë²¨, ë°©í–¥ í™”ì‚´í‘œì™€ í•¨ê»˜ ì‹œê°�í™”í•©ë‹ˆë‹¤.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    df = pd.read_csv(validation_csv_path)
    df_rna = df[df["ID"].str.startswith(rna_id)].sort_values("resid")
    true_coords = df_rna[['x_1', 'y_1', 'z_1']].dropna().values

    if true_coords.shape[0] < 2:
        print(f"â�Œ RNA ID {rna_id}ì�˜ ì‹¤ì œ ì¢Œí‘œê°€ ë¶€ì¡±í•©ë‹ˆë‹¤.")
        return

    N = true_coords.shape[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
            color='red', linewidth=1.5, label='Ground Truth Backbone')
    ax.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
               color='red', s=20, alpha=0.8)

    # residue ë²ˆí˜¸ ë�¼ë²¨
    for i in range(N):
        x, y, z = true_coords[i]
        ax.text(x, y, z, f"{i+1}", size=6, color='black', alpha=0.7)

    # ë°©í–¥ í™”ì‚´í‘œ
    for i in range(N - 1):
        start = true_coords[i]
        direction = true_coords[i + 1] - true_coords[i]
        ax.quiver(
            start[0], start[1], start[2],
            direction[0], direction[1], direction[2],
            color='red', linewidth=0.5, arrow_length_ratio=0.5, alpha=1.0
        )

    ax.set_title(f"Ground Truth RNA Structure with Backbone\n{rna_id}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()



visualize_ground_truth_coords_with_labels_and_arrows("R1107")



def visualize_prediction_vs_ground_truth_with_backbone_and_arrows(
    rna_id,
    sample_idx=0,
    validation_csv_path="/kaggle/input/stanford-rna-3d-folding/validation_labels.csv"
):
    """
    íŠ¹ì • RNA IDì—� ëŒ€í•´ ì˜ˆì¸¡ ë°� ì‹¤ì œ êµ¬ì¡°ë¥¼ í•¨ê»˜ ì‹œê°�í™”í•©ë‹ˆë‹¤.
    ë°±ë³¸ ì„ , residue ë²ˆí˜¸, ë°©í–¥ í™”ì‚´í‘œ í�¬í•¨.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    df = pd.read_csv(validation_csv_path)
    df_rna = df[df["ID"].str.startswith(rna_id)].sort_values("resid")
    true_coords = df_rna[['x_1', 'y_1', 'z_1']].dropna().values

    pred_coords = rna_id_to_final_coords.get(rna_id, [None]*5)[sample_idx]
    if pred_coords is None or pred_coords.shape != true_coords.shape:
        print(f"â�Œ ì¢Œí‘œ ë¶ˆì�¼ì¹˜ ë˜�ëŠ” ì—†ì�Œ: {rna_id}")
        return

    aligned_pred = kabsch_align(pred_coords, true_coords)
    N = aligned_pred.shape[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ì‹¤ì œ êµ¬ì¡°
    ax.plot(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
            color='red', linewidth=1.5, label='Ground Truth Backbone')
    ax.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2],
               color='red', s=20, alpha=0.7)

    # ì˜ˆì¸¡ êµ¬ì¡°
    ax.plot(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
            color='blue', linewidth=1.5, label=f'Predicted Backbone (Sample {sample_idx+1})')
    ax.scatter(aligned_pred[:, 0], aligned_pred[:, 1], aligned_pred[:, 2],
               color='blue', s=20, alpha=0.7)

    # residue ë²ˆí˜¸ (ì˜ˆì¸¡ ê¸°ì¤€)
    for i in range(N):
        x, y, z = aligned_pred[i]
        ax.text(x, y, z, f"{i+1}", size=6, color='black', alpha=0.6)

    # ë°©í–¥ í™”ì‚´í‘œ: ì˜ˆì¸¡ (íŒŒë�€ í™”ì‚´í‘œ)
    for i in range(N - 1):
        start = aligned_pred[i]
        direction = aligned_pred[i + 1] - aligned_pred[i]
        ax.quiver(
            start[0], start[1], start[2],
            direction[0], direction[1], direction[2],
            color='blue', linewidth=0.5, arrow_length_ratio=0.2, alpha=1.0
        )

    # ë°©í–¥ í™”ì‚´í‘œ: ì‹¤ì œ (íšŒìƒ‰ í™”ì‚´í‘œ)
    for i in range(N - 1):
        start = true_coords[i]
        direction = true_coords[i + 1] - true_coords[i]
        ax.quiver(
            start[0], start[1], start[2],
            direction[0], direction[1], direction[2],
            color='red', linewidth=0.5, arrow_length_ratio=0.5, alpha=1.0
        )

    ax.set_title(f"Predicted vs Ground Truth RNA Structure\n{rna_id} (Sample {sample_idx + 1})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()



visualize_prediction_vs_ground_truth_with_backbone_and_arrows("R1107", sample_idx=0)

