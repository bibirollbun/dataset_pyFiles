# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
import pandas as pd

path = "/kaggle/input/movie-recomendation-fall-2020/train.txt"  # đổi nếu cần

rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= 3:
            rows.append([int(nums[0]), int(nums[1]), int(nums[2])])

df_train = pd.DataFrame(rows, columns=["userid", "movieid", "rating"])
print(df_train.head())
print(df_train.dtypes)


path = "/kaggle/input/movie-recomendation-fall-2020/test.txt"  # đổi nếu cần

rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        nums = re.findall(r"-?\d+", line)
        if len(nums) >= 2:
            rows.append([int(nums[0]), int(nums[1])])

df_test = pd.DataFrame(rows, columns=["userid", "movieid"])
print(df_test.head())
print(df_test.dtypes)


import torch
import torch.nn.functional as F
from torch import nn
from typing import Sequence, List
class NeuMF(nn.Module):
    """
    NeuMF với one-hot embedding:
      - User/Item -> one-hot (cố định) -> Linear chiếu sang latent (thay cho nn.Embedding)
      - Nhánh GMF: element-wise product giữa 2 latent vector
      - Nhánh MLP: concat 2 latent, qua nhiều Dense + ReLU (+ Dropout)
      - Fusion: concat(GMF, MLP) -> Dropout -> (Dropout head) -> Linear(…, 1)

    Tham số dropout:
      - dropout_proj  : dropout ngay sau chiếu one-hot -> latent (cả GMF và MLP)
      - dropout_hidden: dropout sau mỗi Dense của tháp MLP
      - dropout_fusion: dropout sau khi concat GMF & MLP
      - dropout_fc    : dropout ngay trước lớp Linear cuối cùng
    """
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 k_gmf: int = 8,
                 k_mlp: int = 32,
                 mlp_layers=(64, 32, 16),
                 dropout_hidden: float = 0.3,
                 dropout_proj: float = 0.1,
                 dropout_fusion: float = 0.2,
                 dropout_fc: float = 0.2):
        super().__init__()

        # One-hot "embeddings": identity, freeze để dùng như one-hot lookup
        self.oh_user = nn.Embedding.from_pretrained(torch.eye(num_users), freeze=True)
        self.oh_item = nn.Embedding.from_pretrained(torch.eye(num_items), freeze=True)

        # Chiếu one-hot -> latent (thay cho nn.Embedding)
        self.gmf_user_proj = nn.Linear(num_users, k_gmf, bias=False)
        self.gmf_item_proj = nn.Linear(num_items, k_gmf, bias=False)
        self.mlp_user_proj = nn.Linear(num_users, k_mlp, bias=False)
        self.mlp_item_proj = nn.Linear(num_items, k_mlp, bias=False)

        # Dropout sau projection
        self.do_proj_gmf_u = nn.Dropout(dropout_proj)
        self.do_proj_gmf_i = nn.Dropout(dropout_proj)
        self.do_proj_mlp_u = nn.Dropout(dropout_proj)
        self.do_proj_mlp_i = nn.Dropout(dropout_proj)

        # Tháp MLP: (Linear -> ReLU -> Dropout) * len(mlp_layers)
        mlp = []
        in_dim = k_mlp * 2
        for units in mlp_layers:
            mlp += [
                nn.Linear(in_dim, units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_hidden),
            ]
            in_dim = units
        self.mlp_layers = nn.Sequential(*mlp)

        # Dropout sau khi concat(GMF, MLP)
        self.do_fusion = nn.Dropout(dropout_fusion)

        # Head: Dropout trước fc
        fusion_dim = k_gmf + (mlp_layers[-1] if mlp_layers else k_mlp * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(fusion_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier cho các Linear
        for m in [self.gmf_user_proj, self.gmf_item_proj, self.mlp_user_proj, self.mlp_item_proj]:
            nn.init.xavier_uniform_(m.weight)
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        lin = self.head[1]  # Linear cuối
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        users: LongTensor shape (B,)
        items: LongTensor shape (B,)
        return: FloatTensor shape (B,)  — điểm số (rating dự đoán)
        """
        # One-hot vectors (0/1)
        u_oh = self.oh_user(users)   # (B, num_users)
        i_oh = self.oh_item(items)   # (B, num_items)

        # ----- GMF path -----
        gu = self.gmf_user_proj(u_oh)      # (B, k_gmf)
        gi = self.gmf_item_proj(i_oh)      # (B, k_gmf)
        gu = self.do_proj_gmf_u(gu)
        gi = self.do_proj_gmf_i(gi)
        gmf = gu * gi                      # element-wise

        # ----- MLP path -----
        mu = self.mlp_user_proj(u_oh)      # (B, k_mlp)
        mi = self.mlp_item_proj(i_oh)      # (B, k_mlp)
        mu = self.do_proj_mlp_u(mu)
        mi = self.do_proj_mlp_i(mi)
        x = torch.cat([mu, mi], dim=-1)    # (B, 2*k_mlp)
        x = self.mlp_layers(x)

        # ----- Fusion + head -----
        z = torch.cat([gmf, x], dim=-1)
        z = self.do_fusion(z)
        score = self.head(z).squeeze(-1)   # (B,)
        return score


import torch
import torch.nn.functional as F
from torch import nn
from typing import Sequence, List

class DMF(nn.Module):
    """
    Deep Matrix Factorization (học trực tiếp A):
      - A: (num_users, num_items) là tham số học được.
      - forward(users, items):
          + user vector = A[u, :]  (1 hàng)
          + item vector = A[:, v]  (1 cột)
          + qua MLP riêng -> L2-normalize -> cosine -> map [1,5]
    """
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 d1: int = 64,                    # (không dùng ở bản này)
                 hidden: Sequence[int] = (64, 32, 16),
                 dropout: float = 0.2,
                 use_bn: bool = False):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items

        # Ma trận A học trực tiếp (khởi tạo nhỏ quanh 0 để điểm ~3 sau khi map)
        self.A = nn.Parameter(torch.randn(num_users, num_items) * 0.01)

        # Tower MLP: user nhận vectơ kích thước num_items; item nhận vectơ kích thước num_users
        def make_mlp(in_dim):
            layers = []
            last = in_dim
            for h in hidden:
                layers += [nn.Linear(last, h)]
                if use_bn:
                    layers += [nn.BatchNorm1d(h)]
                layers += [nn.ReLU(inplace=True), nn.Dropout(dropout)]
                last = h
            return nn.Sequential(*layers), last

        self.user_mlp, Du = make_mlp(num_items)
        self.item_mlp, Dv = make_mlp(num_users)
        assert Du == Dv, "Hai tower phải có cùng output dim để tính cosine."
        self.out_dim = Du
        self.head = nn.Linear(self.out_dim * 2, 1)   # học cách ghép pu & qv -> score
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Khởi tạo tuyến tính
        for m in list(self.user_mlp.modules()) + list(self.item_mlp.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # --------------------------- forward (giống Neu) ---------------------------
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        users/items: LongTensor (B,)
        return: FloatTensor (B,) in [1, 5]
        """
        # Lấy hàng/cột từ A
        rows = self.A[users, :]                   # (B, num_items)
        cols = self.A[:, items].transpose(0, 1)   # (B, num_users)
    
        # Qua MLP hai nhánh
        pu = self.user_mlp(rows) if len(self.user_mlp) else rows   # (B, D)
        qv = self.item_mlp(cols) if len(self.item_mlp) else cols   # (B, D)
    
        # (tuỳ chọn) normalize nhẹ, có thể bỏ nếu muốn để Linear tự học
        # pu = F.normalize(pu, p=2, dim=-1)
        # qv = F.normalize(qv, p=2, dim=-1)
    
        # Ghép và cho qua Linear head
        z = torch.cat([pu, qv], dim=-1)           # (B, 2D)
        s = self.head(z).squeeze(-1)              # (B,)
    
        # Ràng buộc đầu ra về [1, 5] bằng sigmoid thay vì clamp
        
        return s

    



import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple

class LightGCN(nn.Module):
    """
    LightGCN tối giản cho implicit CF.
    - E^(0) = concat([E_user, E_item]) với shape [(M+N), d] (user trước, item sau).
    - Propagation: E^(k+1) = A_tilde @ E^(k), A_tilde = D^{-1/2} A D^{-1/2}.
    - Layer-combine: E_final = sum_{k=0..K} alpha_k * E^(k), mặc định alpha_k = 1/(K+1).
    - Score(u,i) = <e_u, e_i>.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        edges: Optional[torch.LongTensor] = None,   # shape [2, E], (user_id, item_id)
        alpha: Optional[torch.Tensor] = None,       # (K+1,), nếu None -> uniform
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # E^(0): embedding riêng user & item
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Alpha cho trộn tầng
        no_edges = (edges is None) or (edges.numel() == 0)
        if alpha is None:
            if no_edges:
                alpha = torch.zeros(num_layers + 1, dtype=dtype); alpha[0] = 1.0
            else:
                alpha = torch.full((num_layers + 1,), 1.0/(num_layers + 1), dtype=dtype)
        else:
            assert alpha.numel() == num_layers + 1, "alpha phải có K+1 phần tử"
            alpha = alpha.to(dtype=dtype)
        self.register_buffer("alpha", alpha)

        # A_tilde (sparse)
        if no_edges:
            num_nodes = num_users + num_items
            A_tilde = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.tensor([], dtype=dtype),
                (num_nodes, num_nodes),
                device=device,
                dtype=dtype
            ).coalesce()
        else:
            A_tilde = self._build_norm_adj(edges, device=device, dtype=dtype)

        # Lưu buffer chỉ số & giá trị để tái tạo nhanh mỗi lần propagate
        self.register_buffer("A_tilde_indices", A_tilde.indices())
        self.register_buffer("A_tilde_values",  A_tilde.values())
        self.A_tilde_size = A_tilde.size()

    # ---------- helpers ----------
    def _build_norm_adj(
        self,
        edges: torch.LongTensor,
        device: Optional[torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Xây A_tilde = D^{-1/2} A D^{-1/2} cho đồ thị hai phía (vô hướng)."""
        M, N = self.num_users, self.num_items
        if device is None:
            device = edges.device

        u = edges[0].to(torch.long)               # [E]
        i = (edges[1].to(torch.long) + M)         # shift item id: [E] trong [M, M+N)

        # cạnh vô hướng: (u,i) & (i,u)
        src = torch.cat([u, i])
        dst = torch.cat([i, u])
        indices = torch.stack([src, dst], dim=0)  # [2, 2E]
        values = torch.ones(indices.size(1), dtype=dtype, device=device)
        num_nodes = M + N

        A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes),
                                    device=device, dtype=dtype).coalesce()

        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)  # tránh 0
        deg_inv_sqrt = deg.pow(-0.5)
        row, col = A.indices()
        norm_vals = deg_inv_sqrt[row] * A.values() * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(A.indices(), norm_vals, A.size(),
                                       device=device, dtype=dtype).coalesce()

    def _E0(self) -> torch.Tensor:
        """E^(0) = concat([E_user, E_item]) với shape [(M+N), d]."""
        return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

    def _propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lan truyền K bước và trộn tầng. Trả về (E_user_final, E_item_final)."""
        E0 = self._E0()
        Es = [E0]

        if self.A_tilde_values.numel() > 0:
            A_tilde = torch.sparse_coo_tensor(
                self.A_tilde_indices, self.A_tilde_values, self.A_tilde_size,
                device=E0.device, dtype=E0.dtype
            )
            Ek = E0
            for _ in range(self.num_layers):
                Ek = torch.sparse.mm(A_tilde, Ek)
                Es.append(Ek)
        else:
            # không cạnh → các tầng sau = 0, alpha[0]=1 ⇒ E_final = E0
            for _ in range(self.num_layers):
                Es.append(torch.zeros_like(E0))

        E_final = torch.zeros_like(E0)
        for k, Ek in enumerate(Es):
            E_final = E_final + self.alpha[k] * Ek

        return E_final[: self.num_users], E_final[self.num_users :]

    # ---------- public API ----------
    @torch.no_grad()
    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về (E_user_final [M,d], E_item_final [N,d])."""
        return self._propagate()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        users, items: LongTensor [B]
        Trả về: scores [B] = <e_u, e_i>
        """
        U, I = self._propagate()
        eu = U[users]            # [B, d]
        ei = I[items]            # [B, d]
        return (eu * ei).sum(dim=1)

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> torch.Tensor:
        """
        Pairwise BPR loss; regularize chỉ E^(0) như LightGCN gốc.
        users, pos_items, neg_items: LongTensor [B]
        """
        U, I = self._propagate()
        eu = U[users]
        ei = I[pos_items]
        ej = I[neg_items]
        y_pos = (eu * ei).sum(dim=1)
        y_neg = (eu * ej).sum(dim=1)
        loss = -torch.nn.functional.logsigmoid(y_pos - y_neg).mean()

        # L2 chỉ trên E^(0)
        reg = (
            self.user_emb.weight.norm(p=2).pow(2)
            + self.item_emb.weight.norm(p=2).pow(2)
        ) / (self.num_users + self.num_items)

        return loss + l2_reg * reg



import torch
import torch.nn as nn

class BiasMF(nn.Module):
    """
    Biased Matrix Factorization:
      r_hat(u,i) = mu + b_u + b_i + <p_u, q_i>
    Train bằng MSELoss như bạn đang làm.
    """
    def __init__(self, num_users: int, num_items: int, k: int = 64, use_global_bias: bool = True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, k)
        self.item_emb = nn.Embedding(num_items, k)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.global_bias = nn.Parameter(torch.zeros(1)) if use_global_bias else None

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        pu = self.user_emb(users)              # [B,k]
        qi = self.item_emb(items)              # [B,k]
        dot = (pu * qi).sum(dim=1)             # [B]
        bu = self.user_bias(users).squeeze(1)  # [B]
        bi = self.item_bias(items).squeeze(1)  # [B]
        out = dot + bu + bi
        if self.global_bias is not None:
            out = out + self.global_bias
        return out



import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# (4) Wide & Deep (drop-in: forward(u,i)->rating)
# =========================================================
class WideDeep(nn.Module):
    """
    Wide: mu + b_u + b_i (+ optional linear terms)
    Deep: concat([e_u, e_i]) -> MLP -> scalar
    """
    def __init__(self, num_users, num_items, emb_dim=64, mlp_layers=(128,64,32), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Wide terms
        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        # Deep MLP
        in_dim = 2 * emb_dim
        layers = []
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, users, items):
        eu = self.user_emb(users)
        ei = self.item_emb(items)
        deep = self.mlp(torch.cat([eu, ei], dim=1)).squeeze(1)

        out = deep
        if self.use_bias:
            out = out + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias
        return out


class NFM(nn.Module):
    """
    NFM = (linear terms) + MLP( bi-interaction vector )
    Với 2 field (user,item): bi-interaction vector ~ eu ⊙ ei
    """
    def __init__(self, num_users, num_items, emb_dim=64, mlp_layers=(128,64,32), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        # MLP on interaction vector
        in_dim = emb_dim
        layers = []
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, users, items):
        eu = self.user_emb(users)   # [B,d]
        ei = self.item_emb(items)   # [B,d]
        inter_vec = eu * ei         # [B,d]  (bi-interaction)
        deep = self.mlp(inter_vec).squeeze(1)

        out = deep
        if self.use_bias:
            out = out + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias
        return out


class _CIN2Fields(nn.Module):
    """
    CIN cho đúng 2 field (user,item):
      X0: [B, F=2, D]
      Mỗi layer tạo feature maps H_l: [B, H, D]
    """
    def __init__(self, emb_dim, cin_layers=(16, 16)):
        super().__init__()
        self.emb_dim = emb_dim
        self.cin_layers = cin_layers

        # For 2 fields: outer product between Xk (Hk fields) and X0 (2 fields)
        # We use conv1d-like weight to mix the 2*Hk interaction maps -> H_next
        self.W = nn.ModuleList()
        H_prev = 2  # initial fields =2
        for H_next in cin_layers:
            self.W.append(nn.Linear(2 * H_prev, H_next, bias=True))
            H_prev = H_next

    def forward(self, X0):
        # X0: [B,2,D]
        B, F, D = X0.shape
        Xk = X0
        outputs = []

        for layer, proj in enumerate(self.W):
            # interaction: for each d, compute Xk[:,:,d] (B,Hprev) with X0[:,:,d] (B,2)
            # build Z: [B, (2*Hprev), D] by stacking pairwise products with X0
            # For 2 fields, product with each of the 2 base fields
            # Xk: [B,Hprev,D]
            Hprev = Xk.size(1)
            xk_d = Xk.permute(0, 2, 1)      # [B,D,Hprev]
            x0_d = X0.permute(0, 2, 1)      # [B,D,2]
            # products per base field: [B,D,Hprev] * [B,D,1] => [B,D,Hprev]
            p0 = xk_d * x0_d[:, :, 0:1]
            p1 = xk_d * x0_d[:, :, 1:2]
            Z = torch.cat([p0, p1], dim=2)  # [B,D,2*Hprev]
            Z = Z.permute(0, 2, 1)          # [B,2*Hprev,D]

            # mix 2*Hprev -> Hnext for each d
            Zt = Z.permute(0, 2, 1)         # [B,D,2*Hprev]
            Hnext = proj(Zt)                # [B,D,Hnext]
            Hnext = Hnext.permute(0, 2, 1)  # [B,Hnext,D]
            Xk = Hnext
            outputs.append(Xk)

        # pool over D and concat all cin layers: sum over D -> [B, sum(H_l)]
        pooled = [h.sum(dim=2) for h in outputs]
        return torch.cat(pooled, dim=1) if pooled else torch.zeros((B,0), device=X0.device)


class xDeepFM(nn.Module):
    def __init__(
        self,
        num_users, num_items,
        emb_dim=32,
        deep_layers=(128,64,32),
        cin_layers=(16,16),
        dropout=0.2,
        use_bias=True
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Linear terms (wide)
        self.use_bias = use_bias
        if use_bias:
            self.user_w = nn.Embedding(num_users, 1)
            self.item_w = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_w.weight)
            nn.init.zeros_(self.item_w.weight)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_w = None
            self.item_w = None
            self.global_bias = None

        # CIN
        self.cin = _CIN2Fields(emb_dim=emb_dim, cin_layers=cin_layers)
        cin_out_dim = sum(cin_layers)

        # Deep
        in_dim = 2 * emb_dim
        layers = []
        for h in deep_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.deep = nn.Sequential(*layers)
        deep_out_dim = deep_layers[-1] if len(deep_layers) else in_dim

        # Final combine
        self.fc = nn.Linear(cin_out_dim + deep_out_dim, 1)

    def forward(self, users, items):
        eu = self.user_emb(users)  # [B,D]
        ei = self.item_emb(items)  # [B,D]
        X0 = torch.stack([eu, ei], dim=1)  # [B,2,D]

        cin_feat = self.cin(X0)            # [B,sum(cin_layers)]
        deep_feat = self.deep(torch.cat([eu, ei], dim=1))  # [B, deep_out]

        out = self.fc(torch.cat([cin_feat, deep_feat], dim=1)).squeeze(1)

        if self.use_bias:
            out = out + self.user_w(users).squeeze(1) + self.item_w(items).squeeze(1) + self.global_bias
        return out


class NGCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, num_layers=2, edges=None, dropout=0.1, use_bias=True):
        super().__init__()
        self.U = num_users
        self.I = num_items
        self.d = emb_dim
        self.K = num_layers
        self.dropout = dropout

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.W1 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=True) for _ in range(num_layers)])
        self.W2 = nn.ModuleList([nn.Linear(emb_dim, emb_dim, bias=True) for _ in range(num_layers)])

        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

        # Build normalized adjacency
        if edges is None or edges.numel() == 0:
            N = num_users + num_items
            A = torch.sparse_coo_tensor(torch.zeros((2,0), dtype=torch.long),
                                        torch.tensor([], dtype=torch.float32),
                                        (N,N)).coalesce()
        else:
            A = build_norm_adj(num_users, num_items, edges, device=edges.device, dtype=torch.float32)

        self.register_buffer("A_idx", A.indices())
        self.register_buffer("A_val", A.values())
        self.A_size = A.size()

    def _propagate(self):
        E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # [N,d]
        A = torch.sparse_coo_tensor(self.A_idx, self.A_val, self.A_size, device=E0.device, dtype=E0.dtype)

        E = E0
        outs = [E0]
        for k in range(self.K):
            neigh = torch.sparse.mm(A, E)  # neighbor aggregation
            msg1 = self.W1[k](neigh)
            msg2 = self.W2[k](E * neigh)   # element-wise interaction (NGCF trick)
            E = F.leaky_relu(msg1 + msg2, negative_slope=0.2)
            E = F.dropout(E, p=self.dropout, training=self.training)
            outs.append(E)

        E_final = torch.stack(outs, dim=0).mean(dim=0)  # average layers
        Ue = E_final[:self.U]
        Ie = E_final[self.U:]
        return Ue, Ie

    def forward(self, users, items):
        Ue, Ie = self._propagate()
        eu = Ue[users]
        ei = Ie[items]
        out = (eu * ei).sum(dim=1)
        if self.use_bias:
            out = out + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResMLPBlock(nn.Module):
    """
    Residual MLP block:
      x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> +x
    """
    def __init__(self, dim: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class _TokenMixer(nn.Module):
    """
    MLP-Mixer token-mixing for 2 tokens (user token, item token).
    For 2 tokens, token-mixing is cheap: mix across token dimension.
    Input: [B, T=2, D]
    """
    def __init__(self, dim: int, token_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        # we mix tokens per channel: transpose to [B, D, T]
        self.fc1 = nn.Linear(2, token_hidden)
        self.fc2 = nn.Linear(token_hidden, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,2,D]
        h = self.ln(x)
        h = h.transpose(1, 2)              # [B,D,2]
        h = self.fc1(h)                    # [B,D,token_hidden]
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)                    # [B,D,2]
        h = self.drop(h)
        h = h.transpose(1, 2)              # [B,2,D]
        return x + h


class _ChannelMixer(nn.Module):
    """
    MLP-Mixer channel-mixing: per token MLP along D
    Input: [B,2,D]
    """
    def __init__(self, dim: int, channel_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, channel_hidden)
        self.fc2 = nn.Linear(channel_hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class _MixerBlock(nn.Module):
    """
    One Mixer block = TokenMixer + ChannelMixer
    """
    def __init__(self, dim: int, token_hidden: int, channel_hidden: int, dropout: float):
        super().__init__()
        self.token_mixer = _TokenMixer(dim, token_hidden=token_hidden, dropout=dropout)
        self.channel_mixer = _ChannelMixer(dim, channel_hidden=channel_hidden, dropout=dropout)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


class ResMLPRec(nn.Module):
    """
    Drop-in Recommender (rating regression):
      - Embedding user & item -> 2 tokens [B,2,D]
      - K Mixer blocks (token mixing + channel mixing)
      - Pool -> residual MLP head -> scalar rating

    forward(users, items) -> [B] float
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int = 128,
        mixer_layers: int = 3,
        token_hidden: int = 32,
        channel_hidden: int = 256,
        head_hidden: int = 256,
        head_layers: int = 2,
        dropout: float = 0.2,
        use_bias: bool = True,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # Mixer backbone
        self.mixers = nn.ModuleList([
            _MixerBlock(emb_dim, token_hidden=token_hidden, channel_hidden=channel_hidden, dropout=dropout)
            for _ in range(mixer_layers)
        ])

        # Pooling + head
        self.pre_head_ln = nn.LayerNorm(emb_dim)

        head_in = 2 * emb_dim  # concat pooled user-token + item-token
        blocks = []
        dim = head_in
        for _ in range(head_layers):
            blocks.append(_ResMLPBlock(dim=dim, hidden=head_hidden, dropout=dropout))
        self.head_blocks = nn.Sequential(*blocks)

        self.out = nn.Linear(head_in, 1)

        # Optional biases (giúp học nhanh như MF)
        self.use_bias = use_bias
        if use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            self.global_bias = nn.Parameter(torch.zeros(1))
        else:
            self.user_bias = None
            self.item_bias = None
            self.global_bias = None

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        eu = self.user_emb(users)  # [B,D]
        ei = self.item_emb(items)  # [B,D]

        x = torch.stack([eu, ei], dim=1)  # [B,2,D]

        for blk in self.mixers:
            x = blk(x)

        # pooling: keep both tokens, apply LN
        u_tok = self.pre_head_ln(x[:, 0, :])
        i_tok = self.pre_head_ln(x[:, 1, :])

        h = torch.cat([u_tok, i_tok], dim=1)  # [B,2D]
        h = self.head_blocks(h)
        pred = self.out(h).squeeze(1)

        if self.use_bias:
            pred = pred + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias

        return pred



# =========================
# (A) DCN: Deep & Cross Network (drop-in)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, cross_layers=3,
                 deep_layers=(128, 64, 32), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        d_in = 2 * emb_dim

        self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(d_in, 1) * 0.02) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(d_in)) for _ in range(cross_layers)])

        layers = []
        dim = d_in
        for h in deep_layers:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        self.deep = nn.Sequential(*layers)
        deep_out = deep_layers[-1] if len(deep_layers) else d_in

        self.fc = nn.Linear(d_in + deep_out, 1)

        self.use_bias = use_bias
        if use_bias:
            self.user_b = nn.Embedding(num_users, 1)
            self.item_b = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_b.weight)
            nn.init.zeros_(self.item_b.weight)
            self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        x0 = torch.cat([self.user_emb(users), self.item_emb(items)], dim=1)  # [B,2D]
        x = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xlw = x @ w          # [B,1]
            x = x0 * xlw + b + x # [B,2D]
        deep = self.deep(x0) if len(self.deep) else x0
        out = self.fc(torch.cat([x, deep], dim=1)).squeeze(1)
        if self.use_bias:
            out = out + self.user_b(users).squeeze(1) + self.item_b(items).squeeze(1) + self.global_b
        return out

# Call:
# model = DCN(num_users=num_users, num_items=num_items, emb_dim=64, cross_layers=3, deep_layers=(128,64,32), dropout=0.2).to(device)



# =========================
# (B) PNN: Product-based Neural Network (drop-in)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class PNN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64,
                 mlp_layers=(128, 64, 32), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        in_dim = 3 * emb_dim  # [eu, ei, eu⊙ei]
        layers = []
        dim = in_dim
        for h in mlp_layers:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self.mlp = nn.Sequential(*layers)

        self.use_bias = use_bias
        if use_bias:
            self.user_b = nn.Embedding(num_users, 1)
            self.item_b = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_b.weight)
            nn.init.zeros_(self.item_b.weight)
            self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        eu = self.user_emb(users)
        ei = self.item_emb(items)
        prod = eu * ei
        x = torch.cat([eu, ei, prod], dim=1)
        out = self.mlp(x).squeeze(1)
        if self.use_bias:
            out = out + self.user_b(users).squeeze(1) + self.item_b(items).squeeze(1) + self.global_b
        return out

# Call:
# model = PNN(num_users=num_users, num_items=num_items, emb_dim=64, mlp_layers=(128,64,32), dropout=0.2).to(device)



# =========================
# (C) AutoInt: Self-Attention Interaction (drop-in)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoInt(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, attn_layers=2,
                 n_heads=4, mlp_layers=(128, 64), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=max(128, 4 * emb_dim),
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=attn_layers)

        in_dim = 2 * emb_dim
        layers = []
        dim = in_dim
        for h in mlp_layers:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self.mlp = nn.Sequential(*layers)

        self.use_bias = use_bias
        if use_bias:
            self.user_b = nn.Embedding(num_users, 1)
            self.item_b = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_b.weight)
            nn.init.zeros_(self.item_b.weight)
            self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        eu = self.user_emb(users)
        ei = self.item_emb(items)
        x = torch.stack([eu, ei], dim=1)   # [B,2,D]
        x = self.encoder(x)               # [B,2,D]
        x = x.reshape(x.size(0), -1)      # [B,2D]
        out = self.mlp(x).squeeze(1)
        if self.use_bias:
            out = out + self.user_b(users).squeeze(1) + self.item_b(items).squeeze(1) + self.global_b
        return out

# Call:
# model = AutoInt(num_users=num_users, num_items=num_items, emb_dim=64, attn_layers=2, n_heads=4, mlp_layers=(128,64), dropout=0.2).to(device)



# =========================
# (D) FiBiNETLite: SE reweight + Bilinear interaction (drop-in)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiBiNETLite(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, se_ratio=4,
                 mlp_layers=(128, 64), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        hidden = max(4, emb_dim // se_ratio)
        self.se_fc1 = nn.Linear(emb_dim, hidden)
        self.se_fc2 = nn.Linear(hidden, 2)

        self.W = nn.Parameter(torch.randn(emb_dim, emb_dim) * 0.02)

        in_dim = 2 * emb_dim + 2
        layers = []
        dim = in_dim
        for h in mlp_layers:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self.mlp = nn.Sequential(*layers)

        self.use_bias = use_bias
        if use_bias:
            self.user_b = nn.Embedding(num_users, 1)
            self.item_b = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_b.weight)
            nn.init.zeros_(self.item_b.weight)
            self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        eu = self.user_emb(users)
        ei = self.item_emb(items)

        z = 0.5 * (eu + ei)
        a = F.relu(self.se_fc1(z))
        w = torch.sigmoid(self.se_fc2(a))   # [B,2]
        eu_se = eu * w[:, 0:1]
        ei_se = ei * w[:, 1:2]

        bil_raw = torch.sum((eu @ self.W) * ei, dim=1, keepdim=True)
        bil_se  = torch.sum((eu_se @ self.W) * ei_se, dim=1, keepdim=True)

        x = torch.cat([eu, ei, bil_raw, bil_se], dim=1)
        out = self.mlp(x).squeeze(1)
        if self.use_bias:
            out = out + self.user_b(users).squeeze(1) + self.item_b(items).squeeze(1) + self.global_b
        return out

# Call:
# model = FiBiNETLite(num_users=num_users, num_items=num_items, emb_dim=64, se_ratio=4, mlp_layers=(128,64), dropout=0.2).to(device)



# =========================
# (E) GateFM: Gated interaction (drop-in)
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class GateFM(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64,
                 gate_hidden=128, mlp_layers=(128, 64, 32), dropout=0.2, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.gate = nn.Sequential(
            nn.Linear(2 * emb_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, emb_dim),
            nn.Sigmoid(),
        )

        in_dim = 3 * emb_dim  # [eu, ei, gated(eu⊙ei)]
        layers = []
        dim = in_dim
        for h in mlp_layers:
            layers += [nn.Linear(dim, h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers += [nn.Linear(dim, 1)]
        self.mlp = nn.Sequential(*layers)

        self.use_bias = use_bias
        if use_bias:
            self.user_b = nn.Embedding(num_users, 1)
            self.item_b = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_b.weight)
            nn.init.zeros_(self.item_b.weight)
            self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        eu = self.user_emb(users)
        ei = self.item_emb(items)
        g = self.gate(torch.cat([eu, ei], dim=1))
        inter = (eu * ei) * g
        x = torch.cat([eu, ei, inter], dim=1)
        out = self.mlp(x).squeeze(1)
        if self.use_bias:
            out = out + self.user_b(users).squeeze(1) + self.item_b(items).squeeze(1) + self.global_b
        return out

# Call:
# model = GateFM(num_users=num_users, num_items=num_items, emb_dim=64, gate_hidden=128, mlp_layers=(128,64,32), dropout=0.2).to(device)



# =========================
# (F) FwFM: Field-weighted Factorization Machine (drop-in)
# =========================
import torch
import torch.nn as nn

class FwFM(nn.Module):
    """
    With 2 fields (user,item):
      score = w_ui * <eu, ei> + (bias terms)
    """
    def __init__(self, num_users, num_items, emb_dim=128, use_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.w_ui = nn.Parameter(torch.tensor(1.0))

        self.use_bias = use_bias
        if use_bias:
            self.user_b = nn.Embedding(num_users, 1)
            self.item_b = nn.Embedding(num_items, 1)
            nn.init.zeros_(self.user_b.weight)
            nn.init.zeros_(self.item_b.weight)
            self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, users, items):
        eu = self.user_emb(users)
        ei = self.item_emb(items)
        dot = (eu * ei).sum(dim=1)
        out = self.w_ui * dot
        if self.use_bias:
            out = out + self.user_b(users).squeeze(1) + self.item_b(items).squeeze(1) + self.global_b
        return out

# Call:
# model = FwFM(num_users=num_users, num_items=num_items, emb_dim=128).to(device)



# NeuMF in PyTorch: GMF + MLP for rating regression + AdamW + LR scheduler + save "Id,Score"
import os, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split   # <-- thêm import

# ================== Hyperparams & configs ==================
SEED = 42
SUB_PATH   = "submission.csv"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = 'cpu'
K_GMF = 8
K_MLP = 32
MLP_LAYERS = (32, 16, 8)
DROPOUT = 0.4
LR = 1e-3
EPOCHS = 100
BATCH_TRAIN = 1024
BATCH_TEST  = 1024
VAL_RATIO = 0.2
PATIENCE = 20
WEIGHT_DECAY = 1e-4  # AdamW weight decay

# ================== Assume df_train / df_test are ready ==================
# df_train: userid,movieid,rating ; df_test: userid,movieid

# ---- mapping ID -> index liên tục (từ cả train + test) ----
all_users = pd.Index(pd.concat([df_train['userid'], df_test['userid']]).unique())
all_items = pd.Index(pd.concat([df_train['movieid'], df_test['movieid']]).unique())
user2idx = {u:i for i,u in enumerate(all_users)}
item2idx = {m:i for i,m in enumerate(all_items)}
num_users = len(user2idx); num_items = len(item2idx)

df_train_idx = df_train.assign(
    user_idx = df_train['userid'].map(user2idx).astype('int64'),
    item_idx = df_train['movieid'].map(item2idx).astype('int64')
)
df_test_idx = df_test.assign(
    user_idx = df_test['userid'].map(user2idx).astype('int64'),
    item_idx = df_test['movieid'].map(item2idx).astype('int64')
)

min_r = float(df_train['rating'].min())
max_r = float(df_train['rating'].max())
global_mean = float(df_train['rating'].mean())
seen_users = set(df_train['userid'].unique())
seen_items = set(df_train['movieid'].unique())

# ================== Dataset / DataLoader ==================
class RatingsDS(Dataset):
    def __init__(self, u, i, y=None):
        self.u = torch.as_tensor(u, dtype=torch.long)
        self.i = torch.as_tensor(i, dtype=torch.long)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.u)
    def __getitem__(self, idx):
        if self.y is None:
            return self.u[idx], self.i[idx]
        return self.u[idx], self.i[idx], self.y[idx]

train_full = RatingsDS(
    df_train_idx['user_idx'].values,
    df_train_idx['item_idx'].values,
    df_train_idx['rating'].values,
)

all_idx = np.arange(len(df_train_idx))
train_idx, val_idx = train_test_split(
    all_idx,
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=df_train_idx['rating']
)

from torch.utils.data import Subset
train_ds = Subset(train_full, train_idx)
val_ds   = Subset(train_full, val_idx)

test_ds  = RatingsDS(df_test_idx['user_idx'].values, df_test_idx['item_idx'].values)

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_ds, batch_size=max(512, BATCH_TRAIN), shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False)

# ================== Model ==================
df_tr = df_train.copy()

# Nếu có rating và muốn lấy positive theo ngưỡng:
if 'rating' in df_tr.columns:
    df_tr = df_tr[df_tr['rating'] >= 3.0]   # hoặc >= 3.5, tuỳ bài

# Map sang chỉ số liên tục
df_tr['user_idx'] = df_tr['userid'].map(user2idx).astype('int64')
df_tr['item_idx'] = df_tr['movieid'].map(item2idx).astype('int64')

# Mỗi (user,item) một cạnh (LightGCN không cần multiple-edges)
df_tr = df_tr.drop_duplicates(subset=['user_idx', 'item_idx'])

u = torch.as_tensor(df_tr['user_idx'].values, dtype=torch.long)
i = torch.as_tensor(df_tr['item_idx'].values, dtype=torch.long)
edges = torch.stack([u, i], dim=0).to(device)   # shape [2, E]


# model = NeuMF(num_users, num_items, K_GMF, K_MLP, MLP_LAYERS, DROPOUT).to(device)
# model = DMF(num_users, num_items, d1=64, hidden=(64,32), dropout=0.3, use_bn=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3, edges=edges.to(device)).to(device)
# model = BiasMF(num_users=num_users, num_items=num_items, k=128).to(device)
# model = WideDeep(num_users=num_users, num_items=num_items, emb_dim=64, mlp_layers=(128,64,32), dropout=0.2).to(device)
# model = NFM(num_users=num_users, num_items=num_items, emb_dim=64, mlp_layers=(128,64,32), dropout=0.2).to(device)
# model = xDeepFM(num_users=num_users, num_items=num_items, emb_dim=32, deep_layers=(128,64,32), cin_layers=(16,16), dropout=0.2).to(device)
# model = NGCF(num_users=num_users, num_items=num_items, emb_dim=64, num_layers=2, edges=edges.to(device), dropout=0.1).to(device)
# model = ResMLPRec(num_users=num_users, num_items=num_items, emb_dim=128, mixer_layers=3, token_hidden=32, channel_hidden=256, head_hidden=256, head_layers=2, dropout=0.2, use_bias=True).to(device)
# model = DCN(num_users=num_users, num_items=num_items, emb_dim=64, cross_layers=3, deep_layers=(128,64,32), dropout=0.2).to(device)
# model = PNN(num_users=num_users, num_items=num_items, emb_dim=64, mlp_layers=(128,64,32), dropout=0.2).to(device)
# model = AutoInt(num_users=num_users, num_items=num_items, emb_dim=64, attn_layers=2, n_heads=4, mlp_layers=(128,64), dropout=0.2).to(device)
# model = FiBiNETLite(num_users=num_users, num_items=num_items, emb_dim=64, se_ratio=4, mlp_layers=(128,64), dropout=0.2).to(device)
# model = GateFM(num_users=num_users, num_items=num_items, emb_dim=64, gate_hidden=128, mlp_layers=(128,64,32), dropout=0.2).to(device)
model = FwFM(num_users=num_users, num_items=num_items, emb_dim=128).to(device)
# ================== Train ==================
device = next(model.parameters()).device
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
)

def rmse(y_true, y_pred):
    return math.sqrt(((y_true - y_pred) ** 2).mean().item())
from sklearn.metrics import r2_score

def r2_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """R^2 dùng sklearn."""
    return r2_score(y_true.detach().cpu().numpy(),
                    y_pred.detach().cpu().numpy())

best_val = float("inf"); patience = PATIENCE; bad = 0
best_state = None

for epoch in range(EPOCHS):
    # ----- train -----
    model.train()
    for u,i,y in tqdm(train_loader):
        u,i,y = u.to(device), i.to(device), y.to(device)
        optimizer.zero_grad()
        p = model(u,i)
        loss = criterion(p, y)
        loss.backward()
        optimizer.step()

    # ----- evaluate train -----
    model.eval()
    ys_tr, ps_tr = [], []
    with torch.no_grad():
        for u,i,y in train_loader:
            u,i,y = u.to(device), i.to(device), y.to(device)
            p = model(u,i)
            ys_tr.append(y.cpu()); ps_tr.append(p.cpu())
    y_tr = torch.cat(ys_tr); p_tr = torch.cat(ps_tr)
    train_rmse = rmse(y_tr, p_tr)
    train_r2   = r2_torch(y_tr, p_tr)

    # ----- validate -----
    ys_v, ps_v = [], []
    with torch.no_grad():
        for u,i,y in tqdm(val_loader):
            u,i,y = u.to(device), i.to(device), y.to(device)
            p = model(u,i)
            ys_v.append(y.cpu()); ps_v.append(p.cpu())
    y_val = torch.cat(ys_v); p_val = torch.cat(ps_v)
    val_rmse = rmse(y_val, p_val)
    val_r2   = r2_torch(y_val, p_val)

    scheduler.step(val_rmse)

    curr_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:02d} | "
          f"train RMSE: {train_rmse:.4f} | val RMSE: {val_rmse:.4f} | "
          f"train R2: {train_r2:.4f} | val R2: {val_r2:.4f} | LR: {curr_lr:.2e}")

    if val_rmse + 1e-6 < best_val:
        best_val = val_rmse; bad = 0
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break

# Load best
if best_state is not None:
    model.load_state_dict(best_state)
model.to(device); model.eval()

# ================== Predict test ==================
preds = []
with torch.no_grad():
    for u,i in test_loader:
        u,i = u.to(device), i.to(device)
        p = model(u,i).cpu().numpy()
        preds.append(p)
pred = np.concatenate(preds, axis=0).reshape(-1)

# clip về [min_r, max_r]
pred = np.clip(pred, min_r, max_r)

# fallback cho cold-start
mask_cold = (~df_test['userid'].isin(seen_users)) | (~df_test['movieid'].isin(seen_items))
if mask_cold.any():
    pred[mask_cold.values] = global_mean

# ================== Save "Id,Score" (Id bắt đầu từ 1) ==================
submission = pd.DataFrame({
    "Id": np.arange(1, len(df_test) + 1, dtype=int),
    "Score": pred.astype(float)
})
submission.to_csv(SUB_PATH, index=False)
print(f"Saved to: {os.path.abspath(SUB_PATH)}")
print(submission.head())


