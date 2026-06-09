# # nn lgb cat
# import os
# import gc
# import numpy as np
# import pandas as pd
# import polars as pls
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pytorch_lightning as pl
# import time

# import lightgbm as lgb
# from catboost import CatBoostRegressor

# # 若需要官方评测接口
# import kaggle_evaluation.jane_street_inference_server

# from torch.optim.lr_scheduler import ExponentialLR

# ##########################################################
# # 0) 全局配置
# ##########################################################
# ONLINE_LEARNING_PARAMS = {
#     "lr":          1e-5,
#     "weight_decay":5e-6,
#     "batch_size":  512,
#     "epochs":      1,
#     "lr_gamma":    0.99
# }

# # NN ckpt
# MODEL_CKPT_PATH = "/kaggle/input/nnlgbcat/other/default/1/offline_best.model (1).ckpt"

# # LGB & Cat
# LGB_MODEL_PATH = "/kaggle/input/nnlgbcat/other/default/1/lgb_model_single_0.txt"
# CAT_MODEL_PATH = "/kaggle/input/nnlgbcat/other/default/1/cat_symmetric_best.cbm"

# # NN特征(仅79列)
# NN_FEATURES_79 = [f"feature_{i:02d}" for i in range(79)]
# # LGB/Cat特征(79 + 9 lag + symbol_id => 89列)
# LGB_CAT_FEATS_89 = (
#     [f"feature_{i:02d}" for i in range(79)]
#     + [f"responder_{i}_lag_1" for i in range(9)]
#     + ["symbol_id"]
# )

# # 融合系数(仅示例)
# ALPHA_NN   = 0.60
# ALPHA_LGB  = 0.60

# # 是否假设行对齐(在合并NN和lags数据时用)
# ASSUME_ROW_ALIGNMENT = False

# ##########################################################
# # 1) 定义 NN(与离线训练一致), 用于在线学习
# ##########################################################
# class NN(pl.LightningModule):
#     def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
#         super().__init__()
#         self.save_hyperparameters()
#         layers = []
#         in_dim = input_dim
#         for i, hidden_dim in enumerate(hidden_dims):
#             layers.append(nn.BatchNorm1d(in_dim))
#             if i > 0:
#                 layers.append(nn.SiLU())
#             if i < len(dropouts):
#                 layers.append(nn.Dropout(dropouts[i]))
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             in_dim = hidden_dim

#         layers.append(nn.Linear(in_dim, 1))
#         layers.append(nn.Tanh())
#         self.model = nn.Sequential(*layers)

#         self.lr = lr
#         self.weight_decay = weight_decay

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return 5.0 * self.model(x).squeeze(-1)

# ##########################################################
# # 2) forward_fill_and_zero
# ##########################################################
# def forward_fill_and_zero(
#     df: pd.DataFrame,
#     sort_cols=["time_id"],
#     group_cols=["symbol_id"],
#     fill_cols=None
# ):
#     """
#     groupby(...) => ffill => fillna(0)
#     """
#     if fill_cols is None:
#         fill_cols = NN_FEATURES_79[:]
#         if "weight" in df.columns:
#             fill_cols.append("weight")

#     df = df.sort_values(group_cols + sort_cols, ignore_index=True)
#     df[fill_cols] = df.groupby(group_cols)[fill_cols].ffill()
#     df[fill_cols] = df[fill_cols].fillna(0)
#     return df

# ##########################################################
# # 3) 全局变量 => 3个模型(已加载) & 在线学习cache
# ##########################################################
# # （A）NN
# trained_nn      = None
# test_cache_nn   = {}
# first_date_nn   = None   # 用于跳过首日在线学习

# # （B）LGB, Cat
# lgb_model_single = None
# cat_model_single = None

# # （C）全局 lags_ 用于拼接 9 lag 特征 (Polars方式)
# lags_ = None

# ##########################################################
# # 4) 在脚本加载阶段，就把 3 个模型初始化好
# ##########################################################
# def load_lgb_model_txt(model_file: str) -> lgb.Booster:
#     """加载 LightGBM txt 模型"""
#     with open(model_file, 'r') as f:
#         model_str = f.read()
#     return lgb.Booster(model_str=model_str)

# def load_cat_model(model_file: str) -> CatBoostRegressor:
#     model = CatBoostRegressor()
#     model.load_model(model_file)
#     return model

# # ============ A) 先加载 NN ============
# print(f"[Init] Loading offline NN => {MODEL_CKPT_PATH}")
# _nn_tmp = NN.load_from_checkpoint(
#     MODEL_CKPT_PATH,
#     input_dim=79,
#     hidden_dims=[512,512,256],
#     dropouts=[0.1,0.1],
#     lr=1e-3,
#     weight_decay=5e-4
# )
# dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# _nn_tmp.to(dev)
# _nn_tmp.eval()
# trained_nn = _nn_tmp

# # 约定: 第一次出现 date_id => 记录到 first_date_nn => 跳过在线学习
# # 但要直到 predict() 被调用时，才知道 date_id
# # 因此这里的 first_date_nn 先不赋值

# # ============ B) 加载 LGB & Cat ============
# print(f"[Init] Loading LGB => {LGB_MODEL_PATH}")
# lgb_model_single = load_lgb_model_txt(LGB_MODEL_PATH)
# print(f"[Init] => lgb_model_single => #trees={lgb_model_single.num_trees()}")

# print(f"[Init] Loading Cat => {CAT_MODEL_PATH}")
# cat_model_single = load_cat_model(CAT_MODEL_PATH)
# print("[Init] => cat_model_single => loaded done")

# ##########################################################
# # 5) predict(test, lags)
# ##########################################################
# def predict(test: pls.DataFrame, lags: pls.DataFrame | None) -> pd.DataFrame:
#     """
#     1) NN => 在线学习 => nn_pred
#     2) LGB+Cat => 拼接 lag 特征 => lgb_pred & cat_pred
#     3) final_pred = ALPHA_NN*nn_pred + (1-ALPHA_NN)*( ALPHA_LGB*lgb + (1-ALPHA_LGB)*cat )
#        => clip(-5,5)
#     """
#     global trained_nn, test_cache_nn, first_date_nn
#     global lgb_model_single, cat_model_single
#     global lags_  # 新增

#     # A) polars => pandas  (for NN part)
#     test_pd = test.to_pandas()
#     date_now = test_pd["date_id"].iloc[0]
#     row_ids  = test_pd["row_id"].values if "row_id" in test_pd.columns else np.arange(len(test_pd))

#     # B) NN => 缓存 => 在线学习
#     if first_date_nn is None:
#         # 第一次 => 记录
#         first_date_nn = date_now
#     if date_now not in test_cache_nn:
#         test_cache_nn[date_now] = []
#     test_cache_nn[date_now].append(test_pd)

#     # 当有新一天时, 先对“上一天”做一次在线学习
#     if lags is not None:
#         # 说明这是新的一天(竞赛环境下, 通常会先给你当天 lags)
#         if date_now == first_date_nn:
#             print("[NN] skip online train => first day")
#         else:
#             # 训练上一天的数据
#             prev_day = date_now - 1
#             if prev_day in test_cache_nn:
#                 t0 = time.time()
#                 dev = next(trained_nn.parameters()).device
#                 prev_pd = pd.concat(test_cache_nn[prev_day], ignore_index=True)
                
#                 # 这里假设 lags 也是当前新的, 不一定要用
#                 # 所以我们不再做 merges 依赖 lags, 直接用 test里的第6个responder_6?
#                 # 但原逻辑还是要 merges "responder_6_lag_1" 当作 y
#                 # 你可以保持原逻辑(只要 data alignment OK).
#                 lags_pd = lags.to_pandas()

#                 if ASSUME_ROW_ALIGNMENT:
#                     if len(prev_pd)!=len(lags_pd):
#                         print("[WARN] mismatch row => naive alignment")
#                     merged = prev_pd.copy()
#                     merged["responder_6_lag_1"] = lags_pd["responder_6_lag_1"].values
#                 else:
#                     merged = pd.merge(
#                         prev_pd, lags_pd,
#                         on=["time_id","symbol_id"],
#                         how="inner"
#                     )

#                 merged = forward_fill_and_zero(merged)
#                 X_tr = torch.FloatTensor(merged[NN_FEATURES_79].values).to(dev)
#                 y_tr = torch.FloatTensor(merged["responder_6_lag_1"].values).to(dev)
#                 w_tr = torch.FloatTensor(merged["weight"].values).to(dev)

#                 # freeze BN
#                 trained_nn.train()
#                 for m in trained_nn.modules():
#                     if isinstance(m, nn.BatchNorm1d):
#                         m.eval()
#                         for p in m.parameters():
#                             p.requires_grad=False

#                 opt = torch.optim.Adam(
#                     filter(lambda p: p.requires_grad, trained_nn.parameters()),
#                     lr=ONLINE_LEARNING_PARAMS["lr"],
#                     weight_decay=ONLINE_LEARNING_PARAMS["weight_decay"]
#                 )
#                 sch = ExponentialLR(opt, gamma=ONLINE_LEARNING_PARAMS["lr_gamma"])

#                 for _ep in range(ONLINE_LEARNING_PARAMS["epochs"]):
#                     perm = torch.randperm(X_tr.size(0))
#                     bs_  = ONLINE_LEARNING_PARAMS["batch_size"]
#                     for st in range(0, X_tr.size(0), bs_):
#                         idx = perm[st:st+bs_]
#                         x_b, y_b, w_b = X_tr[idx], y_tr[idx], w_tr[idx]
#                         opt.zero_grad()
#                         p_b = trained_nn(x_b)
#                         loss_b = (w_b*(p_b - y_b)**2).mean()
#                         loss_b.backward()
#                         opt.step()
#                     sch.step()

#                 trained_nn.eval()
#                 # 清理上一天的缓存
#                 del test_cache_nn[prev_day]
#                 print(f"[NN] day {prev_day}->{date_now} => online train => cost={time.time()-t0:.2f}s")

#     # NN推理
#     dev = next(trained_nn.parameters()).device
#     test_ff = forward_fill_and_zero(test_pd)
#     X_nn    = torch.FloatTensor(test_ff[NN_FEATURES_79].values).to(dev)
#     with torch.no_grad():
#         nn_pred = trained_nn(X_nn).cpu().numpy().ravel()

#     # C) 处理 LGB+Cat => 9lag (Polars 的简单写法)
#     #    如果当日开始时 lags 不为 None, 更新全局 lags_
#     if lags is not None:
#         lags_ = lags  # 存储下来, 当天多次 predict 时复用

#     # 在这里拼接最后时刻(上一时刻)的 9 lag 特征 => group_by + last => join
#     # 如果没有 lags_, 就直接给 9 个 lag 列补 0
#     if lags_ is not None:
#         lag_tail = (
#             lags_
#             .group_by(["date_id", "symbol_id"], maintain_order=True)
#             .last()
#             .drop(["time_id"])  # 不需要 time_id
#         )
#         lgbcat = test.join(lag_tail, on=["date_id","symbol_id"], how="left")
#     else:
#         # 补 0
#         for idx in range(9):
#             test = test.with_columns(pl.lit(0.0).alias(f"responder_{idx}_lag_1"))
#         lgbcat = test

#     # 转成 pandas => 送入 lgb/cat
#     lgbcat_pd = lgbcat.to_pandas()
#     X_lgbcat  = lgbcat_pd[LGB_CAT_FEATS_89]
#     lgb_pred  = lgb_model_single.predict(X_lgbcat)
#     cat_pred  = cat_model_single.predict(X_lgbcat)

#     # D) 最终融合 => alpha
#     lgbcat_ens = ALPHA_LGB*lgb_pred + (1.0 - ALPHA_LGB)*cat_pred
#     final_pred = ALPHA_NN*nn_pred + (1.0 - ALPHA_NN)*lgbcat_ens

#     # E) clip => [-5,5]
#     final_pred = np.clip(final_pred, a_min=-5.0, a_max=5.0)

#     # 输出
#     out_df = pd.DataFrame({
#         "row_id":   row_ids,
#         "responder_6": final_pred
#     })
#     assert (out_df.columns == ["row_id","responder_6"]).all()
#     assert len(out_df) == len(test_pd)

#     return out_df


# nn lgb cat
import os
import gc
import numpy as np
import pandas as pd
import polars as pls
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time

import lightgbm as lgb
from catboost import CatBoostRegressor

# 若需要官方评测接口
import kaggle_evaluation.jane_street_inference_server

from torch.optim.lr_scheduler import ExponentialLR

##########################################################
# 0) 全局配置
##########################################################
ONLINE_LEARNING_PARAMS = {
    "lr":          2.05943e-05,     # 在线训练的学习率
    "weight_decay":0.00096991,
    "batch_size":  21260,
    "epochs":      1,
    "lr_gamma":    0.99
}

# NN ckpt (NN1)
MODEL_CKPT_PATH = "/kaggle/input/trial/pytorch/default/1/trialx.model.ckpt"
# 新增: NN2 ckpt (请改为你的实际文件路径)
MODEL2_CKPT_PATH = "/kaggle/input/trial/pytorch/default/1/trialA.model.ckpt"

MODEL3_CKPT_PATH = "/kaggle/input/trial/pytorch/default/1/trialAA.model.ckpt"

MODEL4_CKPT_PATH = "/kaggle/input/trial/pytorch/default/1/trialAAA.model.ckpt"

# LGB & Cat
LGB_MODEL_PATH = "/kaggle/input/nnlgbcat/other/default/1/lgb_model_single_0.txt"
LGB2_MODEL_PATH = "/kaggle/input/lgb_2/other/default/1/lgb_model_single_0_0.txt"
CAT_MODEL_PATH = "/kaggle/input/nnlgbcat/other/default/1/cat_symmetric_best.cbm"

# NN特征(仅79列)
NN_FEATURES_79 = [
    f"feature_{i:02d}" for i in range(79)
    if i not in (9, 10, 11)
]
# LGB/Cat特征(79 + 9 lag + symbol_id => 89列)
LGB_CAT_FEATS_89 = (
    [f"feature_{i:02d}" for i in range(79)]
    + [f"responder_{i}_lag_1" for i in range(9)]
    + ["symbol_id"]
)

# 融合系数(仅示例)
ALPHA_NN   = 0.60
ALPHA_LGB  = 0.60

# 是否假设行对齐(在合并NN和lags数据时用)
ASSUME_ROW_ALIGNMENT = False

##########################################################
# 1) 定义 NN(与离线训练一致), 用于在线学习
##########################################################
class NN(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, dropouts, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        # 与你训练时相同的网络结构
        layers = []
        in_dim = input_dim

        # 第1层: dropout -> Linear -> SiLU
        layers.append(nn.Dropout(dropouts[0]))
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.SiLU())

        # 后续几层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Dropout(dropouts[i]))
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.SiLU())

        # 最后 dropout
        layers.append(nn.Dropout(dropouts[-1]))
        # 输出 => 1 => Tanh => *5
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 5.0 * self.model(x).squeeze(-1)

##########################################################
# 2) forward_fill_and_zero
##########################################################
def forward_fill_and_zero(
    df: pd.DataFrame,
    sort_cols=["time_id"],
    group_cols=["symbol_id"],
    fill_cols=None
):
    """
    groupby(...) => ffill => fillna(0)
    """
    if fill_cols is None:
        fill_cols = NN_FEATURES_79[:]
        if "weight" in df.columns:
            fill_cols.append("weight")

    df = df.sort_values(group_cols + sort_cols, ignore_index=True)
    df[fill_cols] = df.groupby(group_cols)[fill_cols].ffill()
    df[fill_cols] = df[fill_cols].fillna(0)
    return df

##########################################################
# 3) 全局变量 => 3个模型(已加载) & 在线学习cache
##########################################################
# （A）NN1 & NN2，共用 test_cache_nn & first_date_nn
trained_nn      = None   # NN1
trained_nn2     = None   # NN2
trained_nn3     = None   # NN3
trained_nn4     = None   # NN4
test_cache_nn   = {}
first_date_nn   = None   # 用于跳过首日在线学习

# （B）LGB, Cat
lgb_model_single = None
cat_model_single = None

# （C）全局 lags_ 用于拼接 9 lag 特征 (Polars方式)
lags_ = None

##########################################################
# 4) 在脚本加载阶段，就把 3 个模型初始化好
##########################################################
def load_lgb_model_txt(model_file: str) -> lgb.Booster:
    """加载 LightGBM txt 模型"""
    with open(model_file, 'r') as f:
        model_str = f.read()
    return lgb.Booster(model_str=model_str)

def load_cat_model(model_file: str) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(model_file)
    return model

# ============ A) 先加载 NN1 ============
print(f"[Init] Loading offline NN => {MODEL_CKPT_PATH}")
_nn_tmp = NN.load_from_checkpoint(
    MODEL_CKPT_PATH,
    input_dim=76,
    hidden_dims=[384,896,896,394],
    dropouts=[0.1014378698,0.1972033905,0.1123435323,0.2314834093,0.2157768967],
    lr=1e-3,
    weight_decay=5e-4
)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_nn_tmp.to(dev)
_nn_tmp.eval()
trained_nn = _nn_tmp

# ============ A2) 再加载 NN2 ============
print(f"[Init] Loading offline NN2 => {MODEL2_CKPT_PATH}")
_nn_tmp2 = NN.load_from_checkpoint(
    MODEL2_CKPT_PATH,
    input_dim=76,
    hidden_dims=[384,896,896,394],
    dropouts=[0.1014378698,0.1972033905,0.1123435323,0.2314834093,0.2157768967],
    lr=1e-3,
    weight_decay=5e-4
)
_nn_tmp2.to(dev)
_nn_tmp2.eval()
trained_nn2 = _nn_tmp2

print(f"[Init] Loading offline NN3 => {MODEL3_CKPT_PATH}")
_nn_tmp3 = NN.load_from_checkpoint(
    MODEL3_CKPT_PATH,
    input_dim=76,
    hidden_dims=[384,896,896,394],
    dropouts=[0.1014378698,0.1972033905,0.1123435323,0.2314834093,0.2157768967],
    lr=1e-3,
    weight_decay=5e-4
)
_nn_tmp3.to(dev)
_nn_tmp3.eval()
trained_nn3 = _nn_tmp3

print(f"[Init] Loading offline NN4 => {MODEL4_CKPT_PATH}")
_nn_tmp4 = NN.load_from_checkpoint(
    MODEL4_CKPT_PATH,
    input_dim=76,
    hidden_dims=[384,896,896,394],
    dropouts=[0.1014378698,0.1972033905,0.1123435323,0.2314834093,0.2157768967],
    lr=1e-3,
    weight_decay=5e-4
)
_nn_tmp4.to(dev)
_nn_tmp4.eval()
trained_nn4 = _nn_tmp4

# 约定: 第一次出现 date_id => 记录到 first_date_nn => 跳过在线学习
# 但要直到 predict() 被调用时，才知道 date_id
# 因此这里的 first_date_nn 先不赋值

# ============ B) 加载 LGB & Cat ============
print(f"[Init] Loading LGB => {LGB_MODEL_PATH}")
lgb_model_single = load_lgb_model_txt(LGB_MODEL_PATH)
print(f"[Init] => lgb_model_single => #trees={lgb_model_single.num_trees()}")

print(f"[Init] Loading second LGB => {LGB2_MODEL_PATH}")
lgb_model_single2 = load_lgb_model_txt(LGB2_MODEL_PATH)
print(f"[Init] => lgb_model_single2 => #trees={lgb_model_single2.num_trees()}")

print(f"[Init] Loading Cat => {CAT_MODEL_PATH}")
cat_model_single = load_cat_model(CAT_MODEL_PATH)
print("[Init] => cat_model_single => loaded done")

##########################################################
# 5) predict(test, lags)
##########################################################
def predict(test: pls.DataFrame, lags: pls.DataFrame | None) -> pd.DataFrame:
    """
    1) NN1 & NN2 => 在线学习 => nn_pred1, nn_pred2
    2) 取 nn_pred_avg = 0.5*(nn_pred1 + nn_pred2)
    3) LGB+Cat => 同样处理 => lgb_pred & cat_pred
    4) final_pred = ALPHA_NN*nn_pred_avg + (1-ALPHA_NN)*( ALPHA_LGB*lgb + (1-ALPHA_LGB)*cat )
       => clip(-5,5)
    """
    global trained_nn, trained_nn2, trained_nn3, trained_nn4, test_cache_nn, first_date_nn
    global lgb_model_single, cat_model_single
    global lags_  # 新增

    # A) polars => pandas  (for NN part)
    test_pd = test.to_pandas()
    date_now = test_pd["date_id"].iloc[0]
    row_ids  = test_pd["row_id"].values if "row_id" in test_pd.columns else np.arange(len(test_pd))

    # B) NN => 缓存 => 在线学习 (同一份)
    if first_date_nn is None:
        first_date_nn = date_now
    if date_now not in test_cache_nn:
        test_cache_nn[date_now] = []
    test_cache_nn[date_now].append(test_pd)

    if lags is not None:
        # 说明这是新的一天(竞赛环境下, 通常会先给你当天 lags)
        if date_now == first_date_nn:
            print("[NN] skip online train => first day")
        else:
            # 训练上一天的数据 => 对 NN1 & NN2 都要做
            prev_day = date_now - 1
            if prev_day in test_cache_nn:
                t0 = time.time()
                dev_cur = next(trained_nn.parameters()).device
                prev_pd = pd.concat(test_cache_nn[prev_day], ignore_index=True)

                lags_pd = lags.to_pandas()

                if ASSUME_ROW_ALIGNMENT:
                    if len(prev_pd)!=len(lags_pd):
                        print("[WARN] mismatch row => naive alignment")
                    merged = prev_pd.copy()
                    merged["responder_6_lag_1"] = lags_pd["responder_6_lag_1"].values
                else:
                    merged = pd.merge(
                        prev_pd, lags_pd,
                        on=["time_id","symbol_id"],
                        how="inner"
                    )

                merged = forward_fill_and_zero(merged)
                X_tr = torch.FloatTensor(merged[NN_FEATURES_79].values).to(dev_cur)
                y_tr = torch.FloatTensor(merged["responder_6_lag_1"].values).to(dev_cur)
                w_tr = torch.FloatTensor(merged["weight"].values).to(dev_cur)

                # == 1) 在线学习 NN1 ==
                trained_nn.train()
                opt = torch.optim.Adam(
                    trained_nn.parameters(),
                    lr=ONLINE_LEARNING_PARAMS["lr"],
                    weight_decay=ONLINE_LEARNING_PARAMS["weight_decay"]
                )
                sch = ExponentialLR(opt, gamma=ONLINE_LEARNING_PARAMS["lr_gamma"])

                for _ep in range(ONLINE_LEARNING_PARAMS["epochs"]):
                    perm = torch.randperm(X_tr.size(0))
                    bs_  = ONLINE_LEARNING_PARAMS["batch_size"]
                    for st in range(0, X_tr.size(0), bs_):
                        idx = perm[st:st+bs_]
                        x_b, y_b, w_b = X_tr[idx], y_tr[idx], w_tr[idx]
                        opt.zero_grad()
                        p_b = trained_nn(x_b)
                        loss_b = (w_b*(p_b - y_b)**2).mean()
                        loss_b.backward()
                        opt.step()
                    sch.step()
                trained_nn.eval()

                # == 2) 在线学习 NN2 ==
                trained_nn2.train()
                opt = torch.optim.Adam(
                    trained_nn2.parameters(),
                    lr=ONLINE_LEARNING_PARAMS["lr"],
                    weight_decay=ONLINE_LEARNING_PARAMS["weight_decay"]
                )
                sch = ExponentialLR(opt, gamma=ONLINE_LEARNING_PARAMS["lr_gamma"])

                for _ep in range(ONLINE_LEARNING_PARAMS["epochs"]):
                    perm = torch.randperm(X_tr.size(0))
                    bs_  = ONLINE_LEARNING_PARAMS["batch_size"]
                    for st in range(0, X_tr.size(0), bs_):
                        idx = perm[st:st+bs_]
                        x_b, y_b, w_b = X_tr[idx], y_tr[idx], w_tr[idx]
                        opt.zero_grad()
                        p_b = trained_nn2(x_b)
                        loss_b = (w_b*(p_b - y_b)**2).mean()
                        loss_b.backward()
                        opt.step()
                    sch.step()
                trained_nn2.eval()

                # == 3) 在线学习 NN3 ==
                trained_nn3.train()
                opt = torch.optim.Adam(
                    trained_nn3.parameters(),
                    lr=ONLINE_LEARNING_PARAMS["lr"],
                    weight_decay=ONLINE_LEARNING_PARAMS["weight_decay"]
                )
                sch = ExponentialLR(opt, gamma=ONLINE_LEARNING_PARAMS["lr_gamma"])

                for _ep in range(ONLINE_LEARNING_PARAMS["epochs"]):
                    perm = torch.randperm(X_tr.size(0))
                    bs_  = ONLINE_LEARNING_PARAMS["batch_size"]
                    for st in range(0, X_tr.size(0), bs_):
                        idx = perm[st:st+bs_]
                        x_b, y_b, w_b = X_tr[idx], y_tr[idx], w_tr[idx]
                        opt.zero_grad()
                        p_b = trained_nn3(x_b)
                        loss_b = (w_b*(p_b - y_b)**2).mean()
                        loss_b.backward()
                        opt.step()
                    sch.step()
                trained_nn3.eval()

                # == 4) 在线学习 NN4 ==
                trained_nn4.train()
                opt = torch.optim.Adam(
                    trained_nn4.parameters(),
                    lr=ONLINE_LEARNING_PARAMS["lr"],
                    weight_decay=ONLINE_LEARNING_PARAMS["weight_decay"]
                )
                sch = ExponentialLR(opt, gamma=ONLINE_LEARNING_PARAMS["lr_gamma"])

                for _ep in range(ONLINE_LEARNING_PARAMS["epochs"]):
                    perm = torch.randperm(X_tr.size(0))
                    bs_  = ONLINE_LEARNING_PARAMS["batch_size"]
                    for st in range(0, X_tr.size(0), bs_):
                        idx = perm[st:st+bs_]
                        x_b, y_b, w_b = X_tr[idx], y_tr[idx], w_tr[idx]
                        opt.zero_grad()
                        p_b = trained_nn4(x_b)
                        loss_b = (w_b*(p_b - y_b)**2).mean()
                        loss_b.backward()
                        opt.step()
                    sch.step()
                trained_nn4.eval()

                # 清理上一天的缓存
                del test_cache_nn[prev_day]
                print(f"[NN] day {prev_day}->{date_now} => online train => cost={time.time()-t0:.2f}s")

    # NN推理 => NN1
    dev_cur = next(trained_nn.parameters()).device
    test_ff = forward_fill_and_zero(test_pd)
    X_nn    = torch.FloatTensor(test_ff[NN_FEATURES_79].values).to(dev_cur)
    with torch.no_grad():
        nn_pred1 = trained_nn(X_nn).cpu().numpy().ravel()

    # NN推理 => NN2
    with torch.no_grad():
        nn_pred2 = trained_nn2(X_nn).cpu().numpy().ravel()

    # NN推理 => NN3
    with torch.no_grad():
        nn_pred3 = trained_nn3(X_nn).cpu().numpy().ravel()

    # NN推理 => NN4
    with torch.no_grad():
        nn_pred4 = trained_nn4(X_nn).cpu().numpy().ravel()

    # NN avg
    nn_pred_avg = 0.25*(nn_pred1 + nn_pred2 + nn_pred3 + nn_pred4)

    # C) 处理 LGB+Cat => 9lag (Polars 的简单写法)
    if lags is not None:
        lags_ = lags  # 存储下来, 当天多次 predict 时复用

    if lags_ is not None:
        lag_tail = (
            lags_
            .group_by(["date_id", "symbol_id"], maintain_order=True)
            .last()
            .drop(["time_id"])  # 不需要 time_id
        )
        lgbcat = test.join(lag_tail, on=["date_id","symbol_id"], how="left")
    else:
        # 补 0
        for idx in range(9):
            test = test.with_columns(pl.lit(0.0).alias(f"responder_{idx}_lag_1"))
        lgbcat = test

    # 转成 pandas => 送入 lgb/cat
    lgbcat_pd = lgbcat.to_pandas()
    X_lgbcat  = lgbcat_pd[LGB_CAT_FEATS_89]
    lgb_pred1 = lgb_model_single.predict(X_lgbcat)
    lgb_pred2 = lgb_model_single2.predict(X_lgbcat)
    lgb_pred  = 0.5*lgb_pred1 + 0.5*lgb_pred2
    cat_pred  = cat_model_single.predict(X_lgbcat)

    # ----------------------------------------------------
    # 关键：针对 symbol_id <39 和 >=39 做不同融合
    # ----------------------------------------------------
    symbol_arr = test_pd["symbol_id"].values  # numpy array
    mask_new   = (symbol_arr >= 39)

    # 先生成容器
    final_pred = np.zeros(len(test_pd), dtype=np.float32)

    # 1) 对老symbol =>  NN avg + LGB + Cat
    #    融合 => final_pred_old = ALPHA_NN*nn_avg + (1-ALPHA_NN)* [ ALPHA_LGB*lgb + (1-ALPHA_LGB)*cat ]
    #    其中 ALPHA_NN, ALPHA_LGB 全局定义
    lgbcat_fuse = ALPHA_LGB*lgb_pred + (1.0 - ALPHA_LGB)*cat_pred
    final_pred[~mask_new] = ( ALPHA_NN*nn_pred_avg[~mask_new]
                              + (1-ALPHA_NN)*lgbcat_fuse[~mask_new] )

    # 2) 对新symbol => 仅 NN avg + Cat
    #    final_pred_new = ALPHA_NN*nn_avg + (1-ALPHA_NN)*cat
    final_pred[mask_new] = ( ALPHA_NN*nn_pred_avg[mask_new]
                             + (1-ALPHA_NN)*cat_pred[mask_new] )

    # clip
    final_pred = np.clip(final_pred, -5, 5)

    out_df = pd.DataFrame({
        "row_id": row_ids,
        "responder_6": final_pred
    })
    assert (out_df.columns == ["row_id","responder_6"]).all()
    assert len(out_df) == len(test_pd)

    return out_df


inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )

