!pip install recbole


#Đường dẫn của mô hình đã huấn luyện trong input
!cp /kaggle/input/sasrec-new/SASRec.pth /kaggle/working/


import tqdm
import polars as pl
import numpy as np
import pandas as pd
import seaborn as sns
import random
import os 
import h5py
import sys
import gc

from matplotlib import pyplot as plt
import pyarrow.parquet as pq


train = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test.parquet')
test = pl.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')

df = pl.concat([train, test])

df = df.sort(['session', 'ts'])
df = df.with_columns((pl.col('ts') * 1e9).alias('ts'))
df = df.rename({'session': 'session:token', 'aid': 'aid:token', 'ts': 'ts:float'})


!mkdir -p /kaggle/working/recbox_data
df[['session:token', 'aid:token', 'ts:float']].write_csv('/kaggle/working/recbox_data/recbox_data.inter', separator='\t')

del df, train, test
gc.collect()


import logging
from logging import getLogger
import typing
from typing_extensions import Literal
typing.Literal = Literal
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from recbole.utils.case_study import full_sort_topk


MAX_ITEM = 30  

parameter_dict = {
    # === Data ===
    'data_path': '/kaggle/working/',
    'USER_ID_FIELD': 'session',
    'ITEM_ID_FIELD': 'aid',
    'TIME_FIELD': 'ts',
    'user_inter_num_interval': "[5,Inf)",
    'item_inter_num_interval': "[5,Inf)",
    'load_col': {'inter': ['session', 'aid', 'ts']},

    # === Tham số huấn luyện ===
    'epochs': 20,                    
    'stopping_step': 5,
    'train_batch_size': 512,         
    'eval_batch_size': 1024,
    'train_neg_sample_args': None,
    'learning_rate': 5e-4,

    # === Sequence handling ===
    'MAX_ITEM_LIST_LENGTH': MAX_ITEM,
    'MAX_SEQ_LENGTH': MAX_ITEM,       
    'hidden_size': 128,               # embedding dimension
    'num_heads': 4,                   # số head trong self-attention
    'num_layers': 2,                  # số layer Transformer
    'hidden_dropout_prob': 0.2,       # dropout cho feed-forward
    'attn_dropout_prob': 0.2,         # dropout cho attention

    # === Evaluation Metrics ===
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
    'topk': [20],                
    'valid_metric': 'Recall@20', 

    # === Evaluation ===
    'eval_args': {
        'split': {'LS': 'valid_and_test'},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'
    },

    # === Hàm mất mát ===
    'loss_type': 'CE',              

    # === Lưu mô hình ===
    'checkpoint_dir': '/kaggle/working/',   
    'save_best': True,                       
}

# === Khởi tạo config ===
config = Config(model='SASRec', dataset='recbox_data', config_dict=parameter_dict)

# === Khởi tạo random seed và logger ===
init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = getLogger()

# === Tạo handler để log ra màn hình ===
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# === In config để kiểm tra ===
logger.info(config)


model = SASRec(config, train_data.dataset).to(config['device'])
logger.info(model)


import torch

#Đường dẫn của mô hình đã huấn luyện trong input
checkpoint_path = '/kaggle/input/sasrec-new/SASRec.pth'

#Tải lại trọng số đã huấn luyện
ckpt = torch.load(checkpoint_path, map_location=config['device'], weights_only=False)
model.load_state_dict(ckpt['state_dict'])
model.eval()

logger.info("Model weights loaded successfully!")


# ==============================================================================
# ĐỊNH NGHĨA HÀM INFERENCE (TRẢ VỀ CẢ SCORE)
# ==============================================================================
def pred_user_to_item_batch_with_score(item_sequences: List[List[str]], topk: int = 20, batch_size: int = 2048):
    """
    Hàm dự đoán trả về cả Item ID và Score (Logit) để dùng cho Rerank
    """
    model.eval()
    vocab = dataset.field2token_id[dataset.iid_field]
    
    all_items = []
    all_scores = []
    
    # Tự động lấy độ dài tối đa của model
    try:
        max_model_len = model.position_embedding.weight.shape[0]
    except AttributeError:
        max_model_len = 50 
    
    # Loop qua từng batch
    for start_idx in tqdm(range(0, len(item_sequences), batch_size), desc="Predicting Batches"):
        end_idx = start_idx + batch_size
        batch_seqs = item_sequences[start_idx : end_idx]
        
        # Xử lý Sequence (Cắt ngắn & Padding)
        cleaned_sequences = []
        for seq in batch_seqs:
            # Lọc item có trong vocab
            valid_items = [x for x in seq if x in vocab]
            # Cắt nếu quá dài (giữ phần đuôi)
            if len(valid_items) > max_model_len:
                valid_items = valid_items[-max_model_len:]
            cleaned_sequences.append(valid_items)

        lengths = [len(seq) for seq in cleaned_sequences]
        max_len = max(lengths) if lengths else 0
        
        if max_len == 0:
            all_items.extend([[] for _ in batch_seqs])
            all_scores.extend([[] for _ in batch_seqs])
            continue

        token_ids = []
        for seq in cleaned_sequences:
            if len(seq) == 0:
                token_ids.append([0] * max_len)
            else:
                tokens = dataset.token2id(dataset.iid_field, seq)
                if isinstance(tokens, np.ndarray): tokens = tokens.tolist()
                token_ids.append(tokens + [0]*(max_len - len(seq)))

        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=model.device)
        safe_lengths = [l if l > 0 else 1 for l in lengths]
        length_tensor = torch.tensor(safe_lengths, dtype=torch.long, device=model.device)

        seq_field = f"{dataset.iid_field}_list"
        input_interaction = Interaction({
            dataset.iid_field: token_tensor, 
            seq_field: token_tensor,        
            'item_length': length_tensor
        })

        # Dự đoán và lấy điểm
        with torch.no_grad():
            scores = model.full_sort_predict(input_interaction)
            scores[:, 0] = -np.inf # Mask padding token
            
            # Lấy cả Score và Item Index
            batch_scores, topk_iids = torch.topk(scores, topk)
        
        # Chuyển về CPU list
        topk_iids_list = topk_iids.cpu().tolist()
        batch_scores_list = batch_scores.cpu().tolist()
        
        # Dọn dẹp GPU
        del token_tensor, length_tensor, input_interaction, scores, topk_iids, batch_scores
        
        # Decode (Index -> Item ID)
        batch_results_items = []
        batch_results_scores = []
        
        for idx, (l, original_len) in enumerate(zip(topk_iids_list, lengths)):
            if original_len == 0:
                batch_results_items.append([])
                batch_results_scores.append([])
                continue
            try:
                decoded = dataset.id2token(dataset.iid_field, l)
                # Item
                if isinstance(decoded, np.ndarray): batch_results_items.append(decoded.tolist())
                else: batch_results_items.append(list(decoded))
                # Score
                batch_results_scores.append(batch_scores_list[idx])
            except:
                batch_results_items.append([])
                batch_results_scores.append([])
        
        all_items.extend(batch_results_items)
        all_scores.extend(batch_results_scores)

    return all_items, all_scores

# ==============================================================================
# CHUẨN BỊ DỮ LIỆU & CHẠY INFERENCE
# ==============================================================================
print("Loading Test Data...")
test = pl.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')

# Gom nhóm session thành list
print("Grouping sessions...")
session_df = (
    test.group_by("session")
        .agg([pl.col("aid").alias("aid")])
        .sort("session")
)
del test; gc.collect()

# Chuyển sang List Python
session_ids = session_df["session"].to_list()
session_aids = session_df["aid"].to_list()
del session_df; gc.collect()

# Chuyển item sang string 
print("Converting to string sequences...")
all_sessions_str = [[str(x) for x in aids] for aids in session_aids]

# Chạy mô hình
print(f"Running Inference for {len(all_sessions_str)} sessions...")
pred_items, pred_scores = pred_user_to_item_batch_with_score(
    all_sessions_str, 
    topk=100,             
    batch_size=2048      
)

# Dọn dẹp input
del all_sessions_str
gc.collect()

# ==============================================================================
# LƯU KẾT QUẢ (POLARS)
# ==============================================================================
print("Creating Output DataFrame...")
df_sasrec = pl.DataFrame({
    "session": session_ids,
    "aid": pred_items,
    "sasrec_score": pred_scores
})

# Trải phẳng list thành các dòng
print("Exploding list...")
df_sasrec = df_sasrec.explode(["aid", "sasrec_score"])

# Loại bỏ dòng null (nếu có session rỗng)
df_sasrec = df_sasrec.drop_nulls()

# Ép kiểu và Tạo Rank
print("Formatting & Ranking...")
df_sasrec = df_sasrec.with_columns([
    pl.col("session").cast(pl.Int32),
    pl.col("aid").cast(pl.Int32),
    pl.col("sasrec_score").cast(pl.Float32),
    
    # Tạo cột Rank (1 -> 100)
    pl.col("sasrec_score")
      .rank(method="ordinal", descending=True)
      .over("session")
      .alias("sasrec_rank")
      .cast(pl.Int32)
])

# Lưu file
output_path = '/kaggle/working/sasrec_predictions.parquet'
print(f"Saving to {output_path}...")
df_sasrec.write_parquet(output_path)

print(f"DONE! File saved with shape: {df_sasrec.shape}")
print(df_sasrec.head())

del df_sasrec, pred_items, pred_scores, session_ids
gc.collect()

