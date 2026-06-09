# Install dependencies
!pip install pandas numpy matplotlib seaborn polars pyarrow tqdm h5py pydantic -q
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
!pip install recbole -q
!pip install protobuf==3.20.0 -q


# ====== IMPORT THƯ VIỆN ======
import os
import sys
import gc
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

from tqdm.auto import tqdm


# ====== OTTO PARAMETERS ======

# Event type mapping
TYPE_LABELS = {
    "clicks": 0, 
    "carts": 1, 
    "orders": 2
}

# Reverse mapping
ID_TO_TYPE = {v: k for k, v in TYPE_LABELS.items()}

# RecBole config
MAX_ITEM = 20


# Đọc dữ liệu test của OTTO từ hgy1 và hgy2
train_df = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test.parquet')
test_df = pl.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')

inter_df = pl.concat([train_df, test_df])

# Sắp xếp theo session, aid và timestamp
inter_df = inter_df.sort(['session', 'aid', 'ts'])

# RecBole format
inter_df = inter_df.with_columns((pl.col('ts') * 1e9).alias('ts'))
inter_df = inter_df.rename({'session': 'session:token', 'aid': 'aid:token', 'ts': 'ts:float'})

print(inter_df.columns)


directory = "recbox_data"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Created: {directory}")
else:
    print(f"Exists: {directory}")


pandas_df = inter_df[['session:token', 'aid:token', 'ts:float']].to_pandas()

pandas_df.to_csv(
    'recbox_data/recbox_data.inter',
    sep='\t',
    index=False
)

del inter_df, pandas_df, train_df, test_df
gc.collect()


!pip install recbole -q
!pip install kmeans-pytorch -q


import logging
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from recbole.utils.case_study import full_sort_topk


# 1. Vá lỗi các kiểu dữ liệu số và logic
np.float = np.float64
np.float_ = np.float64
np.int = np.int64
np.int_ = np.int64
np.bool = np.bool_
np.bool_ = np.bool_
np.complex = np.complex128
np.complex_ = np.complex128
np.object = np.object_
np.object_ = np.object_

# 2. Vá lỗi các kiểu dữ liệu chuỗi và số nguyên dài (Sửa lỗi np.unicode_)
np.str = np.str_
np.unicode = np.str_     # np.unicode_ đã bị xóa, dùng np.str_ thay thế
np.unicode_ = np.str_
np.long = np.int64       # np.long thường tương ứng với int64

recbole_config = {
    'data_path': '.',
    'USER_ID_FIELD': 'session',
    'ITEM_ID_FIELD': 'aid',
    'TIME_FIELD': 'ts',
    'user_inter_num_interval': "[5,Inf)",
    'item_inter_num_interval': "[5,Inf)",
    'load_col': {'inter': ['session', 'aid', 'ts']},
    'train_neg_sample_args': None,
    'epochs': 1,
    'stopping_step': 3,
    'eval_batch_size': 512,  # Giảm để tiết kiệm memory cho Kaggle free
    'train_batch_size': 512,  # Giảm để tiết kiệm memory cho Kaggle free
    'MAX_ITEM_LIST_LENGTH': MAX_ITEM,
    'eval_args': {
        'split': {'RS': [9, 1, 0]},
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full',
    },
    # ====== CÁC THAM SỐ ĐẶC TRƯNG CHO BERT4Rec (Tối ưu cho Kaggle Free GPU) ======
    'hidden_size': 64,              # Giảm từ mặc định để tiết kiệm memory
    'inner_size': 256,              # Kích thước feed-forward layer
    'n_layers': 2,                  # Số layers Transformer (giảm để nhanh hơn)
    'n_heads': 2,                   # Số attention heads (giảm để tiết kiệm memory)
    'hidden_dropout_prob': 0.3,     # Dropout cho hidden layers
    'attn_dropout_prob': 0.3,       # Dropout cho attention
    'layer_norm_eps': 1e-12,        # Epsilon cho layer normalization
    'initializer_range': 0.02,      # Range cho weight initialization
    'loss_type': 'BPR',             # Loss function: BPR hoặc CE
}

config = Config(model='BERT4Rec', dataset='recbox_data', config_dict=recbole_config)
init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = getLogger()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.info(config)


dataset = create_dataset(config)
logger.info(dataset)


train_data, valid_data, test_data = data_preparation(config, dataset)


model = BERT4Rec(config, train_data.dataset).to(config['device'])
logger.info(model)

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)


del trainer, train_data, valid_data, test_data
gc.collect()


from typing import List, Tuple, Dict
import torch
from pydantic import BaseModel
from recbole.data.interaction import Interaction

class ItemHistory(BaseModel):
    sequence: List[str]
    topk: int

def pred_user_to_item_batch(sequences: List[List[str]], topk: int = 50, device=None):
    """
    Batch inference cho nhiều sessions cùng lúc - tận dụng GPU tối đa.
    
    Args:
        sequences: List of item sequences (mỗi sequence là list of aid strings)
        topk: Số lượng items top-k cần lấy
        device: Device để chạy model (mặc định là model.device)
    
    Returns:
        List of dicts: Mỗi dict chứa 'score_list' và 'item_list' cho một session
    """
    if device is None:
        device = model.device
    
    batch_size = len(sequences)
    if batch_size == 0:
        return []
    
    pad_length = MAX_ITEM
    
    # Pad tất cả sequences và tạo batch tensor
    padded_sequences = []
    item_lengths = []
    
    for seq in sequences:
        item_length = len(seq)
        item_lengths.append(item_length)
        
        # Convert to token IDs
        token_ids = dataset.token2id(dataset.iid_field, seq)
        padded_seq = torch.nn.functional.pad(
            torch.tensor(token_ids, dtype=torch.long),
            (0, pad_length - item_length),
            "constant",
            0,
        )
        padded_sequences.append(padded_seq)
    
    # Stack thành batch tensor
    batch_aid_list = torch.stack(padded_sequences).to(device)  # [batch_size, pad_length]
    batch_item_lengths = torch.tensor(item_lengths, dtype=torch.long).to(device)  # [batch_size]
    
    # Tạo Interaction batch
    input_interaction = Interaction({
        "aid_list": batch_aid_list,
        "item_length": batch_item_lengths,
    })
    
    # Batch prediction trên GPU
    with torch.no_grad():
        scores = model.full_sort_predict(input_interaction)  # [batch_size, item_num]
        scores[:, 0] = -np.inf  # Mask padding token
    
    # Top-k cho mỗi session trong batch
    topk_scores, topk_indices = torch.topk(scores, topk, dim=1)  # [batch_size, topk]
    
    # Convert về list
    results = []
    for i in range(batch_size):
        predicted_item_list = dataset.id2token(
            dataset.iid_field, topk_indices[i].cpu().tolist()
        ).tolist()
        predicted_score_list = topk_scores[i].cpu().tolist()
        
        results.append({
            "score_list": predicted_score_list,
            "item_list": predicted_item_list,
        })
    
    return results

def pred_user_to_item(item_history: ItemHistory):
    """
    Logic giống hệt otto-gru4rec: single session prediction.
    """
    item_history_dict = item_history.dict()
    item_sequence = item_history_dict["sequence"]
    item_length = len(item_sequence)
    pad_length = MAX_ITEM  # pre-defined by recbole

    padded_item_sequence = torch.nn.functional.pad(
        torch.tensor(dataset.token2id(dataset.iid_field, item_sequence)),
        (0, pad_length - item_length),
        "constant",
        0,
    )

    input_interaction = Interaction(
        {
            "aid_list": padded_item_sequence.reshape(1, -1),
            "item_length": torch.tensor([item_length]),
        }
    )
    scores = model.full_sort_predict(input_interaction.to(model.device))
    scores = scores.view(-1, dataset.item_num)
    scores[:, 0] = -np.inf  # pad item score -> -inf
    topk_score, topk_iid_list = torch.topk(scores, item_history_dict["topk"])

    predicted_score_list = topk_score.tolist()[0]
    predicted_item_list = dataset.id2token(
        dataset.iid_field, topk_iid_list.tolist()
    ).tolist()

    recommended_items = {
        "score_list": predicted_score_list,
        "item_list": predicted_item_list,
    }
    return recommended_items


def generate_recommendations_batch(sessions_data: List[Tuple[List[int], List[int]]],
                                   model_topk: int = 20,
                                   batch_size: int = 256) -> List[List[int]]:
    """
    Batch generation recommendations cho nhiều sessions cùng lúc.
    Logic giống otto-gru4rec: nếu session có >= 20 AIDs thì dùng weights, 
    nếu không thì dùng model predictions.
    
    Args:
        sessions_data: List of tuples (AIDs, types) cho mỗi session
        model_topk: Số lượng top-k từ model
        batch_size: Kích thước batch cho inference
    
    Returns:
        List of recommendation lists (mỗi list có 20 items)
    """
    if len(sessions_data) == 0:
        return []
    
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    all_recommendations = []
    
    # Xử lý theo batch
    for batch_start in range(0, len(sessions_data), batch_size):
        batch_end = min(batch_start + batch_size, len(sessions_data))
        batch_sessions = sessions_data[batch_start:batch_end]
        
        batch_labels = []
        
        for AIDs, types in batch_sessions:
            if len(AIDs) >= 20:
                # Nếu có đủ 20 AIDs, dùng logic weights giống hệt otto-gru4rec
                weights = np.logspace(0.1, 1, len(AIDs), base=2, endpoint=True) - 1
                aids_temp = defaultdict(lambda: 0)
                for aid, w, t in zip(AIDs, weights, types):
                    aids_temp[aid] += w * type_weight_multipliers[t]
                
                sorted_aids = [k for k, v in sorted(aids_temp.items(), key=lambda item: -item[1])]
                batch_labels.append(sorted_aids[:20])
            else:
                # Logic giống hệt otto-gru4rec: dùng toàn bộ AIDs (không slice), không convert sang str
                AIDs = list(dict.fromkeys(AIDs))
                item = ItemHistory(sequence=AIDs, topk=model_topk)
                try:
                    nns = [int(v) for v in pred_user_to_item(item)['item_list']]
                except:
                    nns = []

                # Combine logic giống hệt otto-gru4rec
                for word in nns:
                    if len(AIDs) == 20:
                        break
                    if int(word) not in AIDs:
                        AIDs.append(word)

                batch_labels.append(AIDs[:20])
        
        all_recommendations.extend(batch_labels)
    
    return all_recommendations

def generate_recommendations(AIDs: List[int], 
                             types: List[int],
                             model_topk: int = 20) -> List[int]:
    """
    Wrapper cho single session (backward compatibility).
    """
    results = generate_recommendations_batch(
        [(AIDs, types)], 
        model_topk, 
        batch_size=1
    )
    return results[0] if results else []


# Đọc test data
test = pl.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')

session_types = ['clicks', 'carts', 'orders']

# Groupby session để lấy AIDs và types
test_session_AIDs = test.to_pandas().reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types = test.to_pandas().reset_index(drop=True).groupby('session')['type'].apply(list)

del test
gc.collect()

print(f"Total sessions to process: {len(test_session_AIDs)}")


# ====== BATCH INFERENCE ======
# Đảm bảo model đã được đưa lên GPU
print(f"Model device: {model.device}")
print(f"Model is on GPU: {next(model.parameters()).is_cuda}")

# Chuẩn bị sessions data: List of (AIDs, types)
sessions_data = list(zip(test_session_AIDs, test_session_types))

print(f"\nTotal sessions to process: {len(sessions_data):,}")

# Tạo predictions cho tất cả sessions (không phân biệt type, logic giống otto-gru4rec)
BATCH_SIZE = 256  # Xử lý 256 sessions cùng lúc

print(f"\n{'='*50}")
print(f"Generating predictions with batch inference (batch_size={BATCH_SIZE})...")

# Batch inference
labels = generate_recommendations_batch(
    sessions_data=sessions_data,
    model_topk=20,
    batch_size=BATCH_SIZE
)

# Kiểm tra số lượng labels
print(f"Generated {len(labels):,} predictions")
if labels:
    print(f"Sample prediction length: {len(labels[0])} items")
    print(f"Sample prediction: {labels[0][:5]}...")


# Kiểm tra sample predictions
print("\nSample predictions:")
for i in range(3):
    print(f"  Session {i}: {labels[i][:5]}...")


# ====== TẠO SUBMISSION FILE ======
labels_as_strings = [' '.join([str(l) for l in lls]) for lls in labels]

session_ids_list = list(test_session_AIDs.index)

predictions = pd.DataFrame(data={'session_type': session_ids_list, 'labels': labels_as_strings})

prediction_dfs = []

for st in session_types:
    modified_predictions = predictions.copy()
    modified_predictions.session_type = modified_predictions.session_type.astype('str') + f'_{st}'
    prediction_dfs.append(modified_predictions)

submission = pd.concat(prediction_dfs).reset_index(drop=True)

# Kiểm tra final format
print(f"\n{'='*50}")
print("Final submission validation:")
print(f"Total rows: {len(submission):,}")
print(f"Expected rows: {len(session_ids_list) * len(session_types):,}")
print(f"Columns: {submission.columns.tolist()}")

# Kiểm tra mỗi row có đúng format
sample_row = submission.iloc[0]
print(f"\nSample row:")
print(f"  session_type: {sample_row['session_type']}")
print(f"  labels: {sample_row['labels'][:50]}...")
label_count = len(sample_row['labels'].split())
print(f"  Label count: {label_count} (should be 20)")

# Lưu file
submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission saved to submission.csv")
print(f"✓ File contains {len(submission):,} rows")
submission.head(10)




