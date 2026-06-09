import pandas as pd
import numpy as np
from tqdm import tqdm

train_demo = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
train = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
test_demo = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
test = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")


print("æµ‹è¯•é›†åŸºæœ¬ä¿¡æ�¯ï¼š")
test.info()

# æŸ¥çœ‹æ•°æ�®é›†è¡Œæ•°å’Œåˆ—æ•°
rows, columns = test.shape

if rows < 5:
    # æ ·æœ¬è¡Œæ•°å°‘äº�5åˆ™æŸ¥çœ‹å…¨é‡�æ•°æ�®ä¿¡æ�¯
    print("æµ‹è¯•é›†å…¨éƒ¨å†…å®¹ä¿¡æ�¯ï¼š")
    print(test.to_csv(sep='\t', na_rep='nan'))
else:
    # æ ·æœ¬è¡Œæ•°å¤šäº�5åˆ™æŸ¥çœ‹æ•°æ�®å‰�å‡ è¡Œä¿¡æ�¯
    print("æµ‹è¯•é›†å‰�å‡ è¡Œå†…å®¹ä¿¡æ�¯ï¼š")
    print(test.head().to_csv(sep='\t', na_rep='nan'))


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
from datasets import load_dataset, load_from_disk


test.head()


train.head()


train_demo.head()


print(np.shape(train_demo), np.shape(train))
print(np.shape(test_demo), np.shape(test))


print([col for col in train.columns if col not in test.columns])


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class SubjectDataset(Dataset):
    def __init__(self, data_demo, data, have_labels=True):
        self.have_labels = have_labels
        # è�·å�–æ‰€æœ‰å”¯ä¸€åº�åˆ—IDï¼ˆä¿®æ­£å�Ÿæ‹¼å†™é”™è¯¯ï¼‰
        self.unique_sequence_ids = sorted(data["sequence_id"].unique())
        # å»ºç«‹â€œåº�åˆ—ID â†’ å�—è¯•è€…IDâ€�çš„æ˜ å°„
        self.sequence_id_to_subject = data.groupby("sequence_id")["subject"].first().to_dict()
        
        # æ��å�–äººå�£ç»Ÿè®¡æ•°æ�®çš„æ•°å€¼ç‰¹å¾�åˆ—ï¼ˆæ�’é™¤subjectè‡ªèº«ï¼‰
        numerical_demo_cols = data_demo.select_dtypes(include=np.number).columns.tolist()
        if 'subject' in numerical_demo_cols:
            numerical_demo_cols.remove('subject')
        
        # æ�„å»ºâ€œå�—è¯•è€…ID â†’ äººå�£ç»Ÿè®¡ç‰¹å¾�å¼ é‡�â€�çš„æ˜ å°„
        self.subject_to_demographics = {}
        for subject, row in data_demo.set_index('subject').iterrows():
            demo_values = row[numerical_demo_cols].values.astype(np.float32)
            self.subject_to_demographics[subject] = torch.tensor(demo_values)

        # å¤„ç�†æ ‡ç­¾ï¼ˆä»…è®­ç»ƒé›†éœ€è¦�ï¼‰
        self.labels = None
        self.gesture_to_id = None
        if have_labels:
            all_unique_gestures = sorted(data["gesture"].unique())
            self.gesture_to_id = {gesture: i for i, gesture in enumerate(all_unique_gestures)}
            sequence_labels_map = data.groupby("sequence_id")["gesture"].first().to_dict()
            self.labels = torch.tensor(
                [self.gesture_to_id[sequence_labels_map[sid]] for sid in self.unique_sequence_ids],
                dtype=torch.long
            )

        # ç­›é€‰ä¼ æ„Ÿå™¨ç‰¹å¾�åˆ—ï¼ˆæ�’é™¤é��ç‰¹å¾�åˆ—ï¼‰
        cols_to_drop = ["row_id", "subject", "sequence_id", "sequence_type", "orientation", "behavior", "phase", "gesture"]
        self.sequence_feature_cols = [col for col in data.columns if col not in cols_to_drop]
        
        # é¢„å¤„ç�†æ¯�ä¸ªåº�åˆ—çš„ä¼ æ„Ÿå™¨æ•°æ�®ï¼ˆè½¬ç½®ä¸ºâ€œç‰¹å¾�æ•° Ã— æ—¶é—´æ­¥â€�æ ¼å¼�ï¼Œé€‚é…�å��ç»­1Då�·ç§¯ï¼‰
        self.preprocessed_sequences = {}
        print("æ­£åœ¨å¤„ç�†åº�åˆ—æ•°æ�®...")
        for seq_id in tqdm(self.unique_sequence_ids):
            seq_data = data[data["sequence_id"] == seq_id][self.sequence_feature_cols].values.astype(np.float32)
            self.preprocessed_sequences[seq_id] = torch.tensor(seq_data).T  # è½¬ç½®å��å½¢çŠ¶ï¼š(ç‰¹å¾�æ•°, æ—¶é—´æ­¥)

    def __len__(self):
        return len(self.unique_sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.unique_sequence_ids[idx]
        subject = self.sequence_id_to_subject[seq_id]
        # è�·å�–äººå�£ç»Ÿè®¡ç‰¹å¾�å¼ é‡�
        demo_tensor = self.subject_to_demographics[subject]
        # è�·å�–ä¼ æ„Ÿå™¨åº�åˆ—å¼ é‡�
        seq_tensor = self.preprocessed_sequences[seq_id]

        if self.have_labels:
            # è®­ç»ƒ/éªŒè¯�é›†ï¼šè¿”å›�â€œäººå�£ç‰¹å¾� + ä¼ æ„Ÿå™¨åº�åˆ— + æ ‡ç­¾â€�
            label = self.labels[idx]
            return demo_tensor, seq_tensor, label
        else:
            # æµ‹è¯•é›†ï¼šè¿”å›�â€œåº�åˆ—ID + äººå�£ç‰¹å¾� + ä¼ æ„Ÿå™¨åº�åˆ—â€�ï¼ˆç”¨äº�é¢„æµ‹ï¼‰
            return seq_id, demo_tensor, seq_tensor


train_dataset = SubjectDataset(train_demo, train)


test_dataset = SubjectDataset(test_demo, test, have_labels=False)


seed = 42
torch.manual_seed(seed)


from torch.utils.data import random_split
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    """
    è‡ªå®šä¹‰æ‰¹å¤„ç�†å‡½æ•°ï¼šç»Ÿä¸€æ‰¹æ¬¡å†…å�˜é•¿ä¼ æ„Ÿå™¨åº�åˆ—çš„é•¿åº¦ï¼Œå¹¶é€‚é…�æœ‰/æ— æ ‡ç­¾åœºæ™¯
    
    å�‚æ•°:
        batch: å�Ÿå§‹æ•°æ�®æ‰¹æ¬¡ï¼Œæ ¼å¼�ä¸ºï¼š
            - è®­ç»ƒ/éªŒè¯�é›†ï¼š(äººå�£ç»Ÿè®¡ç‰¹å¾�å¼ é‡�, ä¼ æ„Ÿå™¨åº�åˆ—å¼ é‡�, æ ‡ç­¾å¼ é‡�) çš„å…ƒç»„åˆ—è¡¨
            - æµ‹è¯•é›†ï¼š(åº�åˆ—ID, äººå�£ç»Ÿè®¡ç‰¹å¾�å¼ é‡�, ä¼ æ„Ÿå™¨åº�åˆ—å¼ é‡�) çš„å…ƒç»„åˆ—è¡¨
    
    è¿”å›�:
        æ‰¹é‡�åŒ–æ•°æ�®ï¼Œæ ¼å¼�ä¸ºï¼š
            - æœ‰æ ‡ç­¾ï¼š(äººå�£ç»Ÿè®¡ç‰¹å¾�æ‰¹æ¬¡, å¡«å……å��çš„åº�åˆ—æ‰¹æ¬¡, æ ‡ç­¾æ‰¹æ¬¡)
            - æ— æ ‡ç­¾ï¼š(åº�åˆ—IDåˆ—è¡¨, äººå�£ç»Ÿè®¡ç‰¹å¾�æ‰¹æ¬¡, å¡«å……å��çš„åº�åˆ—æ‰¹æ¬¡)
    """
    # åˆ¤æ–­å½“å‰�æ‰¹æ¬¡æ˜¯å�¦åŒ…å�«æ ‡ç­¾ï¼ˆè®­ç»ƒ/éªŒè¯�é›†ç‰¹å¾�æ•°ä¸º3ï¼Œæµ‹è¯•é›†ä¸º2ï¼‰
    has_labels = len(batch[0]) == 3

    # æ‹†åˆ†æ‰¹æ¬¡ä¸­çš„ä¸�å�Œç»„ä»¶
    if has_labels:
        demo_tensors, seq_tensors, label_tensors = zip(*batch)
    else:
        seq_ids, demo_tensors, seq_tensors = zip(*batch)

    # 1. äººå�£ç»Ÿè®¡ç‰¹å¾�æ‰¹é‡�åŒ–ï¼ˆé•¿åº¦å›ºå®šï¼Œç›´æ�¥å †å� ï¼‰
    demo_batch = torch.stack(demo_tensors)

    # 2. ä¼ æ„Ÿå™¨åº�åˆ—å¡«å……ï¼ˆç»Ÿä¸€åˆ°æ‰¹æ¬¡å†…æœ€é•¿åº�åˆ—é•¿åº¦ï¼‰
    # è�·å�–æ‰¹æ¬¡ä¸­æœ€é•¿åº�åˆ—çš„æ—¶é—´æ­¥é•¿åº¦
    max_seq_len = max(seq.shape[1] for seq in seq_tensors)
    
    # å¯¹æ¯�ä¸ªåº�åˆ—è¿›è¡Œå¡«å……
    padded_seqs = []
    for seq in seq_tensors:
        # è®¡ç®—éœ€è¦�å¡«å……çš„é•¿åº¦ï¼ˆæ—¶é—´ç»´åº¦å�³ä¾§è¡¥0ï¼‰
        pad_length = max_seq_len - seq.shape[1]
        if pad_length > 0:
            padded_seq = F.pad(seq, (0, pad_length))  # ä»…åœ¨æ—¶é—´ç»´åº¦è¡¥0
        else:
            padded_seq = seq  # é•¿åº¦è¶³å¤Ÿæ—¶ç›´æ�¥ä½¿ç”¨å�Ÿåº�åˆ—
        padded_seqs.append(padded_seq)
    
    # å †å� å¡«å……å��çš„åº�åˆ—å½¢æˆ�æ‰¹æ¬¡
    seq_batch = torch.stack(padded_seqs)

    # 3. æŒ‰åœºæ™¯è¿”å›�å¯¹åº”æ ¼å¼�çš„æ•°æ�®
    if has_labels:
        label_batch = torch.stack(label_tensors)
        return demo_batch, seq_batch, label_batch
    else:
        return seq_ids, demo_batch, seq_batch


# è¡¥å……å¿…è¦�çš„å¯¼å…¥
import torch
from torch.utils.data import DataLoader  # å…³é”®ï¼šå¯¼å…¥DataLoaderç±»

# æ•°æ�®åŠ è½½å™¨é…�ç½®å�‚æ•°
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windowsç³»ç»Ÿå»ºè®®è®¾ä¸º0ï¼Œé�¿å…�å¤šè¿›ç¨‹é—®é¢˜

# åˆ›å»ºè®­ç»ƒé›†æ•°æ�®åŠ è½½å™¨
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # è®­ç»ƒé›†æ‰“ä¹±é¡ºåº�
    collate_fn=custom_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()  # è‹¥æœ‰GPUï¼Œè‡ªåŠ¨å�¯ç”¨åŠ é€Ÿ
)

# åˆ›å»ºéªŒè¯�é›†æ•°æ�®åŠ è½½å™¨
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # éªŒè¯�é›†ä¸�æ‰“ä¹±é¡ºåº�
    collate_fn=custom_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


import os
import pandas as pd
import numpy as np
import math  # æ–°å¢�ï¼šå¯¼å…¥mathæ¨¡å�—ç”¨äº�æ£€æŸ¥æµ®ç‚¹æ•°
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm, trange
import torch.nn.functional as F
import time

# --------------------------
# 0. å…¨å±€é…�ç½®
# --------------------------
DATA_FOLDER = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
RANDOM_SEED = 42
NUM_WORKERS = 0


# --------------------------
# 1. å·¥å…·å‡½æ•°
# --------------------------
def print_status(message):
    tqdm.write(f"[{time.strftime('%H:%M:%S', time.localtime())}] {message}")


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"æ•°æ�®æ–‡ä»¶ä¸�å­˜åœ¨ï¼Œè¯·æ£€æŸ¥è·¯å¾„ï¼š{file_path}")


def clean_nan_inf(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            fill_val = data[col].mean()
        else:
            fill_val = data[col].mode()[0] if not data[col].mode().empty else ""
        data[col] = data[col].fillna(fill_val)
    return data


# --------------------------
# 2. æ•°æ�®é›†ç±»
# --------------------------
class SubjectDataset(Dataset):
    def __init__(self, data_demo: pd.DataFrame, data: pd.DataFrame, have_labels: bool = False):
        self.have_labels = have_labels
        self.data = clean_nan_inf(data)
        self.unique_sequence_ids = sorted(self.data["sequence_id"].unique())
        self.seq_to_subj = self.data.groupby("sequence_id")["subject"].first().to_dict()
        
        # äººå�£ç‰¹å¾�ï¼šæ˜�ç¡®æŒ‡å®šfloat32
        self.data_demo = clean_nan_inf(data_demo)
        self.numerical_demo_cols = [c for c in self.data_demo.select_dtypes(include=np.number).columns 
                                    if c != "subject"]
        self.demo_mean = self.data_demo[self.numerical_demo_cols].mean()
        self.demo_std = self.data_demo[self.numerical_demo_cols].std() + 1e-8
        self.subj_demo = {
            subj: torch.tensor(
                ((row[self.numerical_demo_cols] - self.demo_mean) / self.demo_std).clip(-5, 5).values,
                dtype=torch.float32
            ) for subj, row in self.data_demo.set_index("subject").iterrows()
        }

        # ä¼ æ„Ÿå™¨ç‰¹å¾�ï¼šè¿‡æ»¤é��æ•°å€¼åˆ—
        self.drop_cols = ["row_id", "subject", "sequence_id", "sequence_type", 
                         "orientation", "behavior", "phase", "gesture"]
        self.feat_cols = [c for c in self.data.columns if c not in self.drop_cols]
        self.feat_cols = [col for col in self.feat_cols if pd.api.types.is_numeric_dtype(self.data[col])]
        print_status(f"ä¼ æ„Ÿå™¨ç‰¹å¾�åˆ—æ•°é‡�ï¼š{len(self.feat_cols)}ï¼ˆåˆ—å��ç¤ºä¾‹ï¼š{self.feat_cols[:3]}...ï¼‰")
        
        # é¢„è®¡ç®—ä¼ æ„Ÿå™¨ç‰¹å¾�å�‡å€¼/æ ‡å‡†å·®
        self.seq_mean = self.data[self.feat_cols].mean()
        self.seq_std = self.data[self.feat_cols].std() + 1e-8
        
        print_status(f"å¼€å§‹å¤„ç�† {len(self.unique_sequence_ids)} ä¸ªåº�åˆ—ï¼ˆæ— å�»å™ª+æ•°æ�®æ¸…æ´—ï¼‰")
        self.sequences = self._process_all_sequences()

    def _process_all_sequences(self):
        sequences = {}
        with tqdm(self.unique_sequence_ids, position=0, leave=False, desc="å¤„ç�†åº�åˆ—") as pbar:
            for seq_id in pbar:
                try:
                    sequences[seq_id] = self._process_single_sequence(seq_id)
                    pbar.set_postfix({"å½“å‰�åº�åˆ—": seq_id}, refresh=False)
                except Exception as e:
                    tqdm.write(f"\nè­¦å‘Šï¼šåº�åˆ— {seq_id} å¤„ç�†å¤±è´¥ï¼ˆ{str(e)}ï¼‰ï¼Œä½¿ç”¨å�‡å€¼å¡«å……")
                    fake_seq = np.full((30, len(self.feat_cols)), self.seq_mean.values, dtype=np.float32)
                    sequences[seq_id] = torch.tensor(fake_seq.T, dtype=torch.float32)
        return sequences

    def _process_single_sequence(self, seq_id: str):
        # æ��å�–åº�åˆ—æ•°æ�®ï¼šå¼ºåˆ¶è½¬ä¸ºfloat32
        seq_data = self.data[self.data["sequence_id"] == seq_id][self.feat_cols].values.astype(np.float32)
        original_len = seq_data.shape[0]
        
        # æ ‡å‡†åŒ–+è£�å‰ª
        seq_data = (seq_data - self.seq_mean.values) / self.seq_std.values
        seq_data = np.clip(seq_data, -5, 5)
        
        # é•¿åº¦å¯¹é½�
        target_len = min(max(original_len, 30), 500)
        seq_data = self._align_sequence_length(seq_data, target_len)
        
        # å¼ºåˆ¶float32
        return torch.tensor(seq_data.T, dtype=torch.float32)

    @staticmethod
    def _align_sequence_length(signal: np.ndarray, target_len: int) -> np.ndarray:
        if len(signal) == target_len:
            return signal
        elif len(signal) > target_len:
            return signal[:target_len]
        else:
            pad_val = np.mean(signal, axis=0).astype(np.float32)
            return np.pad(signal, ((0, target_len - len(signal)), (0, 0)), mode="constant", constant_values=pad_val)

    def __len__(self) -> int:
        return len(self.unique_sequence_ids)

    def __getitem__(self, idx: int) -> tuple:
        seq_id = self.unique_sequence_ids[idx]
        subj_id = self.seq_to_subj[seq_id]
        demo_feat = self.subj_demo[subj_id]
        seq_feat = self.sequences[seq_id]
        
        if self.have_labels:
            label = self.data[self.data["sequence_id"] == seq_id]["gesture"].iloc[0]
            return (seq_id, demo_feat, seq_feat, label)
        return (seq_id, demo_feat, seq_feat)


# --------------------------
# 3. Collateå‡½æ•°
# --------------------------
def custom_collate(batch: list) -> tuple:
    has_labels = len(batch[0]) == 4

    if has_labels:
        seq_ids, demo_list, seq_list, label_list = zip(*batch)
    else:
        seq_ids, demo_list, seq_list = zip(*batch)

    # äººå�£ç‰¹å¾�ï¼šå·²æ˜¯float32ï¼Œç›´æ�¥å †å� 
    demo_batch = torch.stack(demo_list)

    # åº�åˆ—å¡«å……ï¼šå¡«å……å€¼å¼ºåˆ¶float32
    max_seq_len = max(seq.shape[1] for seq in seq_list)
    padded_seqs = []
    for seq in seq_list:
        pad_length = max_seq_len - seq.shape[1]
        if pad_length > 0:
            pad_val = torch.tensor(np.mean(seq.numpy(), axis=1), dtype=torch.float32).unsqueeze(1)
            padded_seq = torch.cat([seq, pad_val.repeat(1, pad_length)], dim=1)
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    seq_batch = torch.stack(padded_seqs)

    if has_labels:
        unique_labels = sorted(set(label_list))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        label_batch = torch.tensor([label_to_id[label] for label in label_list], dtype=torch.long)
        return (seq_ids, demo_batch, seq_batch, label_batch)
    
    return (seq_ids, demo_batch, seq_batch)


# --------------------------
# 4. æ¨¡å�‹
# --------------------------
class HybridClassifier(nn.Module):
    def __init__(self, seq_feat_dim: int, subj_feat_dim: int, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(seq_feat_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2)
        )

        # ä¿®å¤�LSTM dropoutè­¦å‘Šï¼šnum_layers=1æ—¶dropoutæ— æ•ˆï¼Œè®¾ä¸º0
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )

        self.subj_encoder = nn.Sequential(
            nn.Linear(subj_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, mean=0.0, std=0.005)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.01)

    def forward(self, demo_feat: torch.Tensor, seq_feat: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(seq_feat)
        cnn_out = torch.clip(cnn_out, -10, 10)
        
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, (hn, _) = self.lstm(lstm_in)
        seq_final_feat = torch.clip(hn[-1], -10, 10)
        
        subj_final_feat = self.subj_encoder(demo_feat)
        subj_final_feat = torch.clip(subj_final_feat, -10, 10)
        
        combined_feat = torch.cat([seq_final_feat, subj_final_feat], dim=1)
        combined_feat = torch.clip(combined_feat, -10, 10)
        
        return self.classifier(combined_feat)


# --------------------------
# 5. è®­ç»ƒå‡½æ•°ï¼ˆæ ¸å¿ƒä¿®å¤�ï¼šç”¨math.isfiniteæ£€æŸ¥æµ®ç‚¹æ•°ï¼‰
# --------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int, device: torch.device, lr: float = 1e-5) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    model.to(device)
    best_val_loss = float("inf")
    best_model_state = model.state_dict().copy()
    torch.save(best_model_state, "initial_model.pth")

    with trange(num_epochs, position=0, leave=True, desc="è®­ç»ƒæ€»è¿›åº¦") as epoch_pbar:
        for epoch in epoch_pbar:
            model.train()
            train_total_loss = 0.0
            train_total_acc = 0.0
            train_total_samples = 0

            with tqdm(train_loader, position=1, leave=False, 
                      desc=f"Epoch {epoch+1}/{num_epochs} [è®­ç»ƒ]") as batch_pbar:
                for batch in batch_pbar:
                    _, demo_batch, seq_batch, label_batch = batch
                    demo_batch = demo_batch.to(device)
                    seq_batch = seq_batch.to(device)
                    label_batch = label_batch.to(device)

                    optimizer.zero_grad()
                    outputs = model(demo_batch, seq_batch)
                    
                    if torch.isnan(outputs).any():
                        tqdm.write("è­¦å‘Šï¼šæ¨¡å�‹è¾“å‡ºå�«NaNï¼Œè·³è¿‡æœ¬è½®æ›´æ–°")
                        continue
                    
                    loss = criterion(outputs, label_batch)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    loss.backward()
                    optimizer.step()

                    # ä¿®å¤�1ï¼šç”¨math.isfiniteæ£€æŸ¥floatç±»å�‹çš„loss
                    if not math.isfinite(loss.item()):
                        tqdm.write(f"è­¦å‘Šï¼šç¬¬{epoch+1}è½®æ�Ÿå¤±å¼‚å¸¸ï¼Œè·³è¿‡")
                        continue

                    batch_size = demo_batch.size(0)
                    train_total_loss += loss.item() * batch_size
                    preds = outputs.argmax(dim=1)
                    correct = (preds == label_batch).sum().item()
                    train_total_acc += correct
                    train_total_samples += batch_size

                    batch_pbar.set_postfix({
                        "batch_loss": f"{loss.item():.4f}",
                        "batch_acc": f"{correct/batch_size:.4f}"
                    }, refresh=False)

            if train_total_samples == 0:
                tqdm.write("è­¦å‘Šï¼šè®­ç»ƒé›†æ ·æœ¬æ•°ä¸º0ï¼Œè·³è¿‡æœ¬è½®")
                continue

            train_avg_loss = train_total_loss / train_total_samples
            train_avg_acc = train_total_acc / train_total_samples

            # éªŒè¯�é˜¶æ®µ
            model.eval()
            val_total_loss = 0.0
            val_total_acc = 0.0
            val_total_samples = 0

            with torch.no_grad():
                with tqdm(val_loader, position=1, leave=False, 
                          desc=f"Epoch {epoch+1}/{num_epochs} [éªŒè¯�]") as batch_pbar:
                    for batch in batch_pbar:
                        _, demo_batch, seq_batch, label_batch = batch
                        demo_batch = demo_batch.to(device)
                        seq_batch = seq_batch.to(device)
                        label_batch = label_batch.to(device)

                        outputs = model(demo_batch, seq_batch)
                        if torch.isnan(outputs).any():
                            tqdm.write("è­¦å‘Šï¼šéªŒè¯�è¾“å‡ºå�«NaNï¼Œç”¨0å¡«å……")
                            outputs = torch.zeros_like(outputs, dtype=torch.float32)
                        
                        loss = criterion(outputs, label_batch)
                        # ä¿®å¤�2ï¼šç”¨math.isfiniteæ£€æŸ¥floatç±»å�‹çš„loss
                        if not math.isfinite(loss.item()):
                            loss = torch.tensor(10.0, device=device, dtype=torch.float32)

                        batch_size = demo_batch.size(0)
                        val_total_loss += loss.item() * batch_size
                        preds = outputs.argmax(dim=1)
                        correct = (preds == label_batch).sum().item()
                        val_total_acc += correct
                        val_total_samples += batch_size

                        batch_pbar.set_postfix({
                            "val_batch_loss": f"{loss.item():.4f}",
                            "val_batch_acc": f"{correct/batch_size:.4f}"
                        }, refresh=False)

            if val_total_samples == 0:
                tqdm.write("è­¦å‘Šï¼šéªŒè¯�é›†æ ·æœ¬æ•°ä¸º0ï¼Œè·³è¿‡æœ¬è½®")
                continue

            val_avg_loss = val_total_loss / val_total_samples
            val_avg_acc = val_total_acc / val_total_samples

            scheduler.step(val_avg_loss)

            # ä¿®å¤�3ï¼šç”¨math.isfiniteæ£€æŸ¥floatç±»å�‹çš„val_avg_loss
            if val_avg_loss < best_val_loss and math.isfinite(val_avg_loss):
                best_val_loss = val_avg_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, "best_hybrid_classifier.pth")
                print_status(f"Epoch {epoch+1}ï¼šä¿�å­˜æœ€ä½³æ¨¡å�‹ï¼ˆéªŒè¯�æ�Ÿå¤±ï¼š{val_avg_loss:.4f}ï¼‰")

            epoch_pbar.set_postfix({
                "train_loss": f"{train_avg_loss:.4f}",
                "train_acc": f"{train_avg_acc:.4f}",
                "val_loss": f"{val_avg_loss:.4f}",
                "val_acc": f"{val_avg_acc:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.8f}"
            }, refresh=True)

            print_status(f"Epoch {epoch+1}/{num_epochs} æ€»ç»“ï¼š"
                         f"è®­ç»ƒæ�Ÿå¤± {train_avg_loss:.4f}ï¼ˆacc {train_avg_acc:.4f}ï¼‰| "
                         f"éªŒè¯�æ�Ÿå¤± {val_avg_loss:.4f}ï¼ˆacc {val_avg_acc:.4f}ï¼‰")

    model.load_state_dict(best_model_state)
    return model


# --------------------------
# 6. é¢„æµ‹å‡½æ•°
# --------------------------
def predict_test(model: nn.Module, test_loader: DataLoader, device: torch.device, 
                 label_map: dict) -> pd.DataFrame:
    model.eval()
    model.to(device)
    all_seq_ids = []
    all_preds = []

    with torch.no_grad(), tqdm(test_loader, position=0, leave=True, 
                               desc="æµ‹è¯•é›†é¢„æµ‹") as pbar:
        for batch in pbar:
            seq_ids, demo_batch, seq_batch = batch
            demo_batch = demo_batch.to(device)
            seq_batch = seq_batch.to(device)

            outputs = model(demo_batch, seq_batch)
            if torch.isnan(outputs).any():
                tqdm.write("è­¦å‘Šï¼šé¢„æµ‹è¾“å‡ºå�«NaNï¼Œç”¨å¤šæ•°ç±»å¡«å……")
                outputs = torch.zeros_like(outputs, dtype=torch.float32)
                outputs[:, 0] = 1.0
            
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_seq_ids.extend(seq_ids)
            all_preds.extend(preds)

            pbar.set_postfix({"å·²å¤„ç�†åº�åˆ—": len(all_seq_ids)}, refresh=False)

    submission_df = pd.DataFrame({
        "sequence_id": all_seq_ids,
        "predicted_gesture": [label_map[pred_id] for pred_id in all_preds]
    }).drop_duplicates(subset="sequence_id")

    submission_df.to_parquet("cmi_behavior_prediction_submission.parquet", index=False)
    print_status(f"é¢„æµ‹å®Œæˆ�ï¼�æ��äº¤æ–‡ä»¶å·²ä¿�å­˜ï¼ˆå…± {len(submission_df)} æ�¡è®°å½•ï¼‰")
    print_status(f"æ��äº¤æ–‡ä»¶å‰�5è¡Œç¤ºä¾‹ï¼š\n{submission_df.head().to_string(index=False)}")

    return submission_df


# --------------------------
# 7. ä¸»ç¨‹åº�
# --------------------------
def main():
    try:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # 1. åŠ è½½æ•°æ�®
        print_status("=== å¼€å§‹åŠ è½½æ•°æ�® ===")
        train_demo_path = os.path.join(DATA_FOLDER, "train_demographics.csv")
        train_data_path = os.path.join(DATA_FOLDER, "train.csv")
        test_demo_path = os.path.join(DATA_FOLDER, "test_demographics.csv")
        test_data_path = os.path.join(DATA_FOLDER, "test.csv")

        check_file_exists(train_demo_path)
        check_file_exists(train_data_path)
        check_file_exists(test_demo_path)
        check_file_exists(test_data_path)

        train_demo = pd.read_csv(train_demo_path)
        train_data = pd.read_csv(train_data_path)
        test_demo = pd.read_csv(test_demo_path)
        test_data = pd.read_csv(test_data_path)

        print_status(f"æ•°æ�®åŠ è½½å®Œæˆ�ï¼šè®­ç»ƒé›†åº�åˆ—æ•° {train_data['sequence_id'].nunique()} | "
                     f"æµ‹è¯•é›†åº�åˆ—æ•° {test_data['sequence_id'].nunique()} | "
                     f"è¡Œä¸ºç±»åˆ«æ•° {train_data['gesture'].nunique()}")

        # 2. åˆ›å»ºæ•°æ�®é›†
        print_status("\n=== å¼€å§‹åˆ›å»ºæ•°æ�®é›† ===")
        full_train_dataset = SubjectDataset(
            data_demo=train_demo,
            data=train_data,
            have_labels=True
        )
        
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )

        test_dataset = SubjectDataset(
            data_demo=test_demo,
            data=test_data,
            have_labels=False
        )

        print_status(f"æ•°æ�®é›†æ‹†åˆ†å®Œæˆ�ï¼šè®­ç»ƒé›† {len(train_dataset)} æ ·æœ¬ | "
                     f"éªŒè¯�é›† {len(val_dataset)} æ ·æœ¬ | "
                     f"æµ‹è¯•é›† {len(test_dataset)} æ ·æœ¬")

        # 3. åˆ›å»ºæ•°æ�®åŠ è½½å™¨
        print_status("\n=== å¼€å§‹åˆ›å»ºæ•°æ�®åŠ è½½å™¨ ===")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            drop_last=False
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            drop_last=False
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            drop_last=False
        )

        # 4. åˆ�å§‹åŒ–æ¨¡å�‹
        print_status("\n=== å¼€å§‹åˆ�å§‹åŒ–æ¨¡å�‹ ===")
        train_sample = train_dataset[0]
        _, demo_sample, seq_sample, _ = train_sample
        seq_feat_dim = seq_sample.shape[0]
        subj_feat_dim = demo_sample.shape[0]
        num_classes = train_data["gesture"].nunique()

        model = HybridClassifier(
            seq_feat_dim=seq_feat_dim,
            subj_feat_dim=subj_feat_dim,
            num_classes=num_classes
        )

        print_status(f"æ¨¡å�‹åˆ�å§‹åŒ–å®Œæˆ�ï¼šä¼ æ„Ÿå™¨ç‰¹å¾�ç»´åº¦ {seq_feat_dim} | "
                     f"äººå�£ç‰¹å¾�ç»´åº¦ {subj_feat_dim} | ç±»åˆ«æ•° {num_classes}")
        print_status(f"æ¨¡å�‹æ€»å�‚æ•°é‡�ï¼š{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # 5. è®­ç»ƒè®¾å¤‡
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_status(f"\n=== è®­ç»ƒè®¾å¤‡ï¼š{device} ===")
        if torch.cuda.is_available():
            print_status(f"GPUå�‹å�·ï¼š{torch.cuda.get_device_name(0)}")

        # 6. è®­ç»ƒæ¨¡å�‹
        print_status("\n=== å¼€å§‹è®­ç»ƒæ¨¡å�‹ ===")
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=NUM_EPOCHS,
            device=device,
            lr=LEARNING_RATE
        )

        # 7. é¢„æµ‹
        print_status("\n=== å¼€å§‹æµ‹è¯•é›†é¢„æµ‹ ===")
        unique_gestures = sorted(train_data["gesture"].unique())
        id_to_gesture = {idx: gesture for idx, gesture in enumerate(unique_gestures)}
        submission_df = predict_test(
            model=model,
            test_loader=test_loader,
            device=device,
            label_map=id_to_gesture
        )

        print_status("\n=== å…¨éƒ¨æµ�ç¨‹å®Œæˆ� ===")
        return submission_df

    except Exception as e:
        print_status(f"\nç¨‹åº�è¿�è¡Œå‡ºé”™ï¼š{str(e)}")
        raise
 

if __name__ == "__main__":
    submission = main()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split  # ä¿®æ­£ï¼šDatasetä»�æ­£ç¡®æ¨¡å�—å¯¼å…¥
from tqdm import tqdm
import math

# --------------------------
# 1. å…¨å±€é…�ç½®ï¼ˆä¸�è®­ç»ƒä»£ç �å®Œå…¨ä¸€è‡´ï¼Œé�¿å…�å�‚æ•° mismatchï¼‰
# --------------------------
# è¯·æ ¹æ�®ä½ çš„å®�é™…æ–‡ä»¶è·¯å¾„ä¿®æ”¹ï¼�ï¼�ï¼�
PRED_FILE_PATH = "cmi_behavior_prediction_submission.parquet"  # è®­ç»ƒä»£ç �ç”Ÿæˆ�çš„é¢„æµ‹æ–‡ä»¶
MODEL_WEIGHT_PATH = "best_hybrid_classifier.pth"  # è®­ç»ƒä»£ç �ä¿�å­˜çš„æœ€ä½³æ¨¡å�‹æ�ƒé‡�
DATA_FOLDER = "/kaggle/input/cmi-detect-behavior-with-sensor-data"  # æ•°æ�®æ ¹è·¯å¾„

# è®­ç»ƒç›¸å…³å�‚æ•°ï¼ˆä¸�è®­ç»ƒä»£ç �ä¿�æŒ�ä¸€è‡´ï¼‰
BATCH_SIZE = 16
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"è¯„ä¼°è®¾å¤‡ï¼š{DEVICE}")


# --------------------------
# 2. å·¥å…·å‡½æ•°ï¼ˆä¸�è®­ç»ƒä»£ç �ä¸€è‡´ï¼Œç¡®ä¿�æ•°æ�®å¤„ç�†é€»è¾‘ç»Ÿä¸€ï¼‰
# --------------------------
def clean_nan_inf(data: pd.DataFrame) -> pd.DataFrame:
    """æ¸…æ´—æ•°æ�®ä¸­çš„NaN/Infï¼Œä¸�è®­ç»ƒæ—¶ä¸€è‡´"""
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            fill_val = data[col].mean()
        else:
            fill_val = data[col].mode()[0] if not data[col].mode().empty else ""
        data[col] = data[col].fillna(fill_val)
    return data


def process_anomalies(tensor: torch.Tensor) -> torch.Tensor:
    """å¤„ç�†å¼ é‡�ä¸­çš„å¼‚å¸¸å€¼ï¼Œä¸�è®­ç»ƒæ—¶ä¸€è‡´"""
    tensor[torch.isnan(tensor)] = 0.0
    tensor[torch.isinf(tensor)] = 0.0
    return tensor


# --------------------------
# 3. æ•°æ�®é›†ç±»ï¼ˆä¿®æ­£ç»§æ‰¿é”™è¯¯ï¼Œä¸�è®­ç»ƒä»£ç �å®Œå…¨ä¸€è‡´ï¼‰
# --------------------------
class SubjectDataset(Dataset):  # å…³é”®ä¿®æ­£ï¼šç»§æ‰¿Datasetè€Œé��nn.Module
    def __init__(self, data_demo: pd.DataFrame, data_sensor: pd.DataFrame, have_labels: bool = False):
        self.have_labels = have_labels
        self.data_sensor = clean_nan_inf(data_sensor)  # ä¸�è®­ç»ƒä¸€è‡´ï¼šå…ˆæ¸…æ´—æ•°æ�®
        self.unique_sequence_ids = sorted(self.data_sensor["sequence_id"].unique())
        self.seq_to_subj = self.data_sensor.groupby("sequence_id")["subject"].first().to_dict()
        
        # 1. äººå�£ç»Ÿè®¡ç‰¹å¾�ï¼šæ ‡å‡†åŒ–+float32ï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self.data_demo = clean_nan_inf(data_demo)
        self.numerical_demo_cols = [c for c in self.data_demo.select_dtypes(include=np.number).columns 
                                    if c != "subject"]
        # è®¡ç®—å�‡å€¼/æ ‡å‡†å·®ï¼ˆä¸�è®­ç»ƒä¸€è‡´çš„æ ‡å‡†åŒ–é€»è¾‘ï¼‰
        self.demo_mean = self.data_demo[self.numerical_demo_cols].mean()
        self.demo_std = self.data_demo[self.numerical_demo_cols].std() + 1e-8
        self.subj_demo = {
            subj: torch.tensor(
                ((row[self.numerical_demo_cols] - self.demo_mean) / self.demo_std).clip(-5, 5).values,
                dtype=torch.float32
            ) for subj, row in self.data_demo.set_index("subject").iterrows()
        }

        # 2. ä¼ æ„Ÿå™¨ç‰¹å¾�ï¼šè¿‡æ»¤é��æ•°å€¼åˆ—+é¢„è®¡ç®—å…¨å±€å�‡å€¼ï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self.drop_cols = ["row_id", "subject", "sequence_id", "sequence_type", 
                         "orientation", "behavior", "phase", "gesture"]
        self.feat_cols = [c for c in self.data_sensor.columns if c not in self.drop_cols]
        self.feat_cols = [col for col in self.feat_cols if pd.api.types.is_numeric_dtype(self.data_sensor[col])]
        self.seq_mean = self.data_sensor[self.feat_cols].mean()
        self.seq_std = self.data_sensor[self.feat_cols].std() + 1e-8

        # 3. æ ‡ç­¾å¤„ç�†ï¼ˆä»…è®­ç»ƒ/éªŒè¯�é›†ï¼‰
        self.labels = None
        if have_labels:
            # æŒ‰åº�åˆ—IDè�·å�–æ ‡ç­¾ï¼Œä¸�è®­ç»ƒä¸€è‡´
            seq_label_map = self.data_sensor.groupby("sequence_id")["gesture"].first().to_dict()
            # ç”Ÿæˆ�ç±»åˆ«æ˜ å°„ï¼ˆç¡®ä¿�ä¸�è®­ç»ƒæ—¶æ�’åº�ä¸€è‡´ï¼‰
            global gesture_to_id, id_to_gesture
            gestures = sorted(self.data_sensor["gesture"].unique())
            gesture_to_id = {g: i for i, g in enumerate(gestures)}
            id_to_gesture = {i: g for g, i in gesture_to_id.items()}
            # è½¬æ�¢ä¸ºtensoræ ‡ç­¾
            self.labels = torch.tensor(
                [gesture_to_id[seq_label_map[sid]] for sid in self.unique_sequence_ids],
                dtype=torch.long
            )

        # 4. é¢„å¤„ç�†æ‰€æœ‰ä¼ æ„Ÿå™¨åº�åˆ—ï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼šæ ‡å‡†åŒ–+é•¿åº¦å¯¹é½�ï¼‰
        print(f"åŠ è½½ {len(self.unique_sequence_ids)} ä¸ªåº�åˆ—...")
        self.sequences = self._process_all_sequences()

    def _process_all_sequences(self):
        sequences = {}
        for seq_id in tqdm(self.unique_sequence_ids, desc="å¤„ç�†ä¼ æ„Ÿå™¨åº�åˆ—"):
            try:
                sequences[seq_id] = self._process_single_sequence(seq_id)
            except Exception as e:
                tqdm.write(f"è­¦å‘Šï¼šåº�åˆ— {seq_id} å¤„ç�†å¤±è´¥ï¼ˆ{str(e)}ï¼‰ï¼Œç”¨å�‡å€¼å¡«å……")
                fake_seq = np.full((30, len(self.feat_cols)), self.seq_mean.values, dtype=np.float32)
                sequences[seq_id] = torch.tensor(fake_seq.T, dtype=torch.float32)
        return sequences

    def _process_single_sequence(self, seq_id: str):
        # æ��å�–åº�åˆ—æ•°æ�®
        seq_data = self.data_sensor[self.data_sensor["sequence_id"] == seq_id][self.feat_cols].values.astype(np.float32)
        # æ ‡å‡†åŒ–+è£�å‰ªï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼‰
        seq_data = (seq_data - self.seq_mean.values) / self.seq_std.values
        seq_data = np.clip(seq_data, -5, 5)
        # é•¿åº¦å¯¹é½�ï¼ˆ30-500æ­¥ï¼Œä¸�è®­ç»ƒä¸€è‡´ï¼‰
        target_len = min(max(seq_data.shape[0], 30), 500)
        if seq_data.shape[0] > target_len:
            seq_data = seq_data[:target_len]
        elif seq_data.shape[0] < target_len:
            pad_val = np.mean(seq_data, axis=0)
            seq_data = np.pad(seq_data, ((0, target_len - seq_data.shape[0]), (0, 0)), mode="constant", constant_values=pad_val)
        # è½¬ç½®ä¸º [ç‰¹å¾�æ•°, æ—¶é—´æ­¥]ï¼ˆé€‚é…�CNNè¾“å…¥ï¼‰
        return torch.tensor(seq_data.T, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.unique_sequence_ids)

    def __getitem__(self, idx: int) -> tuple:
        seq_id = self.unique_sequence_ids[idx]
        demo_feat = self.subj_demo[self.seq_to_subj[seq_id]]
        seq_feat = self.sequences[seq_id]
        if self.have_labels:
            return (demo_feat, seq_feat, self.labels[idx])
        return (seq_id, demo_feat, seq_feat)


# --------------------------
# 4. è‡ªå®šä¹‰Collateå‡½æ•°ï¼ˆä¸�è®­ç»ƒä»£ç �å®Œå…¨ä¸€è‡´ï¼Œå¤„ç�†å�˜é•¿åº�åˆ—ï¼‰
# --------------------------
def custom_collate(batch: list) -> tuple:
    has_labels = len(batch[0]) == 3  # è®­ç»ƒ/éªŒè¯�é›†ï¼ˆ3å…ƒç´ ï¼‰vs æµ‹è¯•é›†ï¼ˆ3å…ƒç´ ï¼šseq_id+2ç‰¹å¾�ï¼‰

    if has_labels:
        demo_list, seq_list, label_list = zip(*batch)
    else:
        seq_ids, demo_list, seq_list = zip(*batch)

    # 1. äººå�£ç‰¹å¾�ï¼šç›´æ�¥å †å� ï¼ˆå›ºå®šé•¿åº¦ï¼‰
    demo_batch = torch.stack(demo_list)

    # 2. ä¼ æ„Ÿå™¨åº�åˆ—ï¼šæŒ‰æ‰¹æ¬¡æœ€é•¿åº�åˆ—å¡«å……ï¼ˆå�‡å€¼å¡«å……ï¼Œä¸�è®­ç»ƒä¸€è‡´ï¼‰
    max_seq_len = max(seq.shape[1] for seq in seq_list)
    padded_seqs = []
    for seq in seq_list:
        pad_length = max_seq_len - seq.shape[1]
        if pad_length > 0:
            pad_val = torch.tensor(np.mean(seq.numpy(), axis=1), dtype=torch.float32).unsqueeze(1)
            padded_seq = torch.cat([seq, pad_val.repeat(1, pad_length)], dim=1)
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    seq_batch = torch.stack(padded_seqs)

    # 3. æ ‡ç­¾å¤„ç�†
    if has_labels:
        return (demo_batch, seq_batch, torch.stack(label_list))
    return (seq_ids, demo_batch, seq_batch)


# --------------------------
# 5. æ¨¡å�‹ç»“æ�„ï¼ˆä¸�æœ€æ–°è®­ç»ƒä»£ç �å®Œå…¨ä¸€è‡´ï¼Œç¡®ä¿�æ�ƒé‡�åŠ è½½åŒ¹é…�ï¼‰
# --------------------------
class HybridClassifier(nn.Module):
    def __init__(self, seq_feat_dim: int, subj_feat_dim: int, num_classes: int):
        super().__init__()
        # CNNæ¨¡å�—ï¼ˆç®€åŒ–ç‰ˆï¼Œä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self.cnn = nn.Sequential(
            nn.Conv1d(seq_feat_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2)
        )

        # LSTMæ¨¡å�—ï¼ˆ1å±‚ï¼Œä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # å�•å±‚ç¦�ç”¨dropoutï¼Œä¸�è®­ç»ƒä¸€è‡´
            bidirectional=False
        )

        # äººå�£ç‰¹å¾�ç¼–ç �å™¨ï¼ˆç®€åŒ–ç‰ˆï¼Œä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self.subj_encoder = nn.Sequential(
            nn.Linear(subj_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # åˆ†ç±»å™¨ï¼ˆç®€åŒ–ç‰ˆï¼Œä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # æ�ƒé‡�åˆ�å§‹åŒ–ï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼‰
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.005)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, mean=0.0, std=0.005)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.01)

    def forward(self, demo_feat: torch.Tensor, seq_feat: torch.Tensor) -> torch.Tensor:
        # ä¸�è®­ç»ƒä¸€è‡´çš„å‰�å�‘ä¼ æ’­é€»è¾‘
        cnn_out = self.cnn(seq_feat)
        cnn_out = torch.clip(cnn_out, -10, 10)
        
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, (hn, _) = self.lstm(lstm_in)
        seq_final_feat = torch.clip(hn[-1], -10, 10)
        
        subj_final_feat = self.subj_encoder(demo_feat)
        subj_final_feat = torch.clip(subj_final_feat, -10, 10)
        
        combined_feat = torch.cat([seq_final_feat, subj_final_feat], dim=1)
        combined_feat = torch.clip(combined_feat, -10, 10)
        
        return self.classifier(combined_feat)


# --------------------------
# 6. æ ¸å¿ƒè¯„ä¼°æµ�ç¨‹ï¼ˆåˆ†æ­¥éª¤ï¼šæ•°æ�®åŠ è½½â†’æ¨¡å�‹åŠ è½½â†’éªŒè¯�é›†è¯„ä¼°â†’æµ‹è¯•é›†åˆ†æ��ï¼‰
# --------------------------
def main():
    # --------------------------
    # æ­¥éª¤1ï¼šåŠ è½½åŸºç¡€æ•°æ�®ï¼ˆè®­ç»ƒ/éªŒè¯�/æµ‹è¯•ï¼‰
    # --------------------------
    print("\n" + "="*60)
    print("æ­¥éª¤1ï¼šåŠ è½½æ•°æ�®å¹¶åˆ�å§‹åŒ–æ•°æ�®é›†")
    print("="*60)
    try:
        # åŠ è½½è®­ç»ƒé›†ï¼ˆç”¨äº�åˆ’åˆ†éªŒè¯�é›†ï¼‰å’Œæµ‹è¯•é›†
        train_demo = pd.read_csv(f"{DATA_FOLDER}/train_demographics.csv")
        train_sensor = pd.read_csv(f"{DATA_FOLDER}/train.csv")
        test_demo = pd.read_csv(f"{DATA_FOLDER}/test_demographics.csv")
        test_sensor = pd.read_csv(f"{DATA_FOLDER}/test.csv")
        # åŠ è½½é¢„æµ‹ç»“æ�œ
        pred_df = pd.read_parquet(PRED_FILE_PATH)
        print(f"âœ… æ•°æ�®åŠ è½½å®Œæˆ�ï¼š")
        print(f"  - è®­ç»ƒé›†åº�åˆ—æ•°ï¼š{train_sensor['sequence_id'].nunique()}")
        print(f"  - æµ‹è¯•é›†åº�åˆ—æ•°ï¼š{test_sensor['sequence_id'].nunique()}")
        print(f"  - é¢„æµ‹ç»“æ�œåº�åˆ—æ•°ï¼š{pred_df['sequence_id'].nunique()}")
    except Exception as e:
        raise ValueError(f"â�Œ æ•°æ�®åŠ è½½å¤±è´¥ï¼š{str(e)}ï¼Œè¯·æ£€æŸ¥æ–‡ä»¶è·¯å¾„ï¼�")

    # åˆ�å§‹åŒ–è®­ç»ƒæ•°æ�®é›†ï¼ˆå�«æ ‡ç­¾ï¼‰ï¼Œç”¨äº�åˆ’åˆ†éªŒè¯�é›†
    full_train_dataset = SubjectDataset(
        data_demo=train_demo,
        data_sensor=train_sensor,
        have_labels=True
    )
    # åˆ’åˆ†éªŒè¯�é›†ï¼ˆä¸�è®­ç»ƒä»£ç �å®Œå…¨ä¸€è‡´ï¼š8:2æ‹†åˆ†+å›ºå®šç§�å­�ï¼‰
    torch.manual_seed(RANDOM_SEED)
    train_size = int(0.8 * len(full_train_dataset))
    val_dataset = random_split(full_train_dataset, [train_size, len(full_train_dataset)-train_size])[1]
    # åˆ�å§‹åŒ–éªŒè¯�é›†åŠ è½½å™¨
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
        pin_memory=False
    )
    print(f"âœ… éªŒè¯�é›†åˆ�å§‹åŒ–å®Œæˆ�ï¼š{len(val_dataset)} ä¸ªæ ·æœ¬")

    # --------------------------
    # æ­¥éª¤2ï¼šåŠ è½½æ¨¡å�‹æ�ƒé‡�å¹¶åˆ�å§‹åŒ–æ¨¡å�‹
    # --------------------------
    print("\n" + "="*60)
    print("æ­¥éª¤2ï¼šåŠ è½½æ¨¡å�‹æ�ƒé‡�å¹¶éªŒè¯�")
    print("="*60)
    try:
        # ä»�éªŒè¯�é›†æ ·æœ¬è�·å�–è¾“å…¥ç»´åº¦ï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼‰
        sample_demo, sample_seq, _ = val_dataset[0]
        seq_feat_dim = sample_seq.shape[0]  # ä¼ æ„Ÿå™¨ç‰¹å¾�ç»´åº¦ï¼ˆå¦‚333ï¼‰
        subj_feat_dim = sample_demo.shape[0]  # äººå�£ç‰¹å¾�ç»´åº¦ï¼ˆå¦‚7ï¼‰
        num_classes = len(gesture_to_id)  # ç±»åˆ«æ•°ï¼ˆå¦‚18ï¼‰
        
        # åˆ�å§‹åŒ–æ¨¡å�‹ï¼ˆä¸�è®­ç»ƒä¸€è‡´çš„å�‚æ•°ï¼‰
        model = HybridClassifier(
            seq_feat_dim=seq_feat_dim,
            subj_feat_dim=subj_feat_dim,
            num_classes=num_classes
        )
        # åŠ è½½æ�ƒé‡�
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()  # åˆ‡æ�¢è¯„ä¼°æ¨¡å¼�ï¼ˆç¦�ç”¨Dropout/BatchNormæ›´æ–°ï¼‰
        print(f"âœ… æ¨¡å�‹åŠ è½½å®Œæˆ�ï¼š")
        print(f"  - ä¼ æ„Ÿå™¨ç‰¹å¾�ç»´åº¦ï¼š{seq_feat_dim}")
        print(f"  - äººå�£ç‰¹å¾�ç»´åº¦ï¼š{subj_feat_dim}")
        print(f"  - ç±»åˆ«æ•°ï¼š{num_classes}")
        print(f"  - æ¨¡å�‹æ€»å�‚æ•°é‡�ï¼š{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        raise ValueError(f"â�Œ æ¨¡å�‹åŠ è½½å¤±è´¥ï¼š{str(e)}ï¼Œè¯·ç¡®ä¿�æ�ƒé‡�æ–‡ä»¶è·¯å¾„æ­£ç¡®ä¸”æ¨¡å�‹ç»“æ�„ä¸�è®­ç»ƒä¸€è‡´ï¼�")

    # --------------------------
    # æ­¥éª¤3ï¼šéªŒè¯�é›†è¯„ä¼°ï¼ˆæ ¸å¿ƒæŒ‡æ ‡ï¼šå‡†ç¡®ç�‡ã€�F1ã€�æ··æ·†çŸ©é˜µï¼‰
    # --------------------------
    print("\n" + "="*60)
    print("æ­¥éª¤3ï¼šéªŒè¯�é›†æ€§èƒ½è¯„ä¼°")
    print("="*60)
    # åˆ�å§‹åŒ–æ�Ÿå¤±å‡½æ•°ï¼ˆä¸�è®­ç»ƒä¸€è‡´ï¼‰
    criterion = nn.CrossEntropyLoss()
    # å­˜å‚¨æ‰€æœ‰çœŸå®�æ ‡ç­¾å’Œé¢„æµ‹æ ‡ç­¾
    all_true = []
    all_pred = []
    total_loss = 0.0

    with torch.no_grad():  # å…³é—­æ¢¯åº¦è®¡ç®—ï¼ŒèŠ‚çœ�å†…å­˜
        for batch in tqdm(val_loader, desc="è¯„ä¼°éªŒè¯�é›†"):
            demo_batch, seq_batch, label_batch = batch
            # æ•°æ�®ç§»è‡³è®¾å¤‡å¹¶å¤„ç�†å¼‚å¸¸å€¼
            demo_batch = demo_batch.to(DEVICE)
            seq_batch = process_anomalies(seq_batch).to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            
            # æ¨¡å�‹é¢„æµ‹
            outputs = model(demo_batch, seq_batch)
            loss = criterion(outputs, label_batch)
            total_loss += loss.item()
            
            # è½¬æ�¢ä¸ºnumpyæ ¼å¼�ï¼Œä¾¿äº�è®¡ç®—æŒ‡æ ‡
            preds = outputs.argmax(dim=1).cpu().numpy()
            trues = label_batch.cpu().numpy()
            all_true.extend(trues)
            all_pred.extend(preds)

    # è®¡ç®—æ ¸å¿ƒæŒ‡æ ‡
    val_acc = accuracy_score(all_true, all_pred)
    val_f1_macro = f1_score(all_true, all_pred, average="macro")
    val_f1_micro = f1_score(all_true, all_pred, average="micro")
    avg_loss = total_loss / len(val_loader)

    # æ‰“å�°æ ¸å¿ƒæŒ‡æ ‡
    print(f"\nğŸ“Š éªŒè¯�é›†æ ¸å¿ƒæŒ‡æ ‡ï¼š")
    print(f"  - å¹³å�‡æ�Ÿå¤±ï¼š{avg_loss:.4f}")
    print(f"  - å‡†ç¡®ç�‡ï¼ˆAccuracyï¼‰ï¼š{val_acc:.4f}")
    print(f"  - å®�F1ï¼ˆMacro-F1ï¼‰ï¼š{val_f1_macro:.4f}")
    print(f"  - å¾®F1ï¼ˆMicro-F1ï¼‰ï¼š{val_f1_micro:.4f}")

    # æ‰“å�°åˆ†ç±»æŠ¥å‘Šï¼ˆæ¯�ç±»è¡Œä¸ºçš„è¯¦ç»†æŒ‡æ ‡ï¼‰
    print(f"\nğŸ“‹ éªŒè¯�é›†åˆ†ç±»æŠ¥å‘Šï¼ˆæŒ‰è¡Œä¸ºç±»åˆ«ï¼‰ï¼š")
    report = classification_report(
        all_true, all_pred,
        target_names=[id_to_gesture[i] for i in range(num_classes)],
        output_dict=False,
        zero_division=0
    )
    print(report)

    # ç»˜åˆ¶æ··æ·†çŸ©é˜µï¼ˆå½’ä¸€åŒ–ï¼Œä¾¿äº�è§‚å¯Ÿé”™è¯¯åˆ†å¸ƒï¼‰
    plt.figure(figsize=(16, 14))
    cm = confusion_matrix(all_true, all_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # æŒ‰çœŸå®�æ ‡ç­¾å½’ä¸€åŒ–

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[id_to_gesture[i] for i in range(num_classes)],
        yticklabels=[id_to_gesture[i] for i in range(num_classes)],
        annot_kws={"fontsize": 6, "fontweight": "bold"},
        linewidths=0.3,
        linecolor="white"
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel("é¢„æµ‹è¡Œä¸ºï¼ˆPredicted Gestureï¼‰", fontsize=12, fontweight="bold")
    plt.ylabel("çœŸå®�è¡Œä¸ºï¼ˆTrue Gestureï¼‰", fontsize=12, fontweight="bold")
    plt.title(
        f"éªŒè¯�é›†æ··æ·†çŸ©é˜µï¼ˆå½’ä¸€åŒ–ï¼‰\n"
        f"å‡†ç¡®ç�‡ï¼š{val_acc:.4f} | å®�F1ï¼š{val_f1_macro:.4f}",
        fontsize=14, pad=20, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # --------------------------
    # æ­¥éª¤4ï¼šæµ‹è¯•é›†é¢„æµ‹å�ˆç�†æ€§åˆ†æ��ï¼ˆç±»åˆ«åˆ†å¸ƒå¯¹æ¯”ï¼‰
    # --------------------------
    print("\n" + "="*60)
    print("æ­¥éª¤4ï¼šæµ‹è¯•é›†é¢„æµ‹å�ˆç�†æ€§åˆ†æ��")
    print("="*60)
    # å�ˆå¹¶æµ‹è¯•é›†ä¸�é¢„æµ‹ç»“æ�œï¼ˆä»…ä¿�ç•™åŒ¹é…�çš„åº�åˆ—ï¼‰
    test_with_pred = pd.merge(
        test_sensor[["sequence_id"]].drop_duplicates(),  # ä»…ä¿�ç•™æµ‹è¯•é›†åº�åˆ—ID
        pred_df[["sequence_id", "predicted_gesture"]],
        on="sequence_id",
        how="inner"
    )
    # æ£€æŸ¥åŒ¹é…�ç�‡
    match_rate = len(test_with_pred) / test_sensor["sequence_id"].nunique()
    print(f"âœ… æµ‹è¯•é›†ä¸�é¢„æµ‹ç»“æ�œåŒ¹é…�ï¼š")
    print(f"  - åŒ¹é…�åº�åˆ—æ•°ï¼š{len(test_with_pred)}")
    print(f"  - åŒ¹é…�ç�‡ï¼š{match_rate:.2%}")
    if match_rate < 0.9:
        print(f"âš ï¸� è­¦å‘Šï¼šåŒ¹é…�ç�‡ä½�äº�90%ï¼Œå�¯èƒ½å­˜åœ¨åº�åˆ—IDä¸�åŒ¹é…�ï¼�")

    # ç±»åˆ«åˆ†å¸ƒå¯¹æ¯”ï¼ˆè®­ç»ƒé›† vs æµ‹è¯•é›†é¢„æµ‹ï¼‰
    # è®­ç»ƒé›†ç±»åˆ«åˆ†å¸ƒï¼ˆæŒ‰åº�åˆ—æ•°è®¡ç®—ï¼Œä¸�æµ‹è¯•é›†ä¸€è‡´ï¼‰
    train_seq_dist = train_sensor.groupby("sequence_id")["gesture"].first().value_counts(normalize=True) * 100
    # æµ‹è¯•é›†é¢„æµ‹ç±»åˆ«åˆ†å¸ƒ
    test_pred_dist = test_with_pred["predicted_gesture"].value_counts(normalize=True) * 100

    # å�ˆå¹¶åˆ†å¸ƒæ•°æ�®ï¼ˆç¡®ä¿�æ‰€æœ‰ç±»åˆ«éƒ½æ˜¾ç¤ºï¼‰
    all_gestures = sorted(gesture_to_id.keys())
    dist_compare = pd.DataFrame({
        "è®­ç»ƒé›†ï¼ˆåº�åˆ—å� æ¯”ï¼‰": train_seq_dist.reindex(all_gestures, fill_value=0),
        "æµ‹è¯•é›†é¢„æµ‹ï¼ˆåº�åˆ—å� æ¯”ï¼‰": test_pred_dist.reindex(all_gestures, fill_value=0)
    })

    # ç»˜åˆ¶ç±»åˆ«åˆ†å¸ƒå¯¹æ¯”å›¾
    plt.figure(figsize=(16, 8))
    dist_compare.plot(
        kind="bar",
        width=0.8,
        color=["#2E86AB", "#A23B72"],
        alpha=0.8
    )
    plt.title(
        "è®­ç»ƒé›†ä¸�æµ‹è¯•é›†é¢„æµ‹ç±»åˆ«åˆ†å¸ƒå¯¹æ¯”ï¼ˆæŒ‰åº�åˆ—æ•°å� æ¯”ï¼‰",
        fontsize=14, pad=20, fontweight="bold"
    )
    plt.xlabel("è¡Œä¸ºç±»åˆ«ï¼ˆGestureï¼‰", fontsize=12)
    plt.ylabel("å� æ¯”ï¼ˆ%ï¼‰", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(fontsize=11, loc="upper right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    # è®¡ç®—KLæ•£åº¦ï¼ˆè¡¡é‡�åˆ†å¸ƒç›¸ä¼¼åº¦ï¼šå€¼è¶Šå°�è¶Šç›¸ä¼¼ï¼‰
    def kl_div(p, q):
        p = np.array(p) + 1e-10  # é�¿å…�log(0)
        q = np.array(q) + 1e-10
        return np.sum(p * np.log(p / q))

    kl_value = kl_div(
        dist_compare["è®­ç»ƒé›†ï¼ˆåº�åˆ—å� æ¯”ï¼‰"] / 100,
        dist_compare["æµ‹è¯•é›†é¢„æµ‹ï¼ˆåº�åˆ—å� æ¯”ï¼‰"] / 100
    )
    print(f"\nğŸ“Š åˆ†å¸ƒç›¸ä¼¼åº¦åˆ†æ��ï¼š")
    print(f"  - KLæ•£åº¦ï¼ˆKL Divergenceï¼‰ï¼š{kl_value:.4f}")
    print(f"    ï¼ˆæ³¨ï¼šKLå€¼<0.5è¡¨ç¤ºåˆ†å¸ƒé«˜åº¦ç›¸ä¼¼ï¼Œ0.5~1.0è¡¨ç¤ºä¸­åº¦ç›¸ä¼¼ï¼Œ>1.0è¡¨ç¤ºå·®å¼‚è¾ƒå¤§ï¼‰")

    print("\n" + "="*60)
    print("æ‰€æœ‰è¯„ä¼°æ­¥éª¤å®Œæˆ�ï¼�")
    print("="*60)


if __name__ == "__main__":
    # å…¨å±€å�˜é‡�ï¼šç±»åˆ«æ˜ å°„ï¼ˆç”±Datasetåˆ�å§‹åŒ–æ—¶ç”Ÿæˆ�ï¼‰
    gesture_to_id = {}
    id_to_gesture = {}
    main()

