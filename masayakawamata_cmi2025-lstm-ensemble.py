# import polars as pl
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from sklearn.model_selection import StratifiedGroupKFold
# from sklearn.preprocessing import LabelEncoder
# from tqdm.auto import tqdm
# import joblib
# import os
# import random
# import warnings

# warnings.filterwarnings("ignore")

# # --- 1. Configuration Block ---
# class CFG:
#     # --- Final Ensemble Configuration ---
#     N_MODELS = 10 
#     EPOCHS = 50 # Final training epochs
    
#     # --- Winning Architecture: "2-Layer Wider + Higher Dropout" ---
#     LSTM_UNITS = [256, 256]
#     LSTM_LAYERS = len(LSTM_UNITS) # Automatically determined
#     LSTM_DROPOUT = 0.6
    
#     # --- Training Parameters (Using best known settings) ---
#     LEARNING_RATE = 1e-3
#     WEIGHT_DECAY = 1e-5
#     LOSS_ALPHA = 0.7 

#     # --- Epoch-wise Bagging Parameters ---
#     BATCH_SIZE_RANGE = (32, 128)
#     SEQ_LEN_RANGE = (96, 128)

#     # --- Fixed Parameters ---
#     N_SPLITS = 5
#     N_CLASSES = 18
#     IMU_DIM = 7
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     SEED = 42

# def seed_everything(seed):
#     random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True

# seed_everything(CFG.SEED)

# # --- 2. Helper Functions & Classes ---
# class F1Loss(nn.Module):
#     def __init__(self, epsilon=1e-7):
#         super().__init__(); self.epsilon = epsilon
#     def forward(self, y_pred, y_true):
#         y_pred_softmax = torch.softmax(y_pred, dim=1)
#         y_true_one_hot = F.one_hot(y_true, num_classes=CFG.N_CLASSES).float()
#         tp = (y_true_one_hot * y_pred_softmax).sum(dim=0)
#         fp = ((1 - y_true_one_hot) * y_pred_softmax).sum(dim=0)
#         fn = (y_true_one_hot * (1 - y_pred_softmax)).sum(dim=0)
#         precision = tp / (tp + fp + self.epsilon); recall = tp / (tp + fn + self.epsilon)
#         f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
#         return 1 - f1.clamp(min=self.epsilon, max=1-self.epsilon).mean()

# class FullSequenceDataset(Dataset):
#     def __init__(self, samples, indices, is_train=False):
#         self.samples = samples; self.indices = indices; self.is_train = is_train
#     def __len__(self):
#         return len(self.indices)
#     def __getitem__(self, idx):
#         true_idx = self.indices[idx]
#         data, label = self.samples[true_idx]['data'], self.samples[true_idx]['label']
#         if self.is_train:
#             noise = np.random.normal(0, 0.05, data.shape).astype(np.float32); data = data + noise
#         return data, label

# def create_collate_fn(seq_len, input_dim):
#     def collate_fn(batch):
#         sequences, labels = zip(*batch)
#         padded_sequences = []
#         for seq in sequences:
#             current_len = seq.shape[1]
#             if current_len < seq_len:
#                 padding = np.zeros((input_dim, seq_len - current_len), dtype=np.float32)
#                 padded_seq = np.concatenate([seq, padding], axis=1)
#             else:
#                 start_point = np.random.randint(0, current_len - seq_len + 1)
#                 padded_seq = seq[:, start_point : start_point + seq_len]
#             padded_sequences.append(padded_seq)
#         return torch.tensor(np.array(padded_sequences), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
#     return collate_fn

# class LSTM_Model_Flexible(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.lstm_layers = nn.ModuleList()
#         input_dim = cfg.IMU_DIM
#         for i, hidden_dim in enumerate(cfg.LSTM_UNITS):
#             dropout_val = cfg.LSTM_DROPOUT if i < len(cfg.LSTM_UNITS) - 1 else 0
#             self.lstm_layers.append(
#                 nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True, dropout=dropout_val, bidirectional=True)
#             )
#             input_dim = hidden_dim * 2
#         self.classifier = nn.Linear(input_dim, cfg.N_CLASSES)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         for lstm_layer in self.lstm_layers:
#             x, _ = lstm_layer(x)
#         return self.classifier(x[:, -1, :])

# # --- 3. Main Training Execution ---
# print("Loading and preparing data into a unified list...")
# train_df = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
# imu_cols = [col for col in train_df.columns if 'acc_' in col or 'rot_' in col]
# all_samples = []
# le = LabelEncoder().fit(train_df['gesture'].unique().sort().to_numpy())
# for seq_id, group_df in tqdm(train_df.group_by("sequence_id", maintain_order=True), desc="Processing Sequences"):
#     imu_data = group_df[imu_cols].fill_null(0.0).to_numpy().T.astype(np.float32)
#     all_samples.append({"data": imu_data, "label": le.transform([group_df['gesture'][0]])[0], "group": group_df['subject'][0]})

# train_indices = np.arange(len(all_samples))

# print(f"\n--- Starting Final Ensemble Training for {CFG.N_MODELS} models ---")
# for model_idx in range(CFG.N_MODELS):
#     print(f"\n" + "="*20 + f" Training Model {model_idx+1}/{CFG.N_MODELS} " + "="*20)
    
#     model = LSTM_Model_Flexible(CFG).to(CFG.DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
#     scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)
#     criterion_ce, criterion_f1 = nn.CrossEntropyLoss(), F1Loss()

#     for epoch in range(CFG.EPOCHS):
#         epoch_batch_size = random.randint(*CFG.BATCH_SIZE_RANGE)
#         epoch_seq_len = random.randint(*CFG.SEQ_LEN_RANGE)
        
#         train_dataset = FullSequenceDataset(all_samples, train_indices, is_train=True)
#         train_loader = DataLoader(
#             train_dataset, batch_size=epoch_batch_size, shuffle=True, 
#             num_workers=0, collate_fn=create_collate_fn(epoch_seq_len, CFG.IMU_DIM)
#         )
        
#         model.train()
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS} [Model {model_idx+1}]")
#         for data, batch_labels in pbar:
#             data, batch_labels = data.to(CFG.DEVICE), batch_labels.to(CFG.DEVICE)
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = CFG.LOSS_ALPHA * criterion_ce(outputs, batch_labels) + (1 - CFG.LOSS_ALPHA) * criterion_f1(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()
#             pbar.set_postfix(loss=f"{loss.item():.4f}")
        
#         scheduler.step() # Step the scheduler at the end of each epoch
            
#     model_save_path = f'lstm_ensemble_model_{model_idx}.pth'
#     print(f"Finished training model {model_idx+1}. Saving to {model_save_path}")
#     torch.save(model.state_dict(), model_save_path)

# joblib.dump(le, 'lstm_ensemble_label_encoder.pkl')
# print("\n--- All models for the LSTM ensemble have been trained and saved successfully! ---")


import os
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import joblib
import kaggle_evaluation.cmi_inference_server
import warnings

warnings.filterwarnings("ignore")

# --- 1. Configuration & Model Definition ---
class CFG:
    N_MODELS = 10 # The number of models in the ensemble
    
    # --- Data Shape (must match training) ---
    MAX_LEN = 128 
    N_CLASSES = 18
    IMU_DIM = 7
    
    # --- Architecture (must match training) ---
    LSTM_UNITS = [256, 256]
    LSTM_DROPOUT = 0.6
    
    # --- Environment ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The model architecture MUST be defined to load the weights.
class LSTM_Model_Flexible(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        input_dim = cfg.IMU_DIM
        
        for i, hidden_dim in enumerate(cfg.LSTM_UNITS):
            dropout_val = cfg.LSTM_DROPOUT if i < len(cfg.LSTM_UNITS) - 1 else 0
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim, hidden_dim, num_layers=1, batch_first=True,
                    dropout=dropout_val, bidirectional=True
                )
            )
            input_dim = hidden_dim * 2
            
        self.classifier = nn.Linear(input_dim, cfg.N_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
        out = self.classifier(x[:, -1, :])
        return out

# --- 2. Load All Pre-trained Models and Artifacts ---
INPUT_DIR = '/kaggle/input/cmi2025-lstm-ensemble/' 

print(f"Loading {CFG.N_MODELS} LSTM ensemble models and assets...")
models = []
for i in range(CFG.N_MODELS):
    model = LSTM_Model_Flexible(CFG).to(CFG.DEVICE)
    model_path = os.path.join(INPUT_DIR, f'lstm_ensemble_model_{i}.pth')
    model.load_state_dict(torch.load(model_path, map_location=CFG.DEVICE))
    model.eval()
    models.append(model)

le = joblib.load(os.path.join(INPUT_DIR, 'lstm_ensemble_label_encoder.pkl'))

print(f"All {len(models)} models loaded successfully.")


# --- 3. Define Predict Function ---
def prepare_inference_tensor(sequence: pl.DataFrame) -> torch.Tensor:
    """Prepares a single raw sequence into a padded tensor for inference."""
    imu_cols = [col for col in sequence.columns if 'acc_' in col or 'rot_' in col]
    imu_data_lists = [sequence.get_column(col).fill_null(0.0).to_numpy() for col in imu_cols]
    imu_data_np = np.array(imu_data_lists, dtype=np.float32)
    
    current_len = imu_data_np.shape[1]
    if current_len < CFG.MAX_LEN:
        padding = np.zeros((CFG.IMU_DIM, CFG.MAX_LEN - current_len), dtype=np.float32)
        data_padded = np.concatenate([imu_data_np, padding], axis=1)
    else:
        data_padded = imu_data_np[:, -CFG.MAX_LEN:] # Take the last part of the sequence
        
    return torch.tensor(data_padded, dtype=torch.float32).unsqueeze(0).to(CFG.DEVICE)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predicts a gesture by ensembling the predictions of all loaded LSTM models.
    """
    # 1. Prepare the input tensor
    X_test = prepare_inference_tensor(sequence)
    
    # 2. Get predictions from all models
    all_probs = []
    with torch.no_grad():
        for model in models:
            outputs = model(X_test)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs)
            
    # 3. Score-level Fusion: Average the probabilities across all models
    avg_probs = torch.mean(torch.stack(all_probs), dim=0).cpu().numpy()
    
    # 4. Get the final prediction
    final_pred_idx = np.argmax(avg_probs, axis=1)
    
    # 5. Convert index back to string label
    pred_label = le.inverse_transform(final_pred_idx)[0]
    
    return pred_label

# --- 4. Start the Inference Server ---
print("Starting inference server...")
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

