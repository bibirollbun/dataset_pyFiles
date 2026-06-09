import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
import joblib
import kaggle_evaluation.cmi_inference_server

# Only keep IMU + thermopile features
feature_cols = [  # Same features as training: IMU + thermopile
    'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'thm_0', 'thm_1', 'thm_2', 'thm_3', 'thm_4'
]

# Set the same max_len as in training (hardcoded from training stats)
max_len = 700  

# Load model components
class CNN_GRU_Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.gru = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.cnn(x)         # (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        _, h = self.gru(x)
        h = torch.cat((h[0], h[1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# Initialize model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)  # Should be 'cuda' if GPU is available

le = joblib.load('/kaggle/input/cnn-gru-model/label_encoder.pkl')
model = CNN_GRU_Model(input_dim = len(feature_cols), num_classes = len(le.classes_))
model.load_state_dict(torch.load('/kaggle/input/cnn-gru-model/best_model.pt', map_location = DEVICE))
model.to(DEVICE)
model.eval()

print(next(model.parameters()).device)  # Should match DEVICE


def preprocess_sequence(df: pd.DataFrame) -> np.ndarray:
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    values = df[feature_cols].fillna(0).values.astype(np.float32)
    padded = np.zeros((max_len, len(feature_cols)), dtype=np.float32)
    length = min(len(values), max_len)
    padded[:length] = values[:length]
    return padded

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        df = sequence.to_pandas()
        x = preprocess_sequence(df)
        print(x.shape)  # Should be (700, 12) for your 12 features
        x_tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)  # (1, T, F)
        with torch.no_grad():
            logits = model(x_tensor)
            pred = logits.argmax(dim=1).item()
            assert pred < len(le.classes_), f"Invalid prediction: {pred} (max={len(le.classes_)})"
        return le.inverse_transform([pred])[0]    
        #return le.classes_[pred]
    except Exception as e:
        print(f"PREDICT ERROR: {e}")  # Log the real error
        raise  # Re-raise to see it in Kaggle's logs    


# Manually test with a sample sequence
sample_seq = pl.DataFrame({col: np.random.rand(100) for col in feature_cols})
sample_demo = pl.DataFrame({"age": [30], "gender": ["M"]})
print(predict(sample_seq, sample_demo))  # Should return a class label


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

