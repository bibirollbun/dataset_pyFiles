import os
import pathlib
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import kaggle_evaluation.cmi_inference_server
import joblib


BASE_DIR = pathlib.Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
MODEL_DIR = pathlib.Path("/kaggle/input/csi-lstm-classifier-starter-dataset")
DEVICE = torch.device("cuda") 

# data
ID_COL = "sequence_id"
SEQ_COL = "sequence_counter"
FEATURES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
]
CLASSES = [
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Drink from bottle/cup",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Feel around in tray and pull out an object",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Glasses on/off",
    "Neck - pinch skin",
    "Neck - scratch",
    "Pinch knee/leg skin",
    "Pull air toward your face",
    "Scratch knee/leg skin",
    "Text on phone",
    "Wave hello",
    "Write name in air",
    "Write name on leg",
]

# model
MAX_LENGTH = 40
HIDDEN_SIZE = 256
N_LAYERS = 3
N_CLF_LAYERS = 2
BIDIRECTIONAL = True
DROP = 0.15
NUM_CLASSES = len(CLASSES)


os.listdir(MODEL_DIR)


transformer = joblib.load(MODEL_DIR / "preprocessing.joblib")
transformer


class LSTMClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(DROP)
        self.lstm = nn.LSTM(
            input_size=len(FEATURES),
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROP if N_LAYERS > 1 else 0.0,
            batch_first=False,
        )
        output_size = 2 * HIDDEN_SIZE if BIDIRECTIONAL else HIDDEN_SIZE
        clf_layers = []
        for i in range(N_CLF_LAYERS - 1):
            clf_layers.append(nn.Linear(output_size, output_size))
            clf_layers.append(nn.ReLU())
        clf_layers.append(nn.Linear(output_size, NUM_CLASSES))
        self.clf = nn.Sequential(*clf_layers)

    def forward(self, x, mask):
        x, _ = self.lstm(x)
        x = torch.mean(x * mask.unsqueeze(-1), dim=0)
        x = self.drop(x)
        return self.clf(x)


model = LSTMClassifier()
model.eval()
model.to(DEVICE)


model.load_state_dict(torch.load(MODEL_DIR / "model_best_score.pt", weights_only=True))


train = pl.read_csv( BASE_DIR / "train.csv", n_rows=1000)
sequence = train.filter( pl.col(ID_COL) == "SEQ_000092")
sequence


df = sequence.to_pandas()
df = df.sort_values(by=SEQ_COL)


df = transformer.transform(df[FEATURES])
df = df[-MAX_LENGTH:]
x = torch.tensor(df[FEATURES].values.astype(np.float32))
x = x.unsqueeze(dim=1)
x.shape


mask = torch.ones_like(x[..., -1])
mask.shape


with torch.no_grad():
    x = x.to(DEVICE)
    mask = mask.to(DEVICE)
    logits = model(x, mask).cpu()
logits


CLASSES[int(torch.argmax(logits).item())]


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df = sequence.to_pandas()
    df = df.sort_values(SEQ_COL)
    df = transformer.transform(df)
    df = df[-MAX_LENGTH:]
    x = torch.tensor(df[FEATURES].values.astype(np.float32))
    x = x.unsqueeze(dim=1)
    mask = torch.ones_like(x[..., -1])
    model.eval()
    with torch.no_grad():
        x = x.to(DEVICE)
        mask = mask.to(DEVICE)
        logits = model(x, mask).cpu()
    pred = CLASSES[int(torch.argmax(logits).item())]
    print(pred)
    return pred


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(BASE_DIR / "test.csv", BASE_DIR / "test_demographics.csv")
    )




