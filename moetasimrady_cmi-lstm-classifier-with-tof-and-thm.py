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
from pathlib import Path


from expanded_cmi_utility_scripts import *


BASE_DIR = pathlib.Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
MODEL_DIR = pathlib.Path("/kaggle/input/lstm-repo")
DEVICE = torch.device("cuda") 

# data
ID_COL = "sequence_id"
SEQ_COL = "sequence_counter"

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
MAX_LENGTH = 55
HIDDEN_SIZE = 256
N_LAYERS = 1

BIDIRECTIONAL = True
DROP = 0.2
NUM_CLASSES = len(CLASSES)


os.listdir(MODEL_DIR)


transformer = joblib.load(MODEL_DIR / "second_lstm_preprocessing.joblib")
transformer


CLASS_NOS=18
hidden_size=256
import torch.nn.init as init
from functools import partial
import torch
import torch.nn as nn
from torch.nn import LayerNorm


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv

    def forward(self, x): 
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x -= self.sub
        if self.maxv is not None: x.clamp_max_(self.maxv)
        return x
class LayerNorm_M(nn.Module):
    def __init__(self, hidden_size=256, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1,1, hidden_size))  # scale
        self.beta = nn.Parameter(torch.zeros(1,1, hidden_size))  # shift

    def forward(self, x):
        # x shape: [num_layers, batch_size, hidden_size]
        mean = x.mean(dim=-1, keepdim=True)  # mean over hidden_size
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



class LSTMClassifier(nn.Module):
    def __init__(self, input_size=28, hidden_size=256, num_layers=1, bidirectional=False, num_classes=18):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create LSTM layers and layer norms manually
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer in range(num_layers):
            lstm_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            lstm_layer = nn.LSTM(input_size=lstm_input_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        bidirectional=bidirectional,
                        dropout=0.2,
                        batch_first=False)
            self.lstm_layers.append(
               lstm_layer
            )
            for name, param in lstm_layer.named_parameters():
                if 'weight' in name:  # Apply only to weights, not biases
                    init.xavier_normal_(param)
            self.layer_norms.append(LayerNorm(hidden_size * self.num_directions))

        self.dropout = nn.Dropout(p=0.3)
        # self.leaky_act = GeneralRelu(leak=0.4, sub=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            GeneralRelu(leak=0.4, sub=0.1),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out = x  # (seq_len, batch, input_size)
        h_n = None
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            out, (h_n, _) = lstm(out)   # out: (seq_len, batch, hidden*2), h_n: (num_directions, batch, hidden)
            out = norm(out)

        # Get final hidden state from last layer
        # h_n: (num_directions, batch, hidden_size)
        h_final = h_n.transpose(0, 1).contiguous().view(x.size(1), -1)  # (batch, hidden_size * num_directions)

        h_final = self.dropout(h_final)
        return self.classifier(h_final)

def initialize_weights(module):
    if isinstance(module, nn.Linear):  # Apply to Linear layers
        init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)


model = LSTMClassifier()
initialize_weights(model)
model.eval()
model.to(DEVICE)


os.listdir(MODEL_DIR)


encoder_with_pooling



# defining and loading the autoencoder
encoder_with_pooling.load_state_dict(torch.load("/kaggle/input/tof-autoencoder/autoencoder_second.pth",weights_only=True))


model.load_state_dict(torch.load(MODEL_DIR / "lstm_third_attempt.pth", weights_only=True))


train = pl.read_csv( BASE_DIR / "train.csv", n_rows=1000)
sequence = train.filter( pl.col(ID_COL) == "SEQ_000092")
sequence


df = sequence.to_pandas()
df = df.sort_values(by=SEQ_COL)


headers = sequence.columns
tof_cols = [header for header in headers if header.startswith("tof")]
processed_tof_vals = preprocess_tof_data(df,tof_cols)

flattened_vals = transform_tof_data(encoder_with_pooling,processed_tof_vals)
tof_features = ["tof_feature_{}".format(idx) for idx in range(flattened_vals.shape[-1])]


df = transformer.transform(df[FEATURES[:-16]])


df[tof_features] = flattened_vals.numpy()



df = df[-MAX_LENGTH:]
x = torch.tensor(df[FEATURES].values.astype(np.float32))
x = x.unsqueeze(dim=1)
x.shape


with torch.no_grad():
    x = x.to(DEVICE)

    logits = model(x).cpu()
logits


CLASSES[int(torch.argmax(logits).item())]


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    preprocess_tof_tens = lambda tens: torch.Tensor((tens-tof_data_mean)/tof_data_std).unsqueeze(1).float()
    df = sequence.to_pandas()
    df = df.sort_values(SEQ_COL)
    headers = sequence.columns
    tof_cols = [header for header in headers if header.startswith("tof")]
    processed_tof_vals = preprocess_tof_data(df,tof_cols)
    flattened_vals = transform_tof_data(encoder_with_pooling,processed_tof_vals)
    tof_features = ["tof_feature_{}".format(idx) for idx in range(flattened_vals.shape[-1])]
    df = transformer.transform(df)
    df[tof_features] = flattened_vals.numpy()
    df = df[-MAX_LENGTH:]
    x = torch.tensor(df[FEATURES].values.astype(np.float32))
    x = x.unsqueeze(dim=1)
    mask = torch.ones_like(x[..., -1])
    model.eval()
    with torch.no_grad():
        x = x.to(DEVICE)
        logits = model(x).cpu()
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




