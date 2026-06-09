import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


headers = ['oid_ypos'] + [f"x_{xpos}" for xpos in range(1, 70, 2)]
headers = ','.join(headers)
with open("submission.csv", 'w+') as file:
    file.write(headers)


%%time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


batch_size = 100

TEST_DIR = "/kaggle/input/waveform-inversion/test"
test_filenames = [filename for filename in os.listdir(TEST_DIR)]
print("number of test files:", len(test_filenames))





class FWIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_mlp = nn.Sequential(
            nn.Linear(1000, 512),
            nn.Dropout(0.1),
            nn.LeakyReLU()
        )
        self.positional_embedding = nn.Embedding(70*5, 512)
        self.transformer_layer1 = nn.TransformerEncoderLayer(
            512,
            8,
            dim_feedforward=512,
            batch_first=True
        )
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1])
        xpos = torch.arange(5*70, dtype=torch.long, device=device).view(1, -1)
        xpos = self.positional_embedding(xpos)
        
        x = self.pre_mlp(x) + xpos
        x = self.transformer_layer1(x)
        
        x = x.mean(dim=1)
        x = self.tanh(x)
        return x

class FWIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FWIEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Softplus(),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Softplus(),
            nn.Dropout(0.1),

            nn.Linear(1024, 70*70)
        )
    def forward(self, x):
        xencoder = self.encoder(x)
        xdecoder = self.decoder(xencoder)
        xdecoder = xdecoder.view(-1, 70, 70)
        return xdecoder


models = [
    torch.load("/kaggle/input/fwi-baseline-models/model0.pth", map_location=device),
    #torch.load("/kaggle/input/fwi-baseline-models/model1.pth", map_location=device),
    #torch.load("/kaggle/input/fwi-baseline-models/model2.pth", map_location=device),
    #torch.load("/kaggle/input/fwi-baseline-models/model3.pth", map_location=device),
    #torch.load("/kaggle/input/fwi-baseline-models/model4.pth", map_location=device)
]


def load_testdata(subfilenames):
    test_data = [np.load(os.path.join(TEST_DIR, filename))[np.newaxis, :] for filename in subfilenames]
    test_data = np.concatenate(test_data)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_data = test_data/10
    test_data = test_data.to(device)
    test_data = test_data.permute(0,1, 3, 2)
    test_data = test_data.contiguous()
    return test_data


def infer_model(data):
    batch_size=len(data)
    pred_velocity = np.zeros((batch_size, 70, 70), dtype=np.float32)

    for model in models:
        model.eval()
        with torch.no_grad():
            pred_velocity += model(data).detach().cpu().numpy()
    pred_velocity = pred_velocity/len(models)
    pred_velocity = (pred_velocity*1000)+3000
    return pred_velocity


def append_results_to_file(filenames, pred_velocity_datalist):
    for k,filename in enumerate(filenames):
        oid = filename.replace(".npy", "")
        yhat = pred_velocity_datalist[k]
        
        for y_pos in range(70):
            oidpos = oid+"_y_"+str(y_pos)
            cur_data=[]
            cur_data.append(oidpos)
            for x_pos in range(1, 70, 2):
                pred_value = yhat[y_pos][x_pos]
                pred_value = str(pred_value)
                cur_data.append( pred_value )
            cur_data = ','.join(cur_data)
            with open("submission.csv", 'a') as file:
                file.write("\n")
                file.writelines(cur_data)


%%time
for k in range(0, len(test_filenames), batch_size):
    if (k+batch_size)%500 == 0:
        print(k , k+batch_size)
    subfilenames = test_filenames[k: k+batch_size]
    test_data = load_testdata(subfilenames)
    pred_velocity_datalist = infer_model(test_data)
    append_results_to_file(subfilenames, pred_velocity_datalist)










