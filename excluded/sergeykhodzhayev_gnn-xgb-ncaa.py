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


!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --no-cache-dir
!pip install torchdata==0.6.0 --no-cache-dir
!pip install dgl==2.1.0 --no-cache-dir



import os
os.environ["DGLBACKEND"] = "pytorch"


import torch
import torchdata
import dgl

print(" Torch version:", torch.__version__)
print(" TorchData version:", torchdata.__version__)
print("DGL version:", dgl.__version__)



from torchdata.datapipes.iter import IterDataPipe
print(" TorchData работает!")



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from dgl.nn import GATConv
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# === 1. Загрузка данных ===
m_regular = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv")
m_tourney = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
w_regular = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv")
w_tourney = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv")
seeds = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")
massey_men = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MMasseyOrdinals.csv")
massey_women = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")

match_data = pd.concat([m_regular, m_tourney, w_regular, w_tourney], ignore_index=True)


def extract_seed(seed):
    return int("".join(filter(str.isdigit, str(seed))))
seeds["SeedNum"] = seeds["Seed"].apply(extract_seed)
seeds_dict = seeds.set_index(["Season", "TeamID"])["SeedNum"].to_dict()

# Создание графа 
teams_set = set(match_data["WTeamID"]).union(set(match_data["LTeamID"]))
team_to_idx = {team: idx for idx, team in enumerate(sorted(teams_set))}

edges = [(team_to_idx[row["WTeamID"]], team_to_idx[row["LTeamID"]]) for _, row in match_data.iterrows()]
team_graph = dgl.graph((torch.tensor([e[0] for e in edges]), torch.tensor([e[1] for e in edges])))
team_graph = dgl.add_self_loop(team_graph)

# Добавление узлов 
num_teams = team_graph.num_nodes()
scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=3, weights='distance')

latest_season = massey_men["Season"].max()
latest_rankings = massey_men[massey_men["Season"] == latest_season]
team_rankings = latest_rankings.groupby("TeamID")["OrdinalRank"].mean().to_dict()

features_list = []
for team, idx in team_to_idx.items():
    features = []
    win_counts = match_data[match_data['WTeamID'] == team].shape[0]
    loss_counts = match_data[match_data['LTeamID'] == team].shape[0]
    avg_point_diff = match_data.loc[match_data['WTeamID'] == team, 'WScore'].mean() - match_data.loc[match_data['LTeamID'] == team, 'LScore'].mean()
    
    features.append(win_counts)
    features.append(loss_counts)
    features.append(avg_point_diff)
    features.append(team_rankings.get(team, np.nan))
    features.append(seeds_dict.get((latest_season, team), np.nan))
    
    while len(features) < 20:
        features.append(0)
    
    features_list.append(features)

# Обучение GNN 
class GATModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads=num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, out_feats, num_heads=1)
        self.batch_norm1 = nn.BatchNorm1d(hidden_feats * num_heads)
        self.batch_norm2 = nn.BatchNorm1d(out_feats)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, x):
        x = self.conv1(g, x).flatten(1)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(g, x).squeeze()
        x = self.batch_norm2(x)
        return x

gnn_model = GATModel(in_feats=20, hidden_feats=64, out_feats=20, num_heads=4, dropout=0.3)
optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

features_array = np.array(features_list, dtype=np.float32)
features_array = imputer.fit_transform(features_array)
node_features = torch.tensor(scaler.fit_transform(features_array), dtype=torch.float)

for epoch in range(50):
    optimizer.zero_grad()
    embeddings = gnn_model(team_graph, node_features)
    loss = loss_fn(embeddings, node_features)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

#  эмбеддинги
team_embeddings = embeddings.detach().numpy()
team_embeddings_dict = {team: team_embeddings[idx] for team, idx in team_to_idx.items()}

#  XGBoost 
X_data = []
y_data = []
for _, row in match_data.iterrows():
    if row["WTeamID"] in team_to_idx and row["LTeamID"] in team_to_idx:
        w_embed = team_embeddings[team_to_idx[row["WTeamID"]]]
        l_embed = team_embeddings[team_to_idx[row["LTeamID"]]]
        X_data.append(w_embed - l_embed)
        y_data.append(1)
        X_data.append(l_embed - w_embed)
        y_data.append(0)

X_train, X_test, y_train, y_test = train_test_split(np.array(X_data), np.array(y_data), test_size=0.2, random_state=42)
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict_proba(X_test)[:, 1]
print(f'Brier Score: {brier_score_loss(y_test, y_pred)}')






test_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/SampleSubmissionStage2.csv")
test_df[["Season", "WTeamID", "LTeamID"]] = test_df["ID"].str.split("_", expand=True).astype(int)

test_features = [(team_embeddings[team_to_idx[row["WTeamID"]]] - team_embeddings[team_to_idx[row["LTeamID"]]]) for _, row in test_df.iterrows()]
test_df["Pred"] = xgb_model.predict_proba(np.array(test_features))[:, 1]
test_df[["ID", "Pred"]].to_csv("submission.csv", index=False)
print(" Финальный submission.csv сохранен!")


import shutil

# Пути к файлам
submission_path = "/kaggle/working/submission.csv"
zip_path = "/kaggle/working/submission.zip"

# Создание ZIP-архива
shutil.make_archive(zip_path.replace(".zip", ""), 'zip', "/kaggle/working/", "submission.csv")

print(f"✅ ZIP-архив создан: {zip_path}")


