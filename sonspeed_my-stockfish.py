import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import wandb


file_path = "/kaggle/input/train-your-own-stockfish-nnue/train.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.replace("\t", "").str.strip()


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
my_secret = user_secrets.get_secret("wandb_api_key1") 
wandb.login(key=my_secret)


wandb.init(project="chess-nnue")


def encoding(fen):
    piece_dict = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                  'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
    board_array = np.zeros((8, 8), dtype=int)
    rows = fen.split()[0].split('/')
    
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            else:
                board_array[row_idx, col_idx] = piece_dict[char]
                col_idx += 1

    halfkp = board_array.flatten()
    halfka = (board_array != 0).astype(int).flatten()
    return np.concatenate([halfkp, halfka])  # 128 features


class ChessDataset(Dataset):
    def __init__(self, fen_list, evals):
        self.features = np.array([encoding(fen) for fen in fen_list], dtype=np.float32)
        self.labels = np.array(evals, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])



dataset = ChessDataset(df["FEN"], df["Evaluation"])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class NNUEModel(nn.Module):
    def __init__(self):
        super(NNUEModel, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NNUEModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
num_epochs = 50


best_val_loss = float('inf')
stopping_tolerance = 3  # Stop if val_loss doesn't improve for 3 epochs
no_improve_epochs = 0


for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)  

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += loss_fn(outputs, targets).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= stopping_tolerance:
            print("Validation loss stopped improving, stopping early.")
            break

torch.save(model.state_dict(), "nnue_model.pth")
print("Mô hình đã được lưu!")


# def train_model(config=None):
#     with wandb.init(config=config):
#         config = wandb.config
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = NNUEModel().to(device)
#         optimizer = {
#             "adamw": optim.AdamW(model.parameters(), lr=config.lr),
#             "adam": optim.Adam(model.parameters(), lr=config.lr),
#             "rmsprop": optim.RMSprop(model.parameters(), lr=config.lr)
#         }[config.optimizer]
#         loss_fn = nn.MSELoss()
        
#         for epoch in range(config.epochs):
#             model.train()
#             total_loss = 0
#             for batch in train_loader:
#                 inputs, targets = batch
#                 inputs, targets = inputs.to(device), targets.to(device)  

#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = loss_fn(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
            
#             avg_train_loss = total_loss / len(train_loader)
#             model.eval()
#             val_loss = 0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     inputs, targets = batch
#                     inputs, targets = inputs.to(device), targets.to(device)
#                     outputs = model(inputs)
#                     val_loss += loss_fn(outputs, targets).item()
            
#             avg_val_loss = val_loss / len(val_loader)
#             wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})


# sweep_config = {
#     "method": "bayes",
#     "metric": {"name": "val_loss", "goal": "minimize"},
#     "parameters": {
#         "lr": {"values": [0.001, 0.0005, 0.0001]},
#         "optimizer": {"values": ["adamw", "adam", "rmsprop"]},
#         "epochs": {"values": [30, 50]}
#     }
# }


# sweep_id = wandb.sweep(sweep_config, project="chess-nnue")
# wandb.agent(sweep_id, function=train_model, count=3)



torch.save(model.state_dict(), "nnue_model.pth")
print("Mô hình đã được lưu!")


def load_model(model_path):
    model = NNUEModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_fen(fen, model):
    model.eval()
    feature = torch.tensor(encoding(fen), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(feature).item()
    return pred


sample_file_path = "/kaggle/input/train-your-own-stockfish-nnue/sample_submission.csv"
df_sample = pd.read_csv(sample_file_path)


if "FEN\tPredicted_Evaluation" in df_sample.columns:
    df_sample[["FEN", "Predicted_Evaluation"]] = df_sample["FEN\tPredicted_Evaluation"].str.split("\t", expand=True)
    df_sample.drop(columns=["FEN\tPredicted_Evaluation"], inplace=True)


model = load_model("nnue_model.pth")
df_sample["Predicted_Evaluation"] = df_sample["FEN"].apply(lambda fen: evaluate_fen(fen, model))
df_sample.to_csv("submission.csv", index=False)
print("File kết quả đã được lưu: submission.csv")


