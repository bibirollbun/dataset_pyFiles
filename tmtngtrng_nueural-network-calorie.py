import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import clear_output
from torch.optim.lr_scheduler import OneCycleLR
import itertools
import warnings
warnings.simplefilter('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df = df.dropna().drop_duplicates().drop(columns='id')


df.head()


df.info()


df.describe()


features = ['Age', 'Height', 'Weight', 'Duration','Heart_Rate','Body_Temp']

plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    x = df.sample(1000)
    sb.scatterplot(x=col, y='Calories', data=x)
plt.tight_layout()
plt.show()


df['Sex'] = df['Sex'].map({'male': 1, 'female':0})
df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
df['Weight_Duration'] = df['Weight'] * df['Duration']
col = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

df


df_inp = df.drop(columns='Calories')
df_out = df['Calories']
df_inp


def get_max_min(series: pd.Series) -> list:
    return [series.max(), series.min()]

def max_min_scalse(series: pd.Series) -> pd.Series:
    max_min = get_max_min(series)
    return series.apply(lambda x: (x - max_min[1]) / (max_min[0] - max_min[1]))

for col in df_inp.columns:
    if col != 'Sex':
        df_inp[col] = max_min_scalse(df_inp[col])


df_inp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def rmsle_loss(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=0)
    y_true = torch.clamp(y_true, min=0)
    return torch.sqrt(F.mse_loss(torch.log1p(y_pred), torch.log1p(y_true)))


class CalorieNet(nn.Module):
    def __init__(self, W_init=None, dropout=0.05):
        super(CalorieNet, self).__init__()
        
        self.fc0 = nn.Linear(9, 1024)
        self.bn0 = nn.BatchNorm1d(1024)
        self.dropout0 = nn.Dropout(dropout)

        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout)

        self.fc5 = nn.Linear(64, 1)

        self.final = nn.Softplus()
        self.act = nn.GELU()  

        if W_init is not None:
            with torch.no_grad():
                self.fc1.weight.copy_(W_init)

    def forward(self, x):
        x = self.dropout0(self.act(self.bn0(self.fc0(x))))
        x = self.dropout1(self.act(self.bn1(self.fc1(x))))
        x = self.dropout2(self.act(self.bn2(self.fc2(x))))
        x = self.dropout3(self.act(self.bn3(self.fc3(x))))
        x = self.dropout4(self.act(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = self.final(x) 
        return x


batch_size = 512

X_train, X_test, y_train, y_test = train_test_split(
    df_inp, df_out, test_size=0.8, random_state=42
)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = CalorieNet(dropout=0.15).to(device)
no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm']
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if any(nd in name for nd in no_decay):
        no_decay_params.append(param)
    else:
        decay_params.append(param)
        
optimizer_params = [
    {'params': decay_params, 'weight_decay': 5e-4}, 
    {'params': no_decay_params, 'weight_decay': 0.0}
]
loss_fn_rmsle = rmsle_loss
best_val_loss = float("inf")
patience = 30
counter = 0
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
max_epochs = 500

optimizer = torch.optim.AdamW(
    optimizer_params, 
    betas=(0.9, 0.999), 
    eps=1e-9
)

total_steps = max_epochs * len(train_loader)
scheduler = OneCycleLR(
    optimizer,
    max_lr = 5e-3,        
    total_steps = total_steps,
    pct_start = 0.1,    
    anneal_strategy = 'cos',
    cycle_momentum = False,
    div_factor = 25,      
    final_div_factor = 1e4  
)

for epoch in range(max_epochs):
    model.train()
    total_train_loss = 0
    total_train_rmsle = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")

    for batch_X, batch_y in progress_bar:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if scaler:
            with torch.amp.autocast('cuda'):
                output = model(batch_X)
                loss = rmsle_loss(output, batch_y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch_X)
            loss = rmsle_loss(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()  

        
        with torch.no_grad():
            rmsle = loss_fn_rmsle(output, batch_y)
            total_train_rmsle += rmsle.item()

        total_train_loss += loss.item()

        progress_bar.set_postfix(
            loss=loss.item(),
            rmsle=rmsle.item(),
            lr=scheduler.get_last_lr()[0]
        )

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_rmsle = total_train_rmsle / len(train_loader)
    

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            output = model(batch_X)
            val_loss = loss_fn_rmsle(output, batch_y)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{max_epochs}")
    print(f"Train Loss: {avg_train_loss:.6f}, Train RMSLE: {avg_train_rmsle:.6f}")
    print(f"Val RMSLE: {avg_val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.8f}")

    if avg_val_loss < best_val_loss:
        improvement = (best_val_loss - avg_val_loss) / best_val_loss * 100
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "calorie_model_best.pt")
        clear_output(wait=True)
        print(f"Saved best model (RMSLE: {best_val_loss:.6f}, improved by {improvement:.2f}%)")
        
    else:
        counter += 1
        print(f"No improvement for {counter}/{patience} epochs")
        if counter >= patience:
            
            print(f"Early stopping triggered after {patience} epochs without improvement")
            print(f"Best RMSLE: {best_val_loss:.6f}")
            break


model = CalorieNet(dropout=0.00).to(device)
model.load_state_dict(torch.load("calorie_model_best.pt"))
model.eval()


df_pred = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv').drop(columns='id')

df_pred['Sex'] = df_pred['Sex'].map({'male': 1, 'female': 0})
df_pred['BMI'] = df_pred['Weight'] / ((df_pred['Height'] / 100) ** 2)
df_pred['Weight_Duration'] = df_pred['Weight'] * df_pred['Duration']
def scale_pred(name_col, series: pd.Series) -> pd.Series:
    max_min = get_max_min(df[name_col])
    return series.apply(lambda x: (x - max_min[1]) / (max_min[0] - max_min[1]))

for i in df_pred.columns:
    if i != 'Sex':
        df_pred[i] = scale_pred(i, df_pred[i])




X_pred_tensor = torch.tensor(df_pred.values, dtype=torch.float32).to(device)


with torch.no_grad():
    y_pred = model(X_pred_tensor)
    predicted_calories = y_pred.cpu().numpy().flatten()
df_pred["Predicted_Calories"] = predicted_calories
df_pred.head()



id = range(750000,1000000)
df_submit = pd.DataFrame({
    'id': id,
    'Calories': df_pred['Predicted_Calories']
})
df_submit


df_submit.to_csv('submission.csv', index=False)

