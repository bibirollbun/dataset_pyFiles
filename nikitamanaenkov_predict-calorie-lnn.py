!pip install torchviz


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.image as mpimg
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def feature_engineering(df):
    df = df.copy()
    df['Height_m'] = df['Height'] / 100
    df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
    df['Heart_Rate_ratio'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Cardio_Load'] = df['Duration'] * df['Heart_Rate']
    df['Temp_Heart_ratio'] = df['Body_Temp'] / df['Heart_Rate']
    df['Weight_per_cm'] = df['Weight'] / df['Height']
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=False)
    df['Heart_Rate_Duration_interaction'] = df['Heart_Rate'] * df['Duration']
    df['Body_Temp_Duration_interaction'] = df['Body_Temp'] * df['Duration']
    df['Log_Weight'] = np.log(df['Weight'])
    df['Log_Height'] = np.log(df['Height'])
    df['Log_Cardio_Load'] = np.log(df['Cardio_Load'])
    df['Log_BMI'] = np.log(df['BMI'])
    df['Heart_Rate_ratio_exp'] = np.exp(df['Heart_Rate_ratio'])
    df['Duration_sq'] = df['Duration'] ** 2
    df['Duration_cube'] = df['Duration'] ** 3
    df.drop(columns=['Height_m'], inplace=True)
    return df


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

sex_map = {'male': 1, 'female': 0}
train_df['Sex'] = train_df['Sex'].map(sex_map)
test_df['Sex'] = test_df['Sex'].map(sex_map)

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

y = train_df['Calories'].values
X = train_df.drop(columns=['Calories', 'id']).values
X_test = test_df.drop(columns=['id']).values

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class CaloriesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

train_dataset = CaloriesDataset(X_train, y_train)
val_dataset = CaloriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

X_combined = np.concatenate((X_train, X_val), axis=0)
y_combined = np.concatenate((y_train, y_val), axis=0)

combined_dataset = CaloriesDataset(X_combined, y_combined)


class LiquidNeuronImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.activation = nn.SiLU()  

    def forward(self, x, h):
        dx = self.W(x) + self.U(h)
        dh = (-h + self.activation(dx)) / self.tau
        return h + dh


class LiquidNetworkImproved(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.liquid_layers = nn.ModuleList(
            [LiquidNeuronImproved(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(0.2)

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  
        )

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
    
        for i in range(self.num_layers):
            residual = x  
            h = self.liquid_layers[i](x, h)
            x = h 
    
            if residual.shape == h.shape:
                h = h + residual  
    
            h = self.layer_norms[i](h)
            h = self.dropout(h)
    
        return self.readout(h).squeeze(1)



def rmsle_loss(preds, targets):
    return torch.sqrt(torch.mean((torch.log1p(preds) - torch.log1p(targets)) ** 2))


def train_model_kfold(
    model,
    dataset,
    k=5,
    epochs=100,
    lr=1e-3,
    patience=10,
    min_delta=1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    all_train_losses = []
    all_val_losses = []
    all_lrs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold+1}/{k}")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)

        best_val_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0

        train_losses = []
        val_losses = []
        lrs = []

        for epoch in range(epochs):
            model.train()
            batch_losses = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = torch.sqrt(torch.mean((torch.log1p(preds) - torch.log1p(y_batch)) ** 2))
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            train_loss = np.mean(batch_losses)

            model.eval()
            val_batch_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

                    val_preds = model(X_val)
                    val_loss = torch.sqrt(torch.mean((torch.log1p(val_preds) - torch.log1p(y_val)) ** 2))
                    val_batch_losses.append(val_loss.item())

            val_loss_mean = np.mean(val_batch_losses)

            scheduler.step(val_loss_mean)
            lrs.append(optimizer.param_groups[0]['lr'])

            train_losses.append(train_loss)
            val_losses.append(val_loss_mean)

            print(f"Epoch {epoch+1}: Train RMSLE={train_loss:.4f}, Val RMSLE={val_loss_mean:.4f}, LR={lrs[-1]:.5f}")

            if val_loss_mean + min_delta < best_val_loss:
                best_val_loss = val_loss_mean
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_lrs.append(lrs)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(train_losses, label=f'Train RMSLE Fold {fold+1}')
        axs[0].plot(val_losses, label=f'Val RMSLE Fold {fold+1}')
        axs[0].set_title('RMSLE over Epochs')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('RMSLE')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(lrs, label=f'LR Fold {fold+1}')
        axs[1].set_title('Learning Rate over Epochs')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Learning Rate')
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    min_len = min(len(losses) for losses in all_train_losses)
    trimmed_train = np.array([losses[:min_len] for losses in all_train_losses])
    trimmed_val = np.array([losses[:min_len] for losses in all_val_losses])
    trimmed_lrs = np.array([lr[:min_len] for lr in all_lrs])
    
    avg_train_losses = np.mean(trimmed_train, axis=0)
    avg_val_losses = np.mean(trimmed_val, axis=0)
    avg_lrs = np.mean(trimmed_lrs, axis=0)

    print(f"Average Train RMSLE: {np.mean(avg_train_losses):.4f}")
    print(f"Average Val RMSLE: {np.mean(avg_val_losses):.4f}")

    return model


def visualize_model_architecture(model, sample_input):
    model.eval()  
    out = model(sample_input)  
    make_dot(out, params=dict(model.named_parameters())).render("model_architecture", format="png", cleanup=True)

model = LiquidNetworkImproved(input_dim=X_combined.shape[1], hidden_dim=128, output_dim=1, num_layers=3)

sample_input = torch.randn(1, X_combined.shape[1])
visualize_model_architecture(model, sample_input)

img = mpimg.imread('/kaggle/working/model_architecture.png')
plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.axis('off') 
plt.show()


model = train_model_kfold(
    model=model,
    dataset=combined_dataset,  
    k=5,  
    epochs=200,
    lr=1e-3,
    patience=15,
    min_delta=1e-5
)


X_test = torch.tensor(X_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    preds = model(X_test.to(model.device if hasattr(model, "device") else "cpu")).cpu().numpy()

submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': preds.flatten()
})

submission['Calories'] = submission['Calories'].clip(lower=0)

submission.to_csv('submission.csv', index=False)
print(submission.head())


