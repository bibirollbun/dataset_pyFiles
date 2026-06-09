import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)
print('CUDA:', torch.cuda.is_available())


# Load FULL data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


print(f'Full Train Shape: {train_df.shape}')
print(f'Test Shape: {test_df.shape}')

TARGET = 'diagnosed_diabetes'
feature_cols = [c for c in train_df.columns if c not in ['id', TARGET]]

# Store full data for later
X_full = train_df[feature_cols].copy()
y_full = train_df[TARGET].values
X_test = test_df[feature_cols].copy()

# Encode categoricals on full data
label_encoders = {}
for col in X_full.select_dtypes('object').columns:
    le = LabelEncoder()
    X_full[col] = le.fit_transform(X_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Scale full data
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
X_test_scaled = scaler.transform(X_test)

print(f'Features: {X_full_scaled.shape[1]}')


# Sample 1% for fast WOA optimization
sample_idx = np.random.choice(len(X_full_scaled), size=int(0.05*len(X_full_scaled)), replace=False)
X_sample = X_full_scaled[sample_idx]
y_sample = y_full[sample_idx]

# Split sample for validation
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)
print(f'WOA Sample - Train: {len(X_train)}, Val: {len(X_val)}')


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x): return self.net(x)

print(SimpleNN(24, [64, 32]))


class WOA:
    def __init__(self, n_whales, max_iter, dim, lb=16, ub=256):
        self.n, self.T, self.dim, self.lb, self.ub = n_whales, max_iter, dim, lb, ub
        self.history = []

    def optimize(self, fitness_fn):
        pop = np.random.uniform(self.lb, self.ub, (self.n, self.dim))
        best, best_fit = None, -np.inf

        print("Initial population:")
        for i in range(self.n):
            h = [int(round(u)) for u in pop[i]]
            f = fitness_fn(h)
            if f > best_fit: best_fit, best = f, pop[i].copy()
            print(f"  Whale {i+1}: {h} -> {f:.4f}")
        self.history.append(best_fit)
        print(f"Best: {best_fit:.4f}\n")

        for t in range(self.T):
            a = 2 - 2*t/self.T
            for i in range(self.n):
                r, p = np.random.random(self.dim), np.random.random()
                A, C = 2*a*r - a, 2*np.random.random(self.dim)
                
                if p >= 0.5:
                    D = np.abs(best - pop[i])
                    l = np.random.uniform(-1, 1, self.dim)
                    pop[i] = D * np.exp(l) * np.cos(2*np.pi*l) + best
                elif np.abs(A).mean() < 1:
                    pop[i] = best - A * np.abs(C * best - pop[i])
                else:
                    rand = pop[np.random.randint(self.n)]
                    pop[i] = rand - A * np.abs(C * rand - pop[i])
                pop[i] = np.clip(pop[i], self.lb, self.ub)

            iter_best_fit = -np.inf
            for i in range(self.n):
                h = [int(round(u)) for u in pop[i]]
                f = fitness_fn(h)
                if f > best_fit: best_fit, best = f, pop[i].copy()
                if f > iter_best_fit: iter_best_fit = f
            self.history.append(best_fit)
            print(f"Iter {t+1}/{self.T}: iter_best={iter_best_fit:.4f}, global_best={best_fit:.4f}, h={[int(round(u)) for u in best]}")

        return [int(round(u)) for u in best], best_fit


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def train_eval(hidden, epochs=20, bs=512, lr=0.001):
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).reshape(-1,1).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    
    model = SimpleNN(X_train.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True)
    
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    
    model.eval()
    with torch.no_grad():
        preds = model(X_v).cpu().numpy().flatten()
    return roc_auc_score(y_val, preds)


woa = WOA(n_whales=10, max_iter=100, dim=3, lb=10, ub=128)
best_hidden, best_auc = woa.optimize(train_eval)

print(f"WOA COMPLETE! Best hidden: {best_hidden}, ROC-AUC: {best_auc:.4f}")


plt.plot(woa.history, 'b-o')
plt.xlabel('Iteration'); plt.ylabel('Best ROC-AUC')
plt.title('WOA Convergence'); plt.grid(True)
plt.show()


from sklearn.model_selection import train_test_split

# Split full data for evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X_full_scaled, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Prepare tensors
X_t = torch.FloatTensor(X_train).to(device)
y_t = torch.FloatTensor(y_train).reshape(-1,1).to(device)
X_v = torch.FloatTensor(X_val).to(device)

# Train model
model = SimpleNN(X_train.shape[1], best_hidden).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.0004)
loss_fn = nn.BCELoss()
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)

model.train()
for epoch in range(100):
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

# Evaluate
model.eval()
with torch.no_grad():
    val_preds = model(X_v).cpu().numpy().flatten()

auc = roc_auc_score(y_val, val_preds)
print(f"Best hidden: {best_hidden}, ROC-AUC on validation set: {auc:.4f}")

