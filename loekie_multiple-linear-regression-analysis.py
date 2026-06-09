import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Data initialisation
data = [
    [0.836, 1536, 64, 256, 16000],
    [0.835, 2048, 64, 256, 16000],
    [0.835, 2048, 128, 256, 16000],
    [0.835, 1792, 64, 256, 16000],
    [0.834, 1034, 64, 136, 16000],
    [0.831, 2048, 128, 200, 16000],
    [0.829, 2048, 512, 512, 16000],
    [0.828, 1536, 64, 512, 16000],
    [0.824, 1280, 64, 256, 16000],
    [0.814, 1152, 64, 136, 16000],
    [0.814, 1024, 64, 256, 16000],
    [0.810, 1034, 64, 180, 16000],
    [0.810, 1034, 128, 130, 16000],
    [0.810, 1024, 64, 144, 16000],
    [0.806, 1024, 64, 120, 16000],
    [0.803, 1536, 64, 256, 10000],
    [0.795, 4096, 64, 448, 16000],
    [0.773, 512, 128, 256, 16000],
    [0.766, 512, 64, 256, 16000],
    [0.764, 512, 128, 352, 16000],
    [0.754, 1536, 128, 64, 16000],
    [0.754, 1536, 64, 64, 16000],
    [0.742, 1024, 256, 64, 12000],
    [0.736, 512, 128, 64, 16000],
    [0.734, 512, 64, 64, 16000],
    [0.828, 1536, 128, 512, 16000],
    [0.822, 1536, 64, 256, 14000],
    [0.824, 1536, 256, 256, 14000],
    [0.822, 2304, 64, 256, 14000],
    [0.769, 512, 1024, 104, 16000],

]

df = pd.DataFrame(data, columns=['LB', 'N_FFT', 'HOP_LENGTH', 'N_MELS', 'FMAX'])

# Preprocess data
X = df.drop('LB', axis=1).values
y = df['LB'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Load data
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)

# Define Neural Network 
class LBRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

model = LBRegressor()

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 25
losses = []

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).squeeze()
    mse = ((preds - y_test_tensor.squeeze()) ** 2).mean().item()
    print(f"Test MSE: {mse:.5f}")

# Plotting the loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

# Compare true value and predicted value 
for i in range(len(y_test)):
    print(f"Actual: {y_test[i]:.3f}, Predicted: {preds[i].item():.3f}")


# Search space 
search_space = {
    'N_FFT': np.arange(512, 4097, 128),       
    'HOP_LENGTH': np.arange(32, 1025, 32),
    'N_MELS': np.arange(64, 513, 8),
    'FMAX': [12000, 14000, 16000]          
}

# Generate parameter combinations 
N_FFT_options = [512, 1024, 1152, 1536, 2048]
HOP_OPTIONS = [64, 128, 256, 512]
MELS_OPTIONS = [64, 128, 136, 180, 200, 256]
FMAX_OPTIONS = [12000, 16000]

search_space = list(product(N_FFT_options, HOP_OPTIONS, MELS_OPTIONS, FMAX_OPTIONS))

# Scale the parameter combinations
random_inputs = np.array(search_space)
random_inputs_scaled = scaler.transform(random_inputs)

# Predict scores for parameter combinations
model.eval()
with torch.no_grad():
    inputs_tensor = torch.tensor(random_inputs_scaled, dtype=torch.float32)
    preds = model(inputs_tensor).squeeze().numpy()

# Sort Results 
results_df = pd.DataFrame(random_inputs, columns=['N_FFT', 'HOP_LENGTH', 'N_MELS', 'FMAX'])
results_df['Predicted_LB'] = preds
result_1 = results_df.sort_values(by='Predicted_LB', ascending=False).reset_index(drop=True)

print("\nTop 10 Predicted Configurations NN:")
print(result_1.head(10))


# Make Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluation
r2 = reg.score(X_test, y_test)
print(f"\nR² on test data: {r2:.4f}")
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

# Generate parameter combinations 
N_FFT_options = [512, 1024, 1152, 1536, 2048]
HOP_OPTIONS = [64, 128, 256, 512]
MELS_OPTIONS = [64, 128, 136, 180, 200, 256]
FMAX_OPTIONS = [12000, 16000]

search_space = list(product(N_FFT_options, HOP_OPTIONS, MELS_OPTIONS, FMAX_OPTIONS))

results = []

# Predict scores for parameter combinations
for combo in search_space:
    X_candidate = np.array(combo).reshape(1, -1)
    X_scaled = scaler.transform(X_candidate)
    pred_lb = reg.predict(X_scaled)[0]
    results.append(round(pred_lb, 5))

# Sort Results 
results_df = pd.DataFrame(search_space, columns=['N_FFT', 'HOP_LENGTH', 'N_MELS', 'FMAX'])
results_df['Predicted_LB'] = results
result_LR = results_df.sort_values(by='Predicted_LB', ascending=False).reset_index(drop=True)

print("\nTop 10 Predicted Configurations NN:")
print(result_LR.head(10))


# This data is from the original notebook
data = [
    [1, 0.77, 0.93, 3.6371, 0.6873, 1024, 512, 128],
    [2, 0.93, 0.95, 2.4111, 0.5505, 1024, 512, 128],
    [3, 0.96, 0.95, 1.8917, 0.5005, 1024, 512, 128],
    [4, 0.98, 0.97, 1.5317, 0.4349, 1024, 512, 128],
]
df_original_data = pd.DataFrame(data, columns = ['epoch', 'auc', 'auc_val', 'loss', 'loss_val', 'fft', 'hop', 'mels'])


# Get the .csv file
df = pd.read_csv('/kaggle/input/training-results-resnet/results_training.csv')

# Extract values from modelnames
df['fft'] = df['name'].apply(lambda x: ''.join(c for c in x.split("_")[2] if c.isdigit()))
df['hop'] = df['name'].apply(lambda x: ''.join(c for c in x.split("_")[3] if c.isdigit()))
df['mels'] = df['name'].apply(lambda x: ''.join(c for c in x.split("_")[4] if c.isdigit()))
df = df.drop(['name'], axis=1)
df = pd.concat([df, df_original_data]).reset_index().drop(['index'], axis=1)

# Extract data for each epoch
epoch1 = df[df['epoch']==1].drop(['epoch'], axis=1)
epoch2 = df[df['epoch']==2].drop(['epoch'], axis=1)
epoch3 = df[df['epoch']==3].drop(['epoch'], axis=1)
epoch4 = df[df['epoch']==4].drop(['epoch'], axis=1)


epoch_list = [epoch1, epoch2, epoch3, epoch4]

for j in range(4):
    print(f"\n{'='*20} Analyzing Epoch {j} {'='*20}")

    ys = ['auc', 'auc_val', 'loss', 'loss_val']
    
    # Multivariate targets
    X = epoch_list[j].drop(columns=ys).values
    Y = epoch_list[j][ys].values  # Y is now a matrix with 4 columns (multivariate)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Multivariate Linear Regression
    reg = LinearRegression()
    reg.fit(X_train, Y_train)

    # Evaluate performance for each target
    R2_scores = reg.score(X_test, Y_test)  # This is averaged R² across outputs
    print(f"Average R² on test data: {R2_scores:.4f}")
    print(f"Coefficients (per output):\n{reg.coef_}")
    print(f"Intercepts (per output): {reg.intercept_}")

    # Parameter search space
    N_FFT_options = [512, 768, 1024, 1152, 1280, 1536, 1792, 2048]
    HOP_OPTIONS = [64, 86, 128, 192, 256, 512]
    MELS_OPTIONS = [64, 128, 136, 180, 200, 256]

    search_space = list(product(N_FFT_options, HOP_OPTIONS, MELS_OPTIONS))

    predictions = []
    for combo in search_space:
        X_candidate = np.array(combo).reshape(1, -1)
        X_scaled = scaler.transform(X_candidate)
        preds = reg.predict(X_scaled)[0]
        predictions.append((*combo, *preds))  # fft, hop, mels, auc, auc_val, loss, loss_val

    results_df = pd.DataFrame(
        predictions,
        columns=['fft', 'hop', 'mels', 'auc', 'auc_val', 'loss', 'loss_val']
    )

    # Display sorted results for each target
    for target in ys:
        ascending = True if 'loss' in target else False
        sorted_df = results_df.sort_values(by=target, ascending=ascending)
        print(f"\n--- Top results sorted by '{target}' ---")
        display(sorted_df.reset_index(drop=True))




