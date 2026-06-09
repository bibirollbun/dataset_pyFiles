import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
df.head()


df.info()


import matplotlib.pyplot as plt

plt.hist(df['accident_risk'], bins=40, edgecolor='black')


df.isna().sum()


numerical_features  = df.select_dtypes(include=['number']).columns.difference(['accident_risk'])
numerical_features 


categorical_features  = df.select_dtypes(exclude=['number']).columns
print(categorical_features)
print(len(categorical_features))


target = 'accident_risk'


# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)


df.info()


bool_cols = df.select_dtypes(include=['bool']).columns
bool_cols


for col in bool_cols:
    df[col] = df[col].astype(int)


df.head()


# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Unsqueeze for a column vector
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


import torch.nn as nn

class AccidentRiskPredictor(nn.Module):
    def __init__(self, input_size):
        super(AccidentRiskPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128) # Matches output of layer1
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64) # Matches output of layer2
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32) # Matches output of layer3
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x 


input_size = X_train_tensor.shape[1]
model = AccidentRiskPredictor(input_size)
print(model)


import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

learning_rate = 0.001
epochs = 50
batch_size = 64
patience = 5
weight_decay = 1e-4

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)


# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Instantiate the model
input_size = X_train_tensor.shape[1]
model = AccidentRiskPredictor(input_size).to(device) # Move the model to the GPU


for epoch in range(epochs):
    if early_stop:
        print("Early stopping triggered.")
        break
        
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_X_val, batch_y_val in val_loader:
            batch_X_val = batch_X_val.to(device)
            batch_y_val = batch_y_val.to(device)
            outputs_val = model(batch_X_val)
            val_loss += criterion(outputs_val, batch_y_val).item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_imporve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}')

print('Training finished.')


from sklearn.metrics import mean_squared_error, r2_score

# Evaluation on the test set
model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Move test data to the GPU
    X_test_tensor = X_test_tensor.to(device)
    
    # Make predictions
    test_predictions = model(X_test_tensor)

    # Move predictions and true values back to CPU for evaluation
    test_predictions_np = test_predictions.squeeze().cpu().numpy()
    y_test_np = y_test_tensor.squeeze().cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_np, test_predictions_np)
    r2 = r2_score(y_test_np, test_predictions_np)
    
    print(f'Root Mean Squared Error on Test Set: {mse:.4f}')
    print(f'R-squared on Test Set: {r2:.4f}')

# Generate submission file (assuming your test set has an 'id' column)
# test_df = pd.read_csv('your_test_data.csv')
# X_kaggle_test = ... (preprocess the Kaggle test data in the same way)
# X_kaggle_test_tensor = torch.tensor(X_kaggle_test.values, dtype=torch.float32)

# with torch.no_grad():
#     kaggle_predictions = model(X_kaggle_test_tensor).squeeze().numpy()
#
# submission_df = pd.DataFrame({'id': test_df['id'], 'accident_risk': kaggle_predictions})
# submission_df.to_csv('submission.csv', index=False)


# Generate submission file (assuming your test set has an 'id' column)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
categorical_features  = test_df.select_dtypes(exclude=['number']).columns
X_kaggle_test = pd.get_dummies(test_df, columns=categorical_features, drop_first=True)
bool_cols = X_kaggle_test.select_dtypes(include=['bool']).columns
for col in bool_cols:
    X_kaggle_test[col] = X_kaggle_test[col].astype(int)

X_kaggle_test = X_kaggle_test.drop('id', axis=1)
X_kaggle_test.head()


X_kaggle_test_tensor = torch.tensor(X_kaggle_test.values, dtype=torch.float32).to(device)

with torch.no_grad():
    kaggle_predictions = model(X_kaggle_test_tensor).cpu().squeeze().numpy()

submission_df = pd.DataFrame({'id': test_df['id'], 'accident_risk': kaggle_predictions})
submission_df.to_csv('submission_t1.csv', index=False)




