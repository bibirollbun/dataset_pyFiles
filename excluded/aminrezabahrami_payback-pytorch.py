import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Re-load original dataframes to ensure fresh state for imputation
original_train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

# Apply categorical encoding to both train and test dataframes
for df_temp in [original_train_df, test_df]:
    df_temp['gender'] = df_temp['gender'].astype('category').cat.codes
    df_temp['marital_status'] = df_temp['marital_status'].astype('category').cat.codes
    df_temp['education_level'] = df_temp['education_level'].astype('category').cat.codes
    df_temp['employment_status'] = df_temp['employment_status'].astype('category').cat.codes
    df_temp['loan_purpose'] = df_temp['loan_purpose'].astype('category').cat.codes
    df_temp['grade_subgrade'] = df_temp['grade_subgrade'].astype('category').cat.codes

# Prepare X and y from the training dataframe
X = original_train_df.drop('loan_paid_back', axis=1).drop('id', axis=1)
y = original_train_df['loan_paid_back']

# Prepare T from the test dataframe
D = test_df['id'] # Store ids for submission
T = test_df.drop('id', axis=1)

# Identify numerical and categorical columns for imputation and scaling
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
categorical_cols_encoded = [col for col in X.columns if col not in numerical_cols]

# Impute missing values (NaN for numerical, -1 for encoded categoricals) in X and T
for col in numerical_cols:
    mean_val = X[col].mean()
    X.loc[:, col] = X[col].fillna(mean_val) # Fixed: use .loc for explicit assignment
    T.loc[:, col] = T[col].fillna(mean_val) # Fixed: use .loc for explicit assignment

for col in categorical_cols_encoded:
    # Impute -1 with the mode (excluding -1) of the training data
    # Check if there are values other than -1 before calculating mode
    if (X[col] != -1).any():
        mode_val_X = X.loc[X[col] != -1, col].mode()[0]
        X.loc[:, col] = X[col].replace(-1, mode_val_X)
    else: # If all values are -1, set a default (e.g., 0 or handle as truly missing)
        X.loc[:, col] = X[col].replace(-1, 0) # Placeholder, adjust as needed

    # Impute -1 in test data as well, using training data's mode
    if (T[col] != -1).any():
        # Use mode_val_X calculated from training data
        T.loc[:, col] = T[col].replace(-1, mode_val_X)
    else:
        T.loc[:, col] = T[col].replace(-1, 0) # Placeholder, adjust as needed


# Initialize and fit the StandardScaler on training data's numerical columns
scaler = StandardScaler()
X_scaled_numerical = scaler.fit_transform(X[numerical_cols])
T_scaled_numerical = scaler.transform(T[numerical_cols])

# Assign scaled numerical features back to DataFrames, converting NumPy arrays to DataFrames first
X.loc[:, numerical_cols] = pd.DataFrame(X_scaled_numerical, columns=numerical_cols, index=X.index)
T.loc[:, numerical_cols] = pd.DataFrame(T_scaled_numerical, columns=numerical_cols, index=T.index)

# Display X.info() and X.describe() as per subtask instruction
print("Info for X DataFrame:")
X.info()
print("\nDescriptive statistics for X DataFrame:")
X.describe()



# Model definition (re-defined for completeness in this block)
class Model(nn.Module):
    def __init__(self,in_features=11,h1=32,h2=32,out_features=1):
        super().__init__()
        self.fc1=nn.Linear(in_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,out_features)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Initialize the model and move it to the device
torch.manual_seed(32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

# Convert preprocessed data to tensors and move to device
X_tensor = torch.FloatTensor(X.values).to(device)
y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(device)

# Calculate pos_weight for BCEWithLogitsLoss
num_pos = (y_tensor == 1).sum()
num_neg = (y_tensor == 0).sum()
pos_weight = num_neg / num_pos

cri = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], device=device)
)

optimz = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nStarting model training...")
for i in range(100):
    optimz.zero_grad()
    pred = model(X_tensor)          # prefer model(X_tensor)
    pred = pred.view(-1, 1)         # ensure shape (N,1)
    loss = cri(pred, y_tensor)      # (logits, float labels)
    loss.backward()
    optimz.step()
    if (i+1) % 10 == 0 or i == 0: # Print loss periodically
        print(f"Epoch {i+1}: Loss = {loss.item():.6f}")

print("Model training complete.")


# Prepare test data (T) for prediction by converting to tensor and moving to device
T_tensor = torch.FloatTensor(T.values).to(device)

model.eval()

preds = []

with torch.no_grad():
    # Process T_tensor in batches or individually based on memory/model preference
    # For simplicity, processing individually as done before. If T_tensor is very large,
    # batching would be more efficient.
    for x_batch in T_tensor:
        p = torch.sigmoid(model(x_batch))   # logits -> probabilities
        preds.append(p.cpu())

preds = torch.cat(preds).numpy().ravel()
preds = np.round(preds, 1)

submission = pd.DataFrame({
    "id": D.values,
    "loan_paid_back": preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv written without rounding predictions.")

