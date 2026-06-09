import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

# Load the dataset
train_file_path = "/kaggle/input/app-of-gen-ai-deep-learning-wustl-spring-2025/train.csv"
test_file_path = "/kaggle/input/app-of-gen-ai-deep-learning-wustl-spring-2025/test.csv"

df = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# Handle categorical columns by encoding them
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    if col in df_test.columns:
        df_test[col] = df_test[col].astype(str)
        df_test[col] = df_test[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
        df_test[col] = label_encoders[col].transform(df_test[col])

# Define features and target
X = df.drop(columns=['id', 'performance_score'])  # Remove ID and target column
y = df['performance_score']

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define a simple PyTorch regression model
class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(RegressionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
model = RegressionNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy().flatten()

# Compute R-squared
r2 = r2_score(y_test, y_pred_test)
print(f"R-squared: {r2:.4f}")



from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Ensure the model is in evaluation mode
model.eval()

# Compute feature importance using permutation importance
def compute_permutation_importance(model, X_test, y_test, metric=r2_score):
    baseline_score = metric(y_test, model(X_test).detach().numpy().flatten())
    importance_scores = []
    
    for i in range(X_test.shape[1]):
        X_test_permuted = X_test.clone()
        X_test_permuted[:, i] = X_test_permuted[:, i][torch.randperm(X_test.shape[0])]  # Shuffle one feature
        permuted_score = metric(y_test, model(X_test_permuted).detach().numpy().flatten())
        importance_scores.append(baseline_score - permuted_score)

    return np.array(importance_scores)

importances = compute_permutation_importance(model, X_test_tensor, y_test)

# Retrieve original feature names from the DataFrame before conversion to NumPy
feature_names = df.drop(columns=['id', 'performance_score']).columns

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in PyTorch Model (Permutation Importance)")
plt.gca().invert_yaxis()
plt.show()



# Prepare Kaggle submission
test_ids = df_test['id']
df_test = df_test.drop(columns=['id'])

# Standardize test data using the same scaler
X_test_final = scaler.transform(df_test)

# Convert to PyTorch tensor
X_test_final_tensor = torch.tensor(X_test_final, dtype=torch.float32)

# Generate predictions
model.eval()
with torch.no_grad():
    submission_preds = model(X_test_final_tensor).numpy().flatten()

# Create submission DataFrame
submission = pd.DataFrame({'id': test_ids, 'performance_score': submission_preds})
submission.to_csv("submission.csv", index=False)

print("Kaggle submission file 'submission.csv' created successfully.")


