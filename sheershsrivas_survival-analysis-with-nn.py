import pandas as pd
import numpy as np
import torch
import torch.nn as nn




df=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
print(df.columns.tolist())


print(df.dtypes.value_counts())


# creating seperate var for categorical and numerical columns for embeddings
categorical_cols=df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
hla_cols = [col for col in df.columns if col.startswith('hla_')]

# Remove hla_ columns from numerical_cols
numerical_cols = [col for col in numerical_cols if col not in hla_cols]

# Remove specific columns: 'ID', 'efs', 'efs_time'
cols_to_remove = ['ID', 'efs', 'efs_time']
numerical_cols = [col for col in numerical_cols if col not in cols_to_remove]

print(len(categorical_cols), len(numerical_cols), len(hla_cols))


print(categorical_cols)


missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Get data types of each column
data_types = df.dtypes

# Combine everything into a DataFrame
missing_summary = pd.DataFrame({
    'Data Type': data_types,
    'Missing Values': missing_values,
    'Missing Percentage': missing_percentage
})

# Display the result
print(missing_summary.sort_values(by='Missing Percentage', ascending=False))


# we drop 5 columns as they have more than 38% values missing
#                        Data Type  Missing Values  Missing Percentage
# tce_match                 object           18996           65.958333
# mrd_hct                   object           16597           57.628472
# cyto_score_detail         object           11923           41.399306
# tce_div_match             object           11396           39.569444
# tce_imm_match             object           11133           38.656250
# Columns to remove
cols_to_remove = ['tce_match', 'mrd_hct', 'cyto_score_detail', 'tce_div_match', 'tce_imm_match']

# Remove specified columns from categorical_cols
categorical_cols = [col for col in categorical_cols if col not in cols_to_remove]


from sklearn.preprocessing import LabelEncoder

label_encoders = {}

# encoded categorical cloumns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert categories to numbers
    label_encoders[col] = le  # Store encoder for inverse transformation later



# List of numeric columns to convert
numeric_to_categorical = ['donor_age', 'karnofsky_score', 'comorbidity_score']

bins_dict = {
    'donor_age': [0, 20, 40, 60, 80, 100],   # Custom bins for donor_age
    'karnofsky_score': [0, 40, 60, 80, 100],  # Custom bins for karnofsky_score
    'comorbidity_score': [0, 1, 2, 3, 4]      # Custom bins for comorbidity_score
}

# Convert each column to categorical using pd.cut
for col in numeric_to_categorical:
    df[col] = pd.cut(df[col], bins=bins_dict[col], labels=False, include_lowest=True)


print(df['donor_age'].dtype)


# after calculating distribution we can make custom bins for each column
hla_bins = {
    'hla_match_c_high': [0.0, 1.0, 2.0],
    'hla_high_res_8': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'hla_low_res_6': [2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_high_res_6': [0.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_high_res_10': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'hla_match_dqb1_high': [0.0, 1.0, 2.0],
    'hla_nmdp_6': [2.0, 3.0, 4.0, 5.0, 6.0],
    'hla_match_c_low': [0.0, 1.0, 2.0],
    'hla_match_drb1_low': [1.0, 2.0],
    'hla_match_dqb1_low': [0.0, 1.0, 2.0],
    'hla_match_a_high': [0.0, 1.0, 2.0],
    'hla_match_b_low': [0.0, 1.0, 2.0],
    'hla_match_a_low': [0.0, 1.0, 2.0],
    'hla_match_b_high': [0.0, 1.0, 2.0],
    'hla_low_res_8': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'hla_match_drb1_high': [0.0, 1.0, 2.0],
    'hla_low_res_10': [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}
# Apply binning using `pd.cut`
for col, bins in hla_bins.items():
    df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

# Create binary indicators for missing hla_ variable values
for col in hla_cols:
    df[col + '_present'] = np.where(df[col].notna(), 1, 0)

hla_present_cols = [col + '_present' for col in hla_cols]



# Replace NaN with -1 for numeric and hla columns
df[numerical_cols + hla_cols] = df[numerical_cols + hla_cols].fillna(-1)


print(df['hla_match_c_high'])
print(df['hla_match_c_high'].max())


# creating embeddings

# Define Embedding Layers for Categorical Variables
embedding_layers = nn.ModuleList([
    nn.Embedding(len(le.classes_), min(50, (len(le.classes_) + 1) // 2))  # min(50, (len + 1) // 2) is an example
    for le in label_encoders.values()
])



# Define Embedding Layers for 'hla_' columns based on the length of bins
hla_embedding_layers = nn.ModuleList([
    nn.Embedding(int(df[col].max()) + 2, 3)  # Embedding size is 3; len(bins) is the number of unique bins for each column
    for col in hla_cols # Create an embedding for each HLA column
])

# For hla_present columns, we will use binary embeddings (i.e., 0 or 1)
# hla_present_embedding_layers = nn.ModuleList([
#     nn.Embedding(2, 1)  # Binary embedding (0 or 1)
#     for _ in hla_present_cols
# ])

# Define Embedding Layers for Numeric (Binned) Variables
numeric_embedding_layers = nn.ModuleList([
    nn.Embedding(int(df[col].max()) + 1, min(50, (df[col].nunique() + 1) // 2))  # Adjust embedding size
    for col in numerical_cols
])

# Forward pass function (a simplified version)
def forward(data1,data2,data4):
    # Embed categorical columns
    print("Entered categorical embedding")
    embedded_categoricals = [embedding_layers[i](data1[:, i]) for i in range(len(categorical_cols))]
    # print(embedded_categoricals)
    embedded_categoricals = torch.cat(embedded_categoricals, dim=1)
    # print(embedded_categoricals)
    # Embed hla_ columns
    # data_hla = data[:, len(categorical_cols):len(categorical_cols) + len(hla_cols)]  # Slice data for hla_cols
    data2+=1
    # print(data_hla.shape)
    print("Entered hla embedding")
    embedded_hla = [hla_embedding_layers[i](data2[:,i]) for i in range(len(hla_cols))]
    print("done with hla_")
    embedded_hla = torch.cat(embedded_hla, dim=1)

    # Embed hla_present columns
    # print("Entered hla_presence embedding")
    # embedded_hla_present = [hla_present_embedding_layers[i](data3[:, i])
    #                         for i in range(len(hla_present_cols))]
    # embedded_hla_present = torch.cat(embedded_hla_present, dim=1)

    # Embed numeric columns (binned)
    print("Entered numeric embedding")
    embedded_numeric = [numeric_embedding_layers[i](data4[:,i].clamp(0, int(df[numerical_cols[i]].max())))
                        for i in range(len(numerical_cols))]
    embedded_numeric = torch.cat(embedded_numeric, dim=1)

    # Concatenate all embeddings (categoricals, hla, hla_present, and numeric)
    concatenated = torch.cat([embedded_categoricals, embedded_hla, embedded_numeric], dim=1)

    return concatenated


# Data must be properly encoded for categorical columns and numeric columns
input_data = torch.tensor(df[categorical_cols + hla_cols+numerical_cols].values,dtype=torch.long)

# Call the forward function
embeddedTensor = forward(torch.tensor(df[categorical_cols].values,dtype=torch.long),torch.tensor(df[hla_cols].values,dtype=torch.long),
                 torch.tensor(df[numerical_cols].values,dtype=torch.long))


# Print the shape of the output embeddings
print(embeddedTensor.shape)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the model
class SurvivalNN(nn.Module):
    def __init__(self, input_size=202, hidden_size=128, output_size=2, dropout_rate=0.3):
        super(SurvivalNN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input → Hidden
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Normalize hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Hidden → Hidden
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)  # Additional layer
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, output_size)  # Hidden → Output
        # Activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # Apply FC + BN + ReLU
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))  # Additional layer
        x = self.dropout(x)
        x = self.fc4(x)  # Linear output for regression task
        return x

# Initialize model
model = SurvivalNN()

# Print model summary
print(model)

# Example input for the model (size = [28800, 219] as per the embeddings)
# dummy_input = torch.randn(28800, 219)

# # Forward pass
# output = model(dummy_input)

# # Print output shape (For regression, output should be continuous)
# print("Output shape:", output.shape)



# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


criterion = nn.HuberLoss()
# MSELoss()


from torch.utils.data import DataLoader, Dataset, random_split


# Define custom dataset
class SurvivalDataset(Dataset):
    def __init__(self, embeddings, efs, efs_time):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        # Normalize efs and efs_time
        self.efs_mean, self.efs_std = efs.mean(), efs.std()
        self.efs_time_mean, self.efs_time_std = efs_time.mean(), efs_time.std()

        self.efs = torch.tensor((efs - self.efs_mean) / self.efs_std, dtype=torch.float32).unsqueeze(1)
        self.efs_time = torch.tensor((efs_time - self.efs_time_mean) / self.efs_time_std, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.efs[idx], self.efs_time[idx]

    def denormalize(self, efs, efs_time):
        """Denormalize predictions to original scale."""
        efs_original = efs * self.efs_std + self.efs_mean
        efs_time_original = efs_time * self.efs_time_std + self.efs_time_mean
        return efs_original, efs_time_original

# Convert dataset to PyTorch tensors
dataset = SurvivalDataset(embeddedTensor, df['efs'].values, df['efs_time'].values)


# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Define DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        X_batch, efs_batch, efs_time_batch = batch  # Unpack batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        predicted_efs, predicted_efs_time = outputs[:, 0], outputs[:, 1]

        loss_efs = criterion(predicted_efs,  efs_batch.squeeze())
        loss_efs_time = criterion(predicted_efs_time, efs_time_batch.squeeze())
        loss = loss_efs + loss_efs_time  # Combined loss
        # print("Loss:", loss.item())
        # Backward pass and optimization

        loss.backward()  # Compute gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # Update parameters

        train_loss += loss.item()
        # predicted_hazard = predicted_hazard.detach()

     # Compute average train loss
    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            X_batch, efs_batch, efs_time_batch = batch
            outputs = model(X_batch)
            predicted_efs, predicted_efs_time = outputs[:, 0], outputs[:, 1]

            loss_efs = criterion(predicted_efs, efs_batch.squeeze())
            loss_efs_time = criterion(predicted_efs_time, efs_time_batch.squeeze())
            loss = 0.8 * loss_efs + 0.2 * loss_efs_time # Combined loss

            val_loss += loss.item()

    # Compute average validation loss
    val_loss /= len(val_loader)

    # Update scheduler
    # scheduler.step(val_loss)


    # if epoch % 5 == 0:  # Print every 5 epochs
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def preprocess_test_data(df, categorical_cols, numeric_to_categorical, bins_dict, hla_bins, numerical_cols, hla_cols, hla_present_cols):
    """
    Preprocess the test data:
    1. Encode categorical columns.
    2. Convert numeric columns to categorical using binning.
    3. Apply binning for HLA columns.
    4. Replace NaN values with -1.
    """
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert categories to numbers
        label_encoders[col] = le  # Store encoder for inverse transformation later

    # Convert numeric columns to categorical using pd.cut
    for col in numeric_to_categorical:
        df[col] = pd.cut(df[col], bins=bins_dict[col], labels=False, include_lowest=True)

    # Apply binning for HLA columns
    for col, bins in hla_bins.items():
        df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

    # Replace NaN with -1 for numeric and HLA columns
    df[numerical_cols + hla_cols] = df[numerical_cols + hla_cols].fillna(-1)

    return df


# Define custom dataset for test set
class TestDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


# Load test data
test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

# Preprocess test data
test_df = preprocess_test_data(test_df, categorical_cols, numeric_to_categorical, bins_dict, hla_bins, numerical_cols, hla_cols, hla_present_cols)

# Generate embeddings using the forward function
embeddedTestTensor = forward(
    torch.tensor(test_df[categorical_cols].values, dtype=torch.long),
    torch.tensor(test_df[hla_cols].values, dtype=torch.long),
    torch.tensor(test_df[numerical_cols].values, dtype=torch.long)
)

print("Embedded Test Tensor Shape:", embeddedTestTensor.shape)
# Create a dataset for the test set
test_dataset = TestDataset(embeddedTestTensor)

# Create a DataLoader for the test set
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


model.eval()
predicted_efs = []
predicted_efs_times = []

efs_mean, efs_std = df['efs'].mean(), df['efs'].std()
efs_time_mean, efs_time_std = df['efs_time'].mean(), df['efs_time'].std()

with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch)
        predicted_efs, predicted_efs_time = outputs[:, 0], outputs[:, 1]
        # predicted_efs.extend(predicted_efs.cpu().numpy())
        predicted_efs_times.extend(predicted_efs_time.cpu().numpy())

print("Predicted efs:", predicted_efs)
# print("Predicted efs_time:", predicted_efs_times
# Convert predictions to a numpy array
# predicted_efs = np.array(predicted_efs)
predicted_efs_times = np.array(predicted_efs_times)

# Denormalize it
predicted_efs_times = predicted_efs_times* efs_time_std + efs_time_mean
# predicted_efs = predicted_efs * efs_std + efs_mean
# print("Denormalized Predicted efs:", predicted_efs)
print("Denormalized Predicted efs_time:", predicted_efs_time)

# Calculate risk scores (inverse of predicted efs_time)
risk_scores = 1 / predicted_efs_times

# Print risk scores
print("Risk Scores:", risk_scores)

submission_df = pd.DataFrame({'ID': test_df['ID'].astype(int), 'prediction': risk_scores})
submission_df.to_csv('submission.csv', index=False)

# If you have true efs_time values for evaluation (e.g., in a validation set), calculate C-index
# from lifelines.utils import concordance_index

# # Example true efs_time values (replace with actual values if available)
# true_efs_times = np.random.rand(len(predicted_efs_times))  # Replace with actual values

# # Calculate C-index
# c_index = concordance_index(true_efs_times, -predicted_efs_times)  # Negative for correct ranking
# print(f"Concordance Index (C-index): {c_index:.4f}")




