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


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# reading dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
extras = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train.head()


print(train.shape)
print(test.shape)
print(extras.shape)


temp_view = pd.concat([train,extras], ignore_index=True)


temp_view.shape


temp_view[temp_view.isnull().any(axis=1)]


temp_view.sample(10)


for i in list(temp_view.columns):
    print(f"{i} :  {temp_view[i].unique()} ---- Element count: {len(temp_view[i].unique())} \n")


# cheking how much percentage is missing data
(temp_view[temp_view.isnull().values == True].shape[0] /temp_view.shape[0] ) *100


print(temp_view["Color"].value_counts())
print(temp_view[temp_view["Color"].isnull() == True].shape[0])


plt.figure(figsize=(8, 5))
plt.hist(temp_view['Color'].dropna(), bins=10, edgecolor='black')  # drop NaN values if necessary
plt.title('Distribution of Color')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


#Random Imputation in color maintaining the distribution
import numpy as np

# Extract non-null values once
non_null_colors = temp_view['Color'].dropna().values

# Create a boolean mask for NaN values in the 'Color' column
mask = temp_view['Color'].isna()

# Fill NaN values with random choices from non_null_colors
temp_view.loc[mask, 'Color'] = np.random.choice(non_null_colors, size=mask.sum())


print(temp_view["Color"].value_counts())
print(temp_view[temp_view["Color"].isnull() == True].shape[0])


plt.figure(figsize=(8, 5))
plt.hist(temp_view['Color'].dropna(), bins=10, edgecolor='black')  # drop NaN values if necessary
plt.title('Distribution of Color')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


for i in list(temp_view.columns):
    print(f"{i} :  {temp_view[i].unique()} ---- Element count: {len(temp_view[i].unique())} \n")


print(temp_view["Waterproof"].value_counts())
print(temp_view[temp_view["Waterproof"].isnull() == True].shape[0])


temp_view.head()


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Assume temp_view is your DataFrame
data = temp_view.copy()

# Define the columns to transform
numeric_cols = ['Compartments', 'Weight Capacity (kg)']
categorical_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Pipeline for numeric columns: impute missing values using median then scale.
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical columns: impute missing values with "Missing" then one-hot encode.
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Combine pipelines using ColumnTransformer.
# 'id' and 'Price' will be passed through unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols),
        ('pass', 'passthrough', ['id', 'Price'])
    ]
)

# Fit and transform the data.
transformed_array = preprocessor.fit_transform(data)

# --- Reconstruct the DataFrame with proper column names ---
# Numeric feature names remain the same.
numeric_feature_names = numeric_cols

# For categorical features, retrieve the one-hot encoded feature names.
cat_feature_names = preprocessor.named_transformers_['cat'] \
                                .named_steps['onehot'] \
                                .get_feature_names_out(categorical_cols)

# The passthrough columns remain unchanged.
passthrough_feature_names = ['id', 'Price']

# Combine all feature names in the order produced by ColumnTransformer:
all_feature_names = list(numeric_feature_names) + list(cat_feature_names) + passthrough_feature_names

# Convert the transformed array back to a DataFrame.
transformed_df = pd.DataFrame(transformed_array, columns=all_feature_names)

# Optionally, display the first few rows.
transformed_df.head()


# features are increased after transformation
print(temp_view.shape)
print(transformed_df.shape)


X = transformed_df.drop(columns=['id', 'Price'])
y = transformed_df['Price']

# Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)



X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)



y_train


batch_size = 1024


# Creating a CustomDataset class
class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features =features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Creating train_dataset object

train_dataset = CustomDataset(X_train,y_train)
val_dataset = CustomDataset(X_val, y_val)


# Create train and test loader
train_loader = DataLoader(train_dataset, batch_size = batch_size,num_workers=2,  shuffle = True,pin_memory=True)
test_loader = DataLoader(val_dataset, batch_size = batch_size,num_workers=2,  shuffle = False,pin_memory=True) # False as we dont want to shuffle the test data which can lead to inaccurate metrics measure.


 # Define Neural Network class

class RegressionNN(nn.Module):
    def __init__ (self, num_features):

        super().__init__()
        self.model = nn.Sequential(
            
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.4),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )

    def forward(self,x):
        return self.model(x)


# setting lr and epochs
learning_rate = 0.0001
epochs = 150


# Instantiate model.
input_dim = X_train.shape[1]
model = RegressionNN(input_dim)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



# Loss function and optimizer.
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)


# training loop
for epoch in range(epochs):

    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:

        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        # forward pass (model.forward(batch_features))
        outputs = model(batch_features)
        
        # calculate loss
        loss = criterion(outputs, batch_labels)
        
        # back pass
        optimizer.zero_grad()
        loss.backward()
        
        # update grad
        optimizer.step()

        total_epoch_loss = total_epoch_loss + loss.item()

    avg_loss = total_epoch_loss/len(train_loader)
    print(f"Epoch : {epoch +1}, Loss: {avg_loss}")


# If the test set doesn't include a 'Price' column, add a dummy column (needed for preprocessor).
if 'Price' not in test.columns:
    test['Price'] = 0

# Transform the test data using the pre-fitted preprocessor.
test_transformed_array = preprocessor.transform(test)
test_all_feature_names = list(numeric_feature_names) + list(cat_feature_names) + list(passthrough_feature_names)
test_transformed_df = pd.DataFrame(test_transformed_array, columns=test_all_feature_names)


# Prepare test features for prediction (drop 'id' and dummy 'Price').
X_test = test_transformed_df.drop(columns=['id', 'Price'])
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)


# Set model to evaluation mode and predict.
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy().flatten()


# Create submission DataFrame: ensure 'id' is from the test data.
submission = pd.DataFrame({
    'id': test_transformed_df['id'],
    'Price': predictions
})


# Save the submission to submission.csv.
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")




