# imports
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


# load data
train_data = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
test_data = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')


with pd.option_context('display.max_columns', None):
    display(train_data)


# data preprocessing

# drop unimportant values
train_data.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)

# encode string values to ints
train_data['matchType'], mapping = pd.factorize(train_data['matchType'])
match_type = {value: index for value, index in enumerate(mapping)}

# remove nan values
train_data.dropna(inplace=True)


train_data.info()


train_data.isna().sum()


train_data['winPlacePerc'].value


Q1 = train_data['winPlacePerc'].quantile(0.25)
Q3 = train_data['winPlacePerc'].quantile(0.75)

IQR = Q3 - Q1

outliers = train_data[(train_data['winPlacePerc'] > 1.5 * Q1) |
           (train_data['winPlacePerc'] < 1.5 * Q3)]


numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('winPlacePerc')


plt.figure(figsize=(12,5))
plt.hist(train_data['killPlace'], bins=100, color='green', rwidth=0.90)
plt.show()


train_data['killPlace'].value_counts()


train_data['killPlace'].plot(kind='bar')


fig, axs = plt.subplots(7, 4, figsize=(20, 20), sharex=True)
axs = axs.flatten()

for idx, col in enumerate(numerical_cols):
    train_data[col].plot(ax=axs[idx])
    axs[idx].set_title(col)

plt.tight_layout()
plt.show()


X = train_data.drop('winPlacePerc', axis=1)
Y = train_data['winPlacePerc']


# NaN values visualization

nan_counts = train_data.isna().sum()

# Plot bar chart
plt.figure(figsize=(20, 10))
nan_counts.plot(kind='bar', color='red', edgecolor='black')
plt.title("Missing Values Per Column")
plt.xlabel("Columns")
plt.ylabel("Count of NaNs")
plt.xticks(rotation=45)
plt.show()


# check how our target data looks like

plt.figure(figsize=(12,5))
plt.hist(train_data['winPlacePerc'], bins=50, color='green', rwidth=0.90)
plt.title("winPlacePerc")
plt.show()


# Check box plot for outliners and distribution

plt.figure(figsize=(12, 5))
plt.boxplot(train_data['winPlacePerc'])
plt.title('Target Boxplot')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# Convert data to DMatrix for XGBoost API

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# Train baseline XGBoost classifier
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.1,
    "max_depth": 6,
    "seed": 42
}
bst = xgb.train(params, dtrain, num_boost_round=50, 
                early_stopping_rounds=5, 
                evals=[(dtrain, "train"), (dtest, "test")], verbose_eval=True)

'''
# 4. Evaluate
y_pred_prob = bst.predict(dtest)
y_pred = y_pred_prob.argmax(axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Hyperparameter tuning with scikit-learn API
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", seed=42)
param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "subsample": [0.6, 0.8, 1.0]
}
grid = GridSearchCV(xgb_clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# 6. Feature importance
best = grid.best_estimator_
importances = best.feature_importances_
plt.barh(iris.feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance")
plt.show()
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# General syntax:

# Creating tensors
x = torch.tensor([1, 2, 3, 4, 5])  # From list
y = torch.zeros(3, 4)              # Zeros tensor
z = torch.randn(2, 3)              # Random normal distribution
w = torch.ones_like(z)             # Same shape as z, filled with ones

# Tensor properties
print(f"Shape: {z.shape}")         # torch.Size([2, 3])
print(f"Data type: {z.dtype}")     # torch.float32
print(f"Device: {z.device}")       # cpu or cuda

# Mathematical operations
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

addition = a + b                    # Element-wise addition
multiplication = a * b              # Element-wise multiplication
dot_product = torch.dot(a, b)       # Dot product
matrix_mult = torch.mm(a.unsqueeze(0), b.unsqueeze(1))  # Matrix multiplication

# Reshaping and indexing
tensor = torch.randn(4, 4)
reshaped = tensor.view(2, 8)        # Reshape (must maintain total elements)
sliced = tensor[:2, 1:3]           # Slicing like NumPy

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move tensors to GPU
tensor_gpu = tensor.to(device)
# or
tensor_gpu = tensor.cuda()         # If you're sure GPU is available

# Always ensure tensors are on the same device for operations


# Pytroch Classes
class Iris_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(perceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        output = self.layers(x)
        return output


# prepare dataset:
iris_data = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris_x = iris_data.drop(['Species', 'Id'], axis=1)

iris_y = iris_data['Species']
iris_y, target = iris_y.factorize()
target_idx = {values: idx for values, idx in enumerate(target)}

# split data to train and test datasets
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, train_size=0.8)

# convert data into tensors
x_train = torch.tensor(x_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# prepare test and train datasets
train_dataset = Iris_Dataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)

test_dataset = Iris_Dataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)


flower_classifier = perceptron(4, 32, 3)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(flower_classifier.parameters(), lr=0.001)

num_epochs = 100
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

flower_classifier.to(device)

for epoch in range(num_epochs):
    # Training phase
    flower_classifier.train()
    running_loss = 0.0
    
    for data, labels in tqdm(train_dataloader, desc='Training loop'):
        # Move inputs and labels to the device
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = flower_classifier(data)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(dataloader.dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    flower_classifier.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc='Validation loop'):
            # Move inputs and labels to the device
            data, labels = data.to(device), labels.to(device)
         
            outputs = flower_classifier(data)
            loss = loss_func(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(dataloader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

