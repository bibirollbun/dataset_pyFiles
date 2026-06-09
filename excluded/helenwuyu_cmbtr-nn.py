#-------------------------
# SET-UP
#-------------------------

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
df_2 = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
df = pd.concat([df, df_2])
    #print(df.head())
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#-------------------------
# CLEANING AND PROCESSING
#-------------------------

# imputing missing values
df_ffill = df.ffill()  # handle missing values with forward fill
df_ffill = df_ffill.bfill()  # and with backfill

# encoding categorical features with one-hot encoding
df_ffill_encoded = pd.get_dummies(df_ffill, drop_first=True)

# splitting the data into features and target variable
X = df_ffill_encoded.drop(columns=['efs']).values  # features
y = df_ffill_encoded['efs'].values                 # target

# scaling X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# converting data into tensors (for neural network)
X = X.astype(np.float32) # first converting booleans in X to ints (1 = True, 0 = False)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print("X: ", X, "\n")
print("Y: ", y, "\n")

# splitting into train/test/validation sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=24)

print("X shape: ", X_train.shape, "\n")
print("Y shape: ", y_train.shape)


#-------------------------
# INITIALIZING MODEL
#-------------------------

# defining a function to create the model
def create_model(input_dim, output_dim, hidden_dim):
  model = nn.Sequential(
      nn.Linear(input_dim, 64), # creating first layer of network
      nn.ReLU(), # activation function
      nn.Linear(64, output_dim),
  )

  return model

model = create_model(X.shape[1], 4, 3)

# defining metric for validation
criterion = nn.CrossEntropyLoss() # best for multi-class classification

# defining optimizer for learning
optimizer = optim.Adam(model.parameters(), lr=0.0001)


#-------------------------
# MODEL TRAINING
#-------------------------

# defining training parameters
num_epochs = 20 # number of training sessions
batch_size = 32 # number of data points used to train at once

# defining validation logs
best_val_loss = float('inf') # var for lowest loss
best_model_state = None # var for best params and state of model

# training loop
for epoch in range(num_epochs):
  model.train() # setting model to training mode

  # loop for training model on each batch of data
  for i in range(0, len(X_train), batch_size):
    X_train_batch = X_train[i:i+batch_size]
    y_train_batch = y_train[i:i+batch_size]

    # resetting optimizer
    optimizer.zero_grad()

    # getting prediction outputs
    outputs = model(X_train_batch)

    # calculating loss (or error) using chosen metric
    loss = criterion(outputs, y_train_batch)

    # doing backward pass
    loss.backward()
    optimizer.step()

  # validating model accuracy at each training session
  model.eval() # setting model to evaluation mode
  with torch.no_grad():
    val_outputs = model(X_val) # getting model's predictions
    val_loss = criterion(val_outputs, y_val) # calculating error

  # if error is current best, save current error and model state
  if val_loss.item() < best_val_loss:
    best_val_loss = val_loss.item()
    best_model_state = model.state_dict() # saving best state (weights)

  # printing loss calculations for every 5th epoch
  if (epoch + 1) % 5 == 0:
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")



#-------------------------
# FINAL EVALUATION
#-------------------------

# load the best model from training
model.load_state_dict(best_model_state)

# Assess model's accuracy and precision
model.eval()  # set model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_preds = torch.max(test_outputs, 1) # getting most probable class
    test_Y_pred = test_outputs.numpy()
    accuracy = accuracy_score(test_preds, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}% \n")

cm = confusion_matrix(y_test, test_preds)
unique_labels = np.unique(y_test)

# Creating the Submission File
with open("submission.csv", "w") as fout:
    fout.write("ID,prediction\n")
    for index, row in df_2.iterrows():
        fout.write(f"{row['ID']},{test_Y_pred[index][0]}\n")


# Reading the test dataset to get the IDs
df_2 = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')

# Creating the Submission File
with open("submission.csv", "w") as fout:
    fout.write("ID,prediction\n")
    for index, row in df_2.iterrows():
        fout.write(f"{row['ID']},{test_Y_pred[index][0]}\n")
    
fout.close()



output = pd.read_csv("submission.csv")
print(output)


pd.DataFrame(output).to_csv(f"submission.csv", index=False)

