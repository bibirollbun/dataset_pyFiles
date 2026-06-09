# IMporting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the files
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


# The shape of the data
df_train.shape


# The number of null values in the data
df_train.isnull().sum()


# View of the first few rows of the dataset
df_train.head()


# Checking for distribution of the data
df_train['rainfall'].value_counts()


# Defining a wrangle function for cleanining the data
def wrangle(filepath):
  # Importing the data through the function
  df = pd.read_csv(filepath)

  # Dropping less contributung columns
  df = df.drop(columns='id')

  # Changing predictions to float values for easy use in the model
  df['rainfall'] =df['rainfall'].apply(lambda x: 1.0 if x == 1 else 0.0)
    
  return df


# Passing the data through the function
df_train_cleaned = wrangle('/kaggle/input/playground-series-s5e3/train.csv')


df_train_cleaned.head()


from sklearn.model_selection import train_test_split
# features
X = df_train_cleaned.drop(columns=['rainfall'])

# target
y = df_train_cleaned['rainfall']


# feature selection
from mlxtend.feature_selection import SequentialFeatureSelector
from catboost import CatBoostClassifier
model_feature_selection=CatBoostClassifier()

forward_feature_selection = SequentialFeatureSelector( 
    model_feature_selection,
    k_features=(1,11),
    forward=True,
    floating=False,
    verbose=2,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,).fit(X, y)


# Best features
forward_feature_selection.k_feature_names_


# Getting columns from X_train
FEATURES = ['day', 'maxtemp', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine']
print (FEATURES)


# After feature engineering
#features
X = X[FEATURES].values

# target
y = df_train_cleaned['rainfall'].values


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X)
# X_scaled[:,5]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=256)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


y_train = np.array(y_train)
y_test = np.array(y_test)


# converting X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Converting y features to tensors
y_train = torch.FloatTensor(y_train).view(-1,1)
y_test = torch.FloatTensor(y_test).view(-1,1)


# Model

class RainFallClassification(nn.Module):
    def __init__(self, in_feature=7, h1=8, h2=8, h3=8, h4=8,h5=4, out_feature=1):
        super(RainFallClassification, self).__init__()
        self.fc1 = nn.Linear(in_feature, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 =nn.Linear(h2, h3)
        self.fc4 =nn.Linear(h3, h4)
        self.fc5 =nn.Linear(h4, h5)
        self.out = nn.Linear(h5, out_feature)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return self.sigmoid(x)

# picking a seed for randomization
torch.manual_seed(256)

# Initialize the model properly
model = RainFallClassification()
        
# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)


# Training the model
epochs = 100
losses = []

for epoch in range(epochs):
    # Get prediction
    pred = model.forward(X_train)

    # Get loss
    loss = criterion(pred, y_train)

    # Appending the losses to keep track of it
    losses.append(loss.detach().numpy())

    # Print output after every 10 epochs
    if epoch%20 ==0:
        print(f"Epoch {epoch + 1} and Loss: {loss} ")

    # Setting back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    


# PLotting the training results
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('losses')
plt.title('Performance graph')
plt.show()


# Evaluating the model
with torch.no_grad():
    y_val = model(X_test)
    loss = criterion(y_val, y_test)
print(f"The loss is: {loss}")


# Evaluation mode (no gradients needed)
model.eval()
with torch.no_grad():
    outputs = model(X_test) # Get probabilities
  
    predicted_labels = (outputs > 0.5).int()  # Convert to binary labels (0 or 1)

# Compare predictions to true labels (y_test)
correct = (predicted_labels == y_test).sum().item()
print(f" Ratio: {correct}/{len(y_test)}\n Accuracy: {correct / len(y_test) * 100:.2f}% \n wrong_labels: {len(y_test)-correct}" )



df_test.head()


# Precessing test data to make predictions

test = df_test.copy(deep=True)
test = test[FEATURES].values


# Converting values to tensor
test = torch.FloatTensor(test)


test.shape, df_test.shape


# Making prediction
with torch.no_grad():
    prediction = model(test)



# Adding prediction to test data
df_test['rainfall'] = prediction



# Final submission
submission = df_test[['id', 'rainfall']]
submission.to_csv('submission.csv', index=False)


#!rm -rf /kaggle/working/

