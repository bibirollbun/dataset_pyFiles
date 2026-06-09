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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer


df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


# Initial exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())


# Set up visualization style
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))


# 1. Target variable distribution
plt.subplot(1, 2, 1)
sns.countplot(x='rainfall', data=df)
plt.title('Class Distribution')


# 2. Numerical features distribution
plt.subplot(1, 2, 2)
sns.histplot(df['temparature'], kde=True, color='blue')
plt.title('Temperature Distribution')
plt.tight_layout()
plt.show()


# 3. Correlation heatmap
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()



# 4. Feature relationships with target
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.boxplot(x='rainfall', y='humidity', data=df)
plt.title('Humidity vs Rainfall')

plt.subplot(2,2,2)
sns.boxplot(x='rainfall', y='pressure', data=df)
plt.title('Pressure vs Rainfall')

plt.subplot(2,2,3)
sns.scatterplot(x='temparature', y='dewpoint', hue='rainfall', data=df)
plt.title('Temperature vs Dewpoint')

plt.subplot(2,2,4)
sns.violinplot(x='rainfall', y='windspeed', data=df)
plt.title('Windspeed Distribution by Rainfall')
plt.tight_layout()
plt.show()


# 5. Time series analysis
plt.figure(figsize=(12,6))
df.groupby('day')['rainfall'].mean().plot()
plt.title('Rainfall Probability by Day')
plt.ylabel('Probability of Rainfall')
plt.show()


# 6. Temporal analysis of meteorological parameters
plt.figure(figsize=(15,10))


# Pressure trend
plt.subplot(3,1,1)
sns.lineplot(x='day', y='pressure', hue='rainfall', data=df, palette='viridis')
plt.title('Atmospheric Pressure Trend with Rainfall Indicators')
plt.xlabel('Day')


#Temperature variation
plt.subplot(3,1,2)
sns.lineplot(x='day', y='temparature', hue='rainfall', data=df, palette='coolwarm')
plt.title('Temperature Variation with Rainfall Indicators')


# Humidity progression
plt.subplot(3,1,3)
sns.lineplot(x='day', y='humidity', hue='rainfall', data=df, palette='Blues')
plt.title('Humidity Levels with Rainfall Indicators')
plt.tight_layout()
plt.show()


# 7. Pairwise relationships analysis
sns.pairplot(df[['temparature', 'humidity', 'pressure', 'windspeed', 'rainfall']], 
             hue='rainfall', palette='Set1', corner=True)
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()


# 8. Wind analysis
plt.figure(figsize=(15,5))

# Wind direction distribution
plt.subplot(1,2,1)
sns.scatterplot(x='winddirection', y='windspeed', hue='rainfall', data=df, palette='tab10')
plt.title('Wind Patterns vs Rainfall')

# Polar plot for wind direction
plt.subplot(1,2,2, projection='polar')
wind_directions = np.deg2rad(df['winddirection'])
sns.scatterplot(x=wind_directions, y=df['windspeed'], hue=df['rainfall'], palette='viridis')
plt.title('Wind Direction Distribution (Polar Plot)')
plt.thetagrids(np.arange(0, 360, 45), labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
plt.tight_layout()
plt.show()


# 9. Cumulative distribution analysis
plt.figure(figsize=(12,6))

for feature in ['humidity', 'dewpoint', 'cloud']:
    sns.ecdfplot(data=df, x=feature, hue='rainfall', palette='dark:red', legend=True)
    
plt.title('Cumulative Distribution of Key Features')
plt.xlabel('Feature Values')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()



# 10. Interactive 3D visualization (requires matplotlib 3D axes)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

x = df['temparature']
y = df['dewpoint']
z = df['humidity']

scatter = ax.scatter(x, y, z, c=df['rainfall'], cmap='coolwarm', s=20)
ax.set_xlabel('Temparature')
ax.set_ylabel('Dewpoint')
ax.set_zlabel('Humidity')
plt.title('3D Relationship: Temp-Dewpoint-Humidity')
fig.colorbar(scatter, label='Rainfall')
plt.show()


# 11. Lag correlation analysis
plt.figure(figsize=(10,6))
pd.plotting.lag_plot(df['rainfall'], lag=1)
plt.title('Rainfall Autocorrelation (1-Day Lag)')
plt.show()


# 12. Monthly aggregation analysis (if temporal context exists)
df['month'] = (df['day'] // 30) + 1  # Create pseudo-monthly buckets
monthly_agg = df.groupby('month').agg({'rainfall':'mean', 'temparature':'median'})

plt.figure(figsize=(12,6))
sns.lineplot(data=monthly_agg, markers=True, dashes=False)
plt.title('Monthly Aggregated Trends')
plt.xlabel('Month')
plt.ylabel('Values')
plt.legend(title='Metrics')
plt.show()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


df_train


# Combine the dataframes
df = pd.concat([df_train, df_test], ignore_index=True)


df


import polars  as pl
import os
playground_series_s5e3_path = "../input/playground-series-s5e3"

for dirname, _, filenames in os.walk(f'{playground_series_s5e3_path}'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


raw_data = pl.scan_csv(f'{playground_series_s5e3_path}/train.csv').collect()
display (raw_data.tail())

raw_test = pl.scan_csv(f'{playground_series_s5e3_path}/test.csv').collect()
display (raw_test.tail())

sample_sub = pl.scan_csv(f'{playground_series_s5e3_path}/sample_submission.csv').collect()


features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']


import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

# Assuming 'raw_data' and 'features' are defined from the previous code

# Prepare data
X = raw_data.select(features).to_numpy()
y = raw_data["rainfall"].to_numpy()

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define the model
class RainfallModel(nn.Module):
    def __init__(self, input_dim):
        super(RainfallModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Output between 0 and 1 for ROC
        )
    def forward(self, x):
        return self.layers(x)

# Hyperparameters
input_dim = len(features)
epochs = 2000
learning_rate = 0.000035
k_folds = 7

# Cross-validation and training
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
roc_auc_scores = []
models = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = RainfallModel(input_dim)
    criterion = nn.BCELoss() # Binary Cross Entropy loss for ROC
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early Stopping
    best_roc_auc = 0
    patience = 50
    no_improvement_count = 0
    roc_history = [] 
    for epoch in tqdm(range(epochs)):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            roc_auc = roc_auc_score(y_val.numpy(), val_outputs.numpy())
            roc_history.append (roc_auc)   
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            no_improvement_count = 0
        else:
            no_improvement_count +=1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch} with {best_roc_auc = }")
                break

    roc_auc_scores.append(best_roc_auc)
    if best_roc_auc > 0.85 :
        models.append (model)
    sns.lineplot (y = roc_history, x = list (range (len(roc_history))))
    # plt.title (f"ROC AUC for fold {fold} ")
plt.show ()
print(f"Average ROC AUC across folds: {np.mean(roc_auc_scores):.4f}")



X_test = raw_test.select(features).to_numpy()
X_test = scaler.transform(X_test)  # Use the same scaler fitted on training data
X_test = torch.tensor(X_test, dtype=torch.float32)

# Predict on the test set using the best model from the last fold (or retrain with all data for a final model)
test_predictions = np.zeros ((X_test.shape [0]))
print (test_predictions.shape)
for model in models :
    model.eval()
    with torch.no_grad():
        temp =  model(X_test).numpy().flatten()
        test_predictions = np.add (test_predictions,temp)
        print ("after")
        print (test_predictions.shape)
        print (test_predictions[0:10])

test_predictions = test_predictions / (len (models))

print (test_predictions[0:20])

clean_test_predictions = np.nan_to_num(    test_predictions , nan=0.8)

# Create submission file
submission_df = sample_sub.with_columns(
    pl.Series("rainfall",clean_test_predictions.flatten())
)
submission_df.write_csv("submission.csv")
print("Submission file created successfully!")


submission_df.write_csv("/kaggle/working/submission.csv")




