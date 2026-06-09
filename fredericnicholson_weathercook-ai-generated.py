import os
# prompt: write a oneline python code that sets the variable COLAB to True if running in the colab jupyter envrionment and False everywhere else

COLAB = 'google.colab' in str(get_ipython())
print ( f"still using {COLAB = }")

if COLAB :
    from google.colab import drive, userdata
    COLAB = True
    print("Note: using Google CoLab")
    import kagglehub
    kagglehub.login()



if COLAB :
    playground_series_s5e3_path = kagglehub.competition_download('playground-series-s5e3')
    print('Data source import complete.')
else :
    playground_series_s5e3_path = "../input/playground-series-s5e3"


import numpy as np # linear algebra
import polars  as pl # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(f'{playground_series_s5e3_path}'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



raw_data = pl.scan_csv(f'{playground_series_s5e3_path}/train.csv').collect()
display (raw_data.tail())

raw_test = pl.scan_csv(f'{playground_series_s5e3_path}/test.csv').collect()
display (raw_test.tail())

sample_sub = pl.scan_csv(f'{playground_series_s5e3_path}/sample_submission.csv').collect()



# prompt: a eda for raw_data

# Visualizations (example using matplotlib and seaborn - install if needed)
# !pip install matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values
print(raw_data.null_count())

# Summary statistics for numerical features
print(raw_data.describe())

# Explore data types of each column
print(raw_data.schema)
print(raw_data.columns)

# Unique values in categorical columns (example for 'feature_0')
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
for f in features :
    sns.barplot ( data = raw_data.group_by(f).len().to_pandas(), x = f, y = "len")
    plt.title(f"Unique values for {f}")
    plt.show()



# Distribution of target variable
plt.figure(figsize=(8, 6))
sns.histplot(data = raw_data['rainfall'], bins = 2 )
plt.title('Distribution of Target Variable')
plt.show()


# Correlation matrix (numerical features)
numerical_cols = raw_data.select(pl.col(pl.Float64)).columns
correlation_matrix = raw_data.select(numerical_cols).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix.to_pandas(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Boxplots for numerical features (example for 'feature_0')
plt.figure(figsize=(4, 3))
for f in features :
  sns.boxplot(x=raw_data[f])
  plt.title(f'Boxplot of {f}')
  plt.show()


import itertools

# Pairplot for a subset of numerical features (example for first 3 numerical features)
if len(numerical_cols) >= 3:
    for (a,b) in itertools.permutations (features, 2) :
        sns.scatterplot(raw_data.select([a,b, "rainfall"]).to_pandas(), x = a, y = b, hue = "rainfall")
        plt.show()


# prompt create a python function that adds "weather index" to a polars data frame based on the following formula

# 'weather_index' = 0.4 * 'humidity' + 0.3 * 'cloud'- 0.3 * 'sunshine'

def calculate_weather_index(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates a "weather index" based on the provided formula and adds it as a new column to the DataFrame.
    The weather index is calculated as: 0.4 * humidity + 0.3 * cloud - 0.3 * sunshine
    Args:
        df: A Polars DataFrame with columns 'humidity', 'cloud', and 'sunshine'.
    Returns:
        A new Polars DataFrame with an added 'weather_index' column.
        Returns the original DataFrame if any of the required columns are missing.
    Raises:
        ValueError: If the DataFrame is missing the 'humidity', 'cloud', or 'sunshine' columns.
    """

    required_columns = ['humidity', 'cloud', 'sunshine']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")

    df = df.with_columns(
        (0.4 * pl.col('humidity') + 0.3 * pl.col('cloud') - 0.3 * pl.col('sunshine')).alias('weather_index')
    )

    return df


# prompt: Generate a machine learning program  in python using pytorch and polars that predict the target variable "rainfall" based on the features  ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall'] use ROC as error function. use polars for data manipuation. use sequentional for the NN. use cross validation and early stopping for the best result. Use normalization

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
processed_data = calculate_weather_index (raw_data.fill_null(strategy = "mean"))
X = processed_data.select(features).to_numpy()
y = processed_data["rainfall"].to_numpy()

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



# prompt: now generate code to predict rainfall on the test data

# Prepare test data

processed_test_data = calculate_weather_index (raw_test.fill_null(strategy = "mean"))
X_test = processed_test_data.select(features).to_numpy()
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



# prompt: now add code to mount google drive and write the result there

if COLAB :
    from google.colab import drive
    drive.mount('/content/drive')
    submission_df.write_csv("/content/drive/MyDrive/submission.csv") # Change the path if needed
    print("Submission file created successfully on Google Drive!")


