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


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


print(train.shape)
print(test.shape)


test.info()


train.head()


train.info()


train.day.unique()


train.rainfall.unique()


import matplotlib.pyplot as plt
import seaborn as sns

# Group by day and calculate average rainfall
rainfall_trend = train.groupby("day")["rainfall"].mean()

# Plot the trend
plt.figure(figsize=(12, 6))
sns.lineplot(x=rainfall_trend.index, y=rainfall_trend.values, marker="o", color="b")
plt.title("Average Rainfall Across the Year")
plt.xlabel("Day of the Year (1-365)")
plt.ylabel("Average Rainfall Probability")
plt.grid(True)
plt.show()




# Approximate mapping (assuming no leap year)
def get_month(day):
    if day <= 31: return "Jan"
    elif day <= 59: return "Feb"
    elif day <= 90: return "Mar"
    elif day <= 120: return "Apr"
    elif day <= 151: return "May"
    elif day <= 181: return "Jun"
    elif day <= 212: return "Jul"
    elif day <= 243: return "Aug"
    elif day <= 273: return "Sep"
    elif day <= 304: return "Oct"
    elif day <= 334: return "Nov"
    else: return "Dec"

# Create a new column for month
train["month"] = train["day"].apply(get_month)

# Convert to categorical type for correct ordering
train["month"] = pd.Categorical(train["month"], 
                             categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 
                             ordered=True)





# Aggregate rainfall by month
monthly_rainfall = train.groupby("month")["rainfall"].mean()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_rainfall.index, y=monthly_rainfall.values, color="blue")
plt.title("Average Rainfall Probability Per Month")
plt.xlabel("Month")
plt.ylabel("Average Rainfall Probability")
plt.grid(axis="y")
plt.show()



train.head()


train.drop(columns=["month"], inplace=True)


train.shape


train.winddirection.unique()


test_ids = test["id"].copy()
def preprocess_data(df, is_train=True):
    """
    Preprocesses the given DataFrame by encoding cyclical features,
    generating rolling averages (for training), and handling missing values.

    Parameters:
    df (pd.DataFrame): The input dataset.
    is_train (bool): Whether this is the training dataset.

    Returns:
    pd.DataFrame: The processed dataset.
    """
    df = df.copy()  # Avoid modifying the original DataFrame

    # Encode cyclical day and wind direction
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365)
    df['wind_sin'] = np.sin(2 * np.pi * df['winddirection'] / 360)
    df['wind_cos'] = np.cos(2 * np.pi * df['winddirection'] / 360)

    # Rolling average and lag features (only for training data)
    if is_train:
        df['rainfall_7d_avg'] = df['rainfall'].rolling(window=7).mean()
        df['rainfall_lag_1'] = df['rainfall'].shift(1)
        df['rainfall_lag_7'] = df['rainfall'].shift(7)

    df['humidity_7d_avg'] = df['humidity'].rolling(window=7).mean()

    # Fill missing values
    df.fillna(method="bfill", inplace=True)

    # Drop unnecessary columns
    drop_cols = ["day", "winddirection"]  # Always drop these
    if 'id' in df.columns:
        drop_cols.append('id')  # Drop 'id' only for training
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df

# Apply preprocessing
train = preprocess_data(train, is_train=True)
test = preprocess_data(test, is_train=False)






# Compute correlation matrix
corr_matrix = train.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



train.drop(columns=['tempreture','dewpoint'], inplace=True, errors="ignore")


from sklearn.model_selection import train_test_split


X = train.drop(columns=['rainfall'])  
y = train['rainfall']  


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
print(f"Target distribution in Train: {y_train.value_counts(normalize=True)}")
print(f"Target distribution in Validation: {y_val.value_counts(normalize=True)}")



!pip install catboost optuna

import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split Data
X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=["rainfall"]), train["rainfall"], test_size=0.2, random_state=42)

# Define the Objective Function for Tuning
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_loguniform("random_strength", 1e-3, 10),
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.1, 10),
        "verbose": 0
    }

    model = CatBoostClassifier(**params, loss_function="Logloss", eval_metric="AUC", random_seed=42)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=0)

    y_val_pred = model.predict_proba(X_val)[:, 1]  # Extract probability for class 1
    return roc_auc_score(y_val, y_val_pred)

# Run the Tuning Process
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Get Best Parameters
best_params = study.best_params
print("Best Parameters:", best_params)




# Train Final CatBoost Model with Best Parameters
best_catboost = CatBoostClassifier(**best_params, loss_function="Logloss", eval_metric="AUC", random_seed=42)
best_catboost.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=100)

# Evaluate Performance
y_val_pred = best_catboost.predict_proba(X_val)[:, 1]  # Probability for class 1
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"CatBoost Validation ROC-AUC: {roc_auc}")





# Ensure test has the same columns as train
missing_cols = set(X_train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0  # Fill missing columns in test with 0

# Ensure same column order
test = test[X_train.columns]

print(f"Train shape: {X_train.shape}, Test shape: {test.shape}")


# Make Predictions on Test Set
y_test_pred = best_catboost.predict_proba(test.drop(columns=["id"], errors="ignore"))[:, 1]




# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,  # Ensure this matches the test set IDs
    'rainfall': y_test_pred  # Predicted probabilities
})

# Save file in Kaggle's working directory
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ Submission file saved! Download it from Kaggle's Files tab.")





