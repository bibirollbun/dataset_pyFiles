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


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def fetch_datasets():
    train_set = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
    test_set = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
    return train_set, test_set


# Function to transform weather data
def transform_weather_data(dataset):
    dataset = dataset.copy()
    dataset["dew_humidity"] = dataset["dewpoint"] * dataset["humidity"]
    dataset["cloud_windspeed"] = dataset["cloud"] * dataset["windspeed"]
    dataset["temp_to_sunshine"] = dataset["sunshine"] / (dataset["temparature"] + 1)
    dataset['wind_temp_interaction'] = dataset['windspeed'] * dataset['temparature']
    dataset['cloud_sun_ratio'] = dataset['cloud'] / (dataset['sunshine'] + 1)
    dataset['humidity_sunshine_interaction'] = dataset['humidity'] * dataset['sunshine']
    
    dataset['month'] = ((dataset['day'] - 1) // 30 + 1).clip(upper=12)
    dataset['season'] = dataset['month'].apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 0)
    dataset['season_cloud_trend'] = dataset['cloud'] * dataset['season']
    
    return dataset.drop(columns=["maxtemp", "winddirection", "humidity", "temparature", "pressure", "day", "season", "month"])


train_dataset, test_dataset = fetch_datasets()
train_dataset = transform_weather_data(train_dataset)
test_dataset = transform_weather_data(test_dataset)


# Prepare features and target
features = train_dataset.drop(['rainfall', 'id'], axis=1)
target = train_dataset['rainfall']
test_features = test_dataset.drop(['id'], axis=1)


# Standardize features
feature_scaler = StandardScaler()
features = feature_scaler.fit_transform(features)
test_features = feature_scaler.transform(test_features)


model_collection = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(10,)),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=6),
    "CatBoost": CatBoostClassifier(random_state=42, iterations=100, learning_rate=0.14, depth=6, verbose=0)
}


# Evaluate models using stratified k-fold cross-validation
NUM_FOLDS = 13
kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
model_performance = {}
roc_data = {}

for model_name, model_instance in model_collection.items():
    predictions = np.zeros(len(target))
    
    for train_indices, val_indices in kfold.split(features, target):
        X_train, X_val = features[train_indices], features[val_indices]
        y_train, y_val = target[train_indices], target[val_indices]

        model_instance.fit(X_train, y_train)
        predictions[val_indices] = model_instance.predict_proba(X_val)[:, 1]
    
    auc_value = roc_auc_score(target, predictions)
    model_performance[model_name] = auc_value
    false_positive_rate, true_positive_rate, _ = roc_curve(target, predictions)
    roc_data[model_name] = (false_positive_rate, true_positive_rate)


# Display model performance
for model_name, auc_value in model_performance.items():
    print(f"{model_name}: AUC = {auc_value:.4f}")

# Select the best-performing model
top_model_name = max(model_performance, key=model_performance.get)
top_model = model_collection[top_model_name]

print(f"\nTop Performing Model: {top_model_name}")


# Retrain the best model on the entire dataset
top_model.fit(features, target)

# Generate predictions for the test set
test_predictions = top_model.predict_proba(test_features)[:, 1]


# Create submission file
submission_file = pd.DataFrame({'id': test_dataset['id'], 'rainfall': test_predictions})

# Save submission file
submission_file.to_csv("submission.csv", index=False)
print("\nSubmission file 'weather_prediction_submission.csv' has been created.")




