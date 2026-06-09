import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV

# Feature Engineering Function
def feature_engineering(df):
    df['day'] = pd.to_datetime(df['day'], errors='coerce')
    
    # Temporal Features
    df['month'] = df['day'].dt.month
    df['day_of_week'] = df['day'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Temperature Features
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['avg_temp'] = (df['maxtemp'] + df['mintemp']) / 2
    df['temp_deviation'] = df['temparature'] - df['avg_temp']
    
    # Dew Point Depression
    df['dew_point_depression'] = df['temparature'] - df['dewpoint']
    
    # Wind Features
    df['wind_dir_rad'] = np.deg2rad(df['winddirection'])
    df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
    df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
    df.drop(columns=['wind_dir_rad'], inplace=True)
    
    # Wind Chill Factor (Simplified)
    df['wind_chill'] = 13.12 + 0.6215 * df['temparature'] - 11.37 * (df['windspeed']**0.16) + 0.3965 * df['temparature'] * (df['windspeed']**0.16)
    
    # Interaction Features
    df['humidity_temp'] = df['humidity'] * df['temparature']
    df['cloud_sunshine'] = df['cloud'] * df['sunshine']
    df['pressure_temp_interaction'] = df['pressure'] * df['avg_temp']
    df['windspeed_temp_interaction'] = df['windspeed'] * df['avg_temp']
    
    # Rolling Window Features
    rolling_features = ['avg_temp', 'windspeed', 'humidity']
    for feature in rolling_features:
        df[f'rolling_{feature}_mean'] = df[feature].rolling(window=7, min_periods=1).mean()
    
    # Lag Features
    for col in ['avg_temp', 'humidity', 'windspeed', 'pressure']:
        df[f'{col}_lag_1'] = df[col].shift(1)
        df[f'{col}_diff_1'] = df[col].diff(1)

    # Binary Encoding for Season
    df['season'] = df['month'].apply(lambda x: 'Spring' if 3 <= x <= 5 else
                                      'Summer' if 6 <= x <= 8 else
                                      'Autumn' if 9 <= x <= 11 else 'Winter')
    df = pd.get_dummies(df, columns=['season'], drop_first=True)

    df.drop(columns=['day'], inplace=True)
    
    return df

# Load Datasets
df1 = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
df2 = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df1.columns = df1.columns.str.strip()

# Convert rainfall to binary values
df1['rainfall'] = df1['rainfall'].str.lower().map({'yes': 1, 'no': 0})
df2.drop(columns=['id'], inplace=True)

# Reorder Columns
column_order = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']
df1 = df1[column_order]
df2 = df2[column_order]

# Combine Data
train = pd.concat([df1, df2], axis=0, ignore_index=True)
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

imputer = SimpleImputer(strategy="mean")

# Drop 'day' before imputation (if it exists)
train.drop(columns=['day'], inplace=True, errors='ignore')
test.drop(columns=['day'], inplace=True, errors='ignore')
train.drop(columns=['id'], inplace=True, errors='ignore')
test_ids = test['id'].copy() if 'id' in test.columns else None
test.drop(columns=['id'], inplace=True, errors='ignore')
# Ensure columns match before transformation
feature_columns = [col for col in train.columns if col != 'rainfall']  # Exclude target variable

train[feature_columns] = imputer.fit_transform(train[feature_columns])
test[feature_columns] = imputer.transform(test[feature_columns])

# Separate Features and Target
X = train.drop(columns=['rainfall'])
y = train['rainfall']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)

# Feature Selection using ExtraTreesClassifier
selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
selector.fit(X, y)
feature_importances = selector.feature_importances_
important_features = np.array(X)[0].shape[0] > 10

# Hyperparameter Tuning using RandomizedSearchCV
param_grid = {
    "n_estimators": [500, 1000, 1500, 2000],
    "max_depth": [None, 100, 110, 120],
    "min_samples_split": [13, 18, 22],
    "min_samples_leaf": [1, 2, 4]
}
clf = RandomizedSearchCV(ExtraTreesClassifier(), param_distributions=param_grid, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
clf.fit(X, y)
best_model = clf.best_estimator_

# Improved Cross-Validation Function
def purged_cross_validation(X, y, n_splits=5, purge_length=1):
    ts_split = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in ts_split.split(X):
        if len(val_idx) > purge_length:
            val_idx = val_idx[purge_length:]
        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]

# Model Training with Purged Cross-Validation
start_time = time.time()
auc_scores = []

for X_train, X_val, y_train, y_val in purged_cross_validation(X, y):
    best_model.fit(X_train, y_train)
    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    fpr, tpr, _ = roc_curve(y_val, val_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")
    auc_scores.append(roc_auc)

print(f"\nAverage AUC Score: {np.mean(auc_scores):.4f}")
print(f"Training Time: {time.time() - start_time:.2f} seconds")

# Final Prediction on Test Data
best_model.fit(X, y)
test_proba = best_model.predict_proba(test)[:, 1]

# Create Submission
submission = pd.DataFrame({'id': test_ids, 'target': test_proba})
submission.to_csv('/kaggle/working/sub.csv', index=False)
print("Submission file saved.")


