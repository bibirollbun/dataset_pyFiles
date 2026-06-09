import numpy as np
import pandas as pd


df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


df_train.shape, df_test.shape


df_train.head()


df_train.isnull().sum()


to_drop_columns = [
    'id',
    'day'
]


for column in df_train.columns:
    if column in to_drop_columns:
        continue
    print(df_train.groupby('rainfall')[column].mean())
    print()


df_train['Humiditymore'] = (df_train['humidity'] > 80).astype(int)
df_test['Humiditymore'] = (df_test['humidity'] > 80).astype(int)
df_train['Cloudmore'] = (df_train['cloud'] > 65).astype(int)
df_test['Cloudmore'] = (df_test['cloud'] > 65).astype(int)
df_train['Sunshineless'] = (df_train['sunshine'] < 5).astype(int)
df_test['Sunshineless'] = (df_test['sunshine'] < 5).astype(int)
df_train['windspeedmore'] = (df_train['windspeed'] > 20).astype(int)
df_test['windspeedmore'] = (df_test['windspeed'] > 20).astype(int)


df_train = df_train.drop(columns = to_drop_columns)
df_test = df_test.drop(columns = to_drop_columns)


df_train.head()


df_test.head()


from sklearn.preprocessing import StandardScaler


X = df_train.drop(columns=['rainfall'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
from tqdm import tqdm


y = df_train['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


param_grid = {
    'max_depth': [3, 6, 9],  # Tree depth
    'learning_rate': [0.01, 0.1, 0.3],  # Learning rate
    'n_estimators': [100, 300, 500],  # Number of boosting rounds
    'subsample': [0.7, 0.9],  # Fraction of samples used
    'colsample_bytree': [0.7, 1],  # Fraction of features used per tree
    'gamma': [0, 1, 5]  # Minimum loss reduction required for split
}

param_combinations = list(itertools.product(*param_grid.values()))


# Track best parameters and highest accuracy
best_params = None
best_accuracy = 0


# Loop through all hyperparameter combinations
for params in tqdm(param_combinations, desc="Processing Evaluation"):
    current_params = dict(zip(param_grid.keys(), params))

    # Train model with current parameters
    model = XGBClassifier(**current_params, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Update best parameters if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = current_params


# Print the best parameters and accuracy
print("\nBest Parameters:", best_params)
print("Best Accuracy:", best_accuracy)


model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')


model.fit(X_scaled, y)


y_pred = model.predict(X_test)


accuracy_score(y_test, y_pred)


df_test_scaled = scaler.transform(df_test)


y_result = model.predict(df_test_scaled)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


result = pd.DataFrame({
    'id': df_test['id'],  
    'rainfall': y_result  
})


result.head()


result.to_csv("submission.csv", index=False)




