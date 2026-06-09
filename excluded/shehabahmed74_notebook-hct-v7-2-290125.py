!pip install lifelines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from lifelines.utils import concordance_index
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from scipy.stats import uniform, randint


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from scipy.stats import uniform, randint

def load_and_preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print("Train Data Overview:")
    print(train_data.head())
    print("Test Data Overview:")
    print(test_data.head())
    
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_data.select_dtypes(include=["object"]).columns.tolist()
    
    train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].median())
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
    test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].median())
    train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])
    test_data[categorical_cols] = test_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])
    
    key_columns = ["ID", "efs", "efs_time"]
    key_train_data = train_data[key_columns]
    key_test_data = test_data[["ID"]]
    
    train_data = pd.get_dummies(train_data.drop(columns=key_columns), drop_first=True)
    test_data = pd.get_dummies(test_data.drop(columns=["ID"]), drop_first=True)
    
    train_data, test_data = train_data.align(test_data, join="left", axis=1, fill_value=0)
    
    train_data = pd.concat([key_train_data, train_data], axis=1)
    test_data = pd.concat([key_test_data, test_data], axis=1)
    
    train_data = shuffle(train_data, random_state=42)
    return train_data, test_data

def prepare_features_and_targets(train_data, test_data):
    X = train_data.drop(columns=["ID", "efs", "efs_time"])
    y = train_data["efs_time"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_selector = SelectFromModel(LGBMRegressor(n_estimators=100, random_state=42))
    X_selected = feature_selector.fit_transform(X_scaled, y)
    
    X_test = test_data.drop(columns=["ID"])
    X_test_scaled = scaler.transform(X_test)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    return X_selected, y, X_test_selected, test_data["ID"]

def get_hyperparameter_space():
    return {
        'n_estimators': randint(500, 1500),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 12),
        'subsample': uniform(0.6, 1.0),
        'colsample_bytree': uniform(0.6, 1.0),
        'reg_alpha': uniform(1e-4, 1e-1),
        'reg_lambda': uniform(1e-4, 1e-1)
    }

def optimize_hyperparameters(X, y, n_iter=50):
    model = LGBMRegressor(random_state=42)
    param_dist = get_hyperparameter_space()
    random_search = RandomizedSearchCV(
        estimator=model, param_distributions=param_dist,
        n_iter=n_iter, scoring='neg_mean_squared_error', cv=5,
        random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)
    joblib.dump(random_search, "/kaggle/working/random_search.pkl")
    return random_search.best_params_

def train_and_predict(X, y, X_test, best_params):
    model = LGBMRegressor(**best_params, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X_test)
    return predictions

def evaluate_model(X, y, best_params):
    model = LGBMRegressor(**best_params, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores, mse_scores, r2_scores = [], [], []
    best_c_index = float('-inf')
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        
        mae_scores.append(mean_absolute_error(y_val, val_predictions))
        mse_scores.append(mean_squared_error(y_val, val_predictions))
        r2_scores.append(r2_score(y_val, val_predictions))
        best_c_index = max(best_c_index, r2_score(y_val, val_predictions))
    
    print(f'Mean Absolute Error: {np.mean(mae_scores)}')
    print(f'Mean Squared Error: {np.mean(mse_scores)}')
    print(f'R² Score: {np.mean(r2_scores)}')
    print(f'Best C Index Reached: {best_c_index}')

train_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
test_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
train_data, test_data = load_and_preprocess_data(train_path, test_path)
X, y, X_test, test_ids = prepare_features_and_targets(train_data, test_data)
best_params = optimize_hyperparameters(X, y, n_iter=50)
predictions = train_and_predict(X, y, X_test, best_params)
evaluate_model(X, y, best_params)

submission = pd.DataFrame({
    "ID": test_ids.astype(int),
    "prediction": predictions
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file saved at /kaggle/working/submission.csv")
print(submission.head())


