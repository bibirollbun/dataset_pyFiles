import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# 1. Data Loading & Mapping

df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv", index_col='id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df_train_extra=pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")

df_train_extra.columns = df_train_extra.columns.str.replace(' ', '')
df_train_extra = df_train_extra[df_train_extra.columns].copy()
df_train_extra['rainfall'] = df_train_extra['rainfall'].map({'no': 0, 'yes': 1})
df_train_extra['humidity'] = df_train_extra['humidity'].astype(float)
df_train_extra['cloud'] = df_train_extra['cloud'].astype(float)
df_train_features = list(df_train)
df_train_extra = df_train_extra[df_train_features]

df_train = pd.concat([df_train, df_train_extra], axis=0, ignore_index=True)

display(df_train.head())
display(df_train.tail())
display(df_test.head())

df_train = df_train.drop_duplicates()
df_train.shape


# 2. Feature Engineering:
df_train['pressure_lag1'] = df_train['pressure'].shift(1)
df_train.fillna(method='bfill', inplace=True)
df_test['pressure_lag1'] = df_test['pressure'].shift(1)
df_test.fillna(method='bfill', inplace=True)

def handle_outliers_iqr(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[feature] = np.clip(df[feature], lower_bound, upper_bound)
    return df

features_to_handle = ['windspeed', 'sunshine', 'cloud']
df_train = handle_outliers_iqr(df_train, features_to_handle)
df_test = handle_outliers_iqr(df_test, features_to_handle)




# 3. Separate features and target
X = df_train[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
             'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
             'windspeed', 'pressure_lag1']]  # Include engineered feature
y = df_train['rainfall']




# 4. Split data into training and validation sets (using StratifiedKFold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# 5. Hyperparameter Optimization with Optuna
def objective(trial, X, y, cv):
    # Define hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 10, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    class_weight = trial.suggest_categorical('class_weight', ['balanced', None])

    # Initialize and train model
    model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       class_weight=class_weight,
                                       random_state=42)

    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Handle Class Imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model.fit(X_train_resampled, y_train_resampled)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)

    return np.mean(auc_scores)  # Return average AUC across folds

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X, y, cv), n_trials=100)

best_params = study.best_params
print(f"Best hyperparameters: {best_params}")




# 6. Train the final model with the best hyperparameters and all data
best_model = RandomForestClassifier(**best_params, random_state=42)

# Apply SMOTE to the entire training data for final model training
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

best_model.fit(X_resampled, y_resampled)



# 7. Prediction on Test Data
X_test_final = df_test[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
                        'dewpoint', 'humidity', 'cloud', 'sunshine',
                        'winddirection', 'windspeed', 'pressure_lag1']]
test_predictions = best_model.predict_proba(X_test_final)[:, 1]



df_test.columns


# 8. Create Submission File
submission_df = pd.DataFrame({'id': df_test['id'], 'rainfall': test_predictions})
submission_df.to_csv('submission.csv', index=False)



display(submission_df)




