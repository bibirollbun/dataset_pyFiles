import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


from sklearn.preprocessing import LabelEncoder

# Feature Engineering
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

train_df['Soil Type'] = le_soil.fit_transform(train_df['Soil Type'])
train_df['Crop Type'] = le_crop.fit_transform(train_df['Crop Type'])
train_df['Fertilizer Name'] = le_fert.fit_transform(train_df['Fertilizer Name'])

test_df['Soil Type'] = le_soil.transform(test_df['Soil Type'])
test_df['Crop Type'] = le_crop.transform(test_df['Crop Type'])


# Nutrient ratios and interactions
for df in [train_df, test_df]:
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-6)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-6)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-6)
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
    df['Soil_Crop'] = df['Soil Type'] * df['Crop Type']
    df['N_total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']


# Features and target
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']
X_test = test_df.drop('id', axis=1)


from sklearn.model_selection import train_test_split
# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


!pip install optuna


import optuna
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 70),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
    }
    model = LGBMClassifier(**params, random_state=42, verbose=-1, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)


# Best model
best_params = study.best_params
lgbm = LGBMClassifier(**best_params, random_state=42, verbose=-1, early_stopping_rounds=50)
lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')


# MAP@3 Calculation
def map_at_3(y_true, y_pred, k=3):
    n = len(y_true)
    ap_sum = 0
    for i in range(n):
        relevant = y_true[i]
        pred = y_pred[i][:k]
        precisions = 0
        rel_count = 0
        for j in range(min(k, len(pred))):
            if pred[j] == relevant:
                rel_count += 1
                precisions += rel_count / (j + 1)
                break
        ap_sum += precisions / min(1, rel_count) if rel_count > 0 else 0
    return ap_sum / n


# Validation: Top 3 predictions
y_pred_proba = lgbm.predict_proba(X_val)
top3_preds = np.argsort(-y_pred_proba, axis=1)[:, :3]
top3_labels = le_fert.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
y_val_labels = le_fert.inverse_transform(y_val)
map_score = map_at_3(y_val_labels, top3_labels)
print(f"MAP@3 Score: {map_score:.4f}")


# Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=lgbm.feature_importances_, y=X.columns)
plt.title('Feature Importance')
plt.show()


# Test predictions
test_pred_proba = lgbm.predict_proba(X_test)
top3_test_preds = np.argsort(-test_pred_proba, axis=1)[:, :3]
top3_test_labels = le_fert.inverse_transform(top3_test_preds.ravel()).reshape(top3_test_preds.shape)


# Submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(preds) for preds in top3_test_labels]
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")




