import numpy as np
import pandas as pd


train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


test_df.isna().sum()


train_df.isna().sum()


test_df['winddirection'].fillna(test_df['winddirection'].mean(), inplace=True)


test_df.rename(columns={'temparature': 'temperature'}, inplace=True)
train_df.rename(columns={'temparature': 'temperature'}, inplace=True)


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


X = train_df.drop('rainfall', axis=1)
y = train_df['rainfall']


X = preprocess_data(X)
test_df_scaled = preprocess_data(test_df)


models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42),
    "LightGBM": lgb.LGBMClassifier(objective='binary', random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


best_model = None
best_auc = 0
for model_name, model in models.items():
    auc_scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]  # Get probability predictions
        auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(auc)
    
    avg_auc = np.mean(auc_scores)
    print(f"{model_name} AUC: {avg_auc:.4f}")
    
    if avg_auc > best_auc:
        best_auc = avg_auc
        best_model = model


best_model.fit(X, y)

test_pred = best_model.predict_proba(test_df_scaled)[:, 1]


submission = pd.DataFrame({
    "id": test_df["id"],
    "rainfall": test_pred
})


submission.to_csv("/kaggle/working/submission.csv", index=False)

print(f"Best AUC achieved: {best_auc:.4f}")




