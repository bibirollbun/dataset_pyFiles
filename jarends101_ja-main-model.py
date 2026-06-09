# Import Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


# Load data
train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


# Encode selected categorical columns
categorical_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']
for col in categorical_cols:
    for df in [train, test]:
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))


# Feature selection based on missing data
features = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
features = [col for col in features if train[col].isnull().mean() < 0.4]
features = [col for col in features if col not in ['isFraud', 'TransactionID']]
features = list(set(features + categorical_cols))


# Impute and scale features
for df in [train, test]:
    for col in features:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('unknown')

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

X = train[features]
y = train['isFraud']


# Prepare Cross-validation 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_iterations = []


#Create for loop to go through cross-validation testing 

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.01,
        n_estimators=4000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        tree_method='gpu_hist',
        use_label_encoder=False,
        early_stopping_rounds=100,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    print(f"Best iteration: {model.best_iteration}")
    best_iterations.append(model.best_iteration)


# find the average best iteration
avg_best_iter = int(np.mean(best_iterations))
print(f"\nAverage best iteration: {avg_best_iter}")



# Train final model on full data
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.01,
    n_estimators=avg_best_iter,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method='gpu_hist',
    use_label_encoder=False,
    verbosity=1
)

final_model.fit(X, y)


# Make prediction and Confusion Matrix
final_preds_proba = final_model.predict_proba(X)[:, 1]
final_preds = (final_preds_proba > 0.5).astype(int)

cm = confusion_matrix(y, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Fraud", "Fraud"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Out-of-Fold Predictions")
plt.show()



# Create test predictions as well as submission file and save the submission file 
test_preds = final_model.predict_proba(test[features])[:, 1]

submission = pd.DataFrame({
    'TransactionID': test['TransactionID'].astype(int),
    'isFraud': test_preds
})
submission.to_csv('submission_xgboost.csv', index=False)

