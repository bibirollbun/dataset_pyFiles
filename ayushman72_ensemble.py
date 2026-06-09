import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import optuna
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data_with_scaling_encoding_dtypes():
    # Define dtypes for train_transaction
    dtypes_train_transaction = {
        'TransactionID': 'int32',
        'isFraud': 'int8',
        'TransactionDT': 'int32',
        'TransactionAmt': 'float32',
        'ProductCD': 'object',
        'card1': 'int16',
        'card2': 'float32',
        'card3': 'float32',
        'card4': 'object',
        'card5': 'float32',
        'card6': 'object',
        'addr1': 'float32',
        'addr2': 'float32',
        'dist1': 'float32',
        'dist2': 'float32',
        'P_emaildomain': 'object',
        'R_emaildomain': 'object',
        # Add all C, D, M, V features
        **{f'C{i}': 'float32' for i in range(1, 15)},
        **{f'D{i}': 'float32' for i in range(1, 16)},
        **{f'M{i}': 'object' for i in range(1, 10)},
        **{f'V{i}': 'float32' for i in range(1, 340)}
    }

    # Define dtypes for train_identity
    dtypes_train_identity = {
        **{f'id_{i:02}': 'float32' for i in range(1, 12)},
        'id_12': 'object',
        'id_13': 'float32',
        'id_14': 'float32',
        'id_15': 'object',
        'id_16': 'object',
        **{f'id_{i:02}': 'float32' for i in range(17, 40)},
        **{f'id_{i:02}': 'object' for i in range(23, 39)},
        'DeviceType': 'object',
        'DeviceInfo': 'object'
    }

    # Define dtypes for test datasets
    dtypes_test_transaction = dtypes_train_transaction.copy()
    del dtypes_test_transaction['isFraud']  # Test data does not contain 'isFraud'

    dtypes_test_identity = dtypes_train_identity.copy()

    # Read in datasets with dtypes
    train_transaction = pd.read_csv(
        '/kaggle/input/ieee-fraud-detection/train_transaction.csv', 
        dtype=dtypes_train_transaction
    )
    train_identity = pd.read_csv(
        '/kaggle/input/ieee-fraud-detection/train_identity.csv', 
        dtype=dtypes_train_identity
    )
    test_transaction = pd.read_csv(
        '/kaggle/input/ieee-fraud-detection/test_transaction.csv', 
        dtype=dtypes_test_transaction
    )
    test_identity = pd.read_csv(
        '/kaggle/input/ieee-fraud-detection/test_identity.csv', 
        dtype=dtypes_test_identity
    )

    # Standardize column names
    test_identity.columns = test_identity.columns.str.replace('-', '_')
    test_transaction.columns = test_transaction.columns.str.replace('-', '_')

    # Merge datasets
    train = train_transaction.merge(train_identity, how='left', on='TransactionID')
    test = test_transaction.merge(test_identity, how='left', on='TransactionID')

    # Free up memory
    del train_transaction, train_identity, test_transaction, test_identity
    gc.collect()

    # Handle missing values
    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)

    # Define categorical features
    categorical_features = [
        'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'id_33', 'id_34', 'DeviceType', 'DeviceInfo'
    ]
    categorical_features += [f'id_{i}' for i in range(12, 39)]

    # Encode categorical features
    for col in categorical_features:
        if col in train.columns:
            train[col] = train[col].astype(str)
            test[col] = test[col].astype(str)

            le = LabelEncoder()
            combined_data = pd.concat([train[col], test[col]], axis=0)
            le.fit(combined_data)
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

    return train, test


# Load the train and test datasets using the provided function
train, test = load_data_with_scaling_encoding_dtypes()

# Prepare features and target as done previously
X = train.drop(columns=['isFraud', 'TransactionID'])
y = train['isFraud']
X_test = test.drop(columns=['TransactionID'], errors='ignore')

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def load_model_and_params(model_name, model_path, params_path):
    model = joblib.load(model_path)
    with open(params_path, 'r') as f:
        params = json.load(f)
    return model, params


optimized_lgb_model, lgb_best_params = load_model_and_params(
    'lgb', 
    '/kaggle/input/lgbm-opt/optimized_lgb_model.pkl', 
    '/kaggle/input/lgbm-opt/lgb_best_params.json'
)


optimized_xgb_model, xgb_best_params = load_model_and_params(
    'xgb', 
    '/kaggle/input/xgboost-opt/optimized_xgb_model.pkl', 
    '/kaggle/input/xgboost-opt/xgb_best_params.json'
)


optimized_cat_model, cat_best_params = load_model_and_params(
    'cat', 
    '/kaggle/input/catboost-opt/optimized_cat_model.pkl', 
    '/kaggle/input/catboost-opt/cat_best_params.json'
)


lgb_val_pred = optimized_lgb_model.predict(X_val, num_iteration=optimized_lgb_model.best_iteration)
xgb_val_pred = optimized_xgb_model.predict_proba(X_val)[:, 1]
cat_val_pred = optimized_cat_model.predict_proba(X_val)[:, 1]

lgb_test_pred = optimized_lgb_model.predict(X_test, num_iteration=optimized_lgb_model.best_iteration)
xgb_test_pred = optimized_xgb_model.predict_proba(X_test)[:, 1]
cat_test_pred = optimized_cat_model.predict_proba(X_test)[:, 1]


def get_report(name,preds,thresh=0.5):
    print(name,"-------","Confusion Matrix",sep = "\n\n")
    preds_bin = (preds>thresh)
    print(confusion_matrix(y_val,preds_bin))
    print("-------","Classification Report",sep = "\n\n")
    print(classification_report(y_val,preds_bin))
    print("-------","Auc",sep = "\n\n")
    print(roc_auc_score(y_val,preds))
    print("-------","ROC",sep ="\n\n")
    fpr, tpr, _ = roc_curve(y_val, preds)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Optimized {name}')
    plt.legend(loc="lower right")
    plt.show()



get_report("LGBm",lgb_val_pred)


get_report("XGBoost",xgb_val_pred)


get_report("Cat Boost",cat_val_pred)


def objective(trial):
    # Suggest weights for each model
    weight_lgb = trial.suggest_float('weight_lgb', 0, 1)
    weight_xgb = trial.suggest_float('weight_xgb', 0, 1)
    weight_cat = trial.suggest_float('weight_cat', 0, 1)
    
    # Normalize weights
    total_weight = weight_lgb + weight_xgb + weight_cat
    weight_lgb /= total_weight
    weight_xgb /= total_weight
    weight_cat /= total_weight
    
    # Compute weighted ensemble predictions
    ensemble_val_pred = (
        weight_lgb * lgb_val_pred +
        weight_xgb * xgb_val_pred +
        weight_cat * cat_val_pred
    )
    
    # Calculate AUC
    auc = roc_auc_score(y_val, ensemble_val_pred)
    return auc


study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=8888)


best_weights = study.best_params
total_weight = sum(best_weights.values())
best_weights = {k: v / total_weight for k, v in best_weights.items()}
print(f"Optimal weights found: {best_weights}")


ensemble_val_pred = (
    best_weights['weight_lgb'] * lgb_val_pred +
    best_weights['weight_xgb'] * xgb_val_pred +
    best_weights['weight_cat'] * cat_val_pred
)


get_report("Ensemble",ensemble_val_pred)


ensemble_test_pred = (
    best_weights['weight_lgb'] * lgb_test_pred +
    best_weights['weight_xgb'] * xgb_test_pred +
    best_weights['weight_cat'] * cat_test_pred
)


submission = pd.DataFrame({
    'TransactionID': test['TransactionID'],
    'isFraud': ensemble_test_pred
})


submission.to_csv('submission.csv', index=False)




