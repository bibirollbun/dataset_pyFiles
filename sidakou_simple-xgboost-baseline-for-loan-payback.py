# Load libraries and check dataset paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import shap
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load training and test datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
predict = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


# Visualize correlation between numerical features
corr = train.select_dtypes(['number']).corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)


# Create new features from grade_subgrade and remove unnecessary columns
def create_features(df):
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)
    
    return df

train = create_features(train)
predict = create_features(predict)


def delete_features(df):
    df = df.drop(columns=(['grade_subgrade','gender','marital_status']))   
    return df
    
train = delete_features(train)
predict = delete_features(predict)


# One-Hot Encoding
def one_hot_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=object_cols, drop_first=False)
    return df

train = one_hot_encode(train)
predict = one_hot_encode(predict)

missing_cols = set(train.columns) - set(predict.columns)
for col in missing_cols:
    predict[col] = 0

predict = predict[train.columns]
predict = predict.drop(columns=['loan_paid_back'])


# Convert boolean columns to integersã€€
def bool_to_int(df):
    bool_columns = df.select_dtypes(include='bool').columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    return df

train = bool_to_int(train)
predict = bool_to_int(predict)


# Add target mean encoding and count encoding features efficiently
# This version avoids DataFrame fragmentation by concatenating columns at once

def add_target_count_features(train, predict, target_col, n_splits=10):
    BASE = [c for c in train.columns if c not in [target_col]]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mean_features = pd.DataFrame(index=train.index)
    count_features = pd.DataFrame(index=train.index)
    mean_features_pred = pd.DataFrame(index=predict.index)
    count_features_pred = pd.DataFrame(index=predict.index)

    for col in BASE:
        if train[col].isnull().all():
            continue

        # === Mean Encoding with K-Fold (leakage prevention) ===
        mean_encoded = np.zeros(len(train))
        for tr_idx, val_idx in kf.split(train):
            tr_fold = train.iloc[tr_idx]
            val_fold = train.iloc[val_idx]
            mean_map = tr_fold.groupby(col)[target_col].mean()
            mean_encoded[val_idx] = val_fold[col].map(mean_map)

        mean_features[f'mean_{col}'] = mean_encoded

        # Apply global mean mapping to prediction data
        global_mean = train.groupby(col)[target_col].mean()
        mean_features_pred[f'mean_{col}'] = predict[col].map(global_mean)

        # === Count Encoding ===
        count_map = train[col].value_counts().to_dict()
        count_features[f'count_{col}'] = train[col].map(count_map)
        count_features_pred[f'count_{col}'] = predict[col].map(count_map)

    # === Concatenate all features at once to avoid fragmentation ===
    train = pd.concat([train, mean_features, count_features], axis=1)
    predict = pd.concat([predict, mean_features_pred, count_features_pred], axis=1)

    # Defragment DataFrames for better performance
    train = train.copy()
    predict = predict.copy()

    print(f"{len(mean_features.columns) + len(count_features.columns)} features created!")
    return train, predict


train, predict = add_target_count_features(train, predict, target_col='loan_paid_back')


# Split dataset into training and testing sets
def split_data(df,test_size=0.2,random_state=42):
    X = df.drop(columns=['id','loan_paid_back'])
    y = df['loan_paid_back']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(train)

predict_X = predict.copy()
predict_X = predict_X.drop(columns=['id'])


# Build and train an XGBoost model
def build_xgboost_model(n_estimators=1000,max_depth=5,learning_rate=0.1,random_state=67):
    
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="auc"
    )
    return model


xgb_model = build_xgboost_model()

xgb_model.fit(X_train, y_train)


# Explain XGBoost model using SHAP values
def explain_model(model):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    shap.plots.waterfall(shap_values[0])

    shap.plots.beeswarm(shap_values)

explain_model(xgb_model)


# Evaluate model performance using common classification metrics
def evaluate_metrics(y_true, y_pred_proba):
    results = []

    y_pred = (y_pred_proba >= 0.5).astype(int)

    def calculate_metrics(y_true, y_pred, y_pred_proba):
        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)
        auc       = roc_auc_score(y_true, y_pred_proba)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
        }

    results.append(calculate_metrics(y_true, y_pred, y_pred_proba))
    return pd.DataFrame(results)


# Evaluate XGBoost model predictionsã€€
train_predict_probs = xgb_model.predict_proba(X_test)[:, 1]

threshold = 0.5
train_predict_binary = (train_predict_probs >= threshold).astype(int)

results = evaluate_metrics(y_test, train_predict_probs)
display(results)


# Plot confusion matrix for model predictions
def plot_confusion(y_true, y_pred, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.show()

plot_confusion(y_test, train_predict_binary, normalize=True, title='Normalized Confusion Matrix')


# Generate submission file from CatBoost predictions
predict_y_probs = xgb_model.predict_proba(predict_X)[:, 1]
predict_df = pd.DataFrame(predict_y_probs, columns=['loan_paid_back'])
submission = pd.concat([predict['id'], predict_df], axis=1)

display(submission.head())
print(submission.isnull().sum())


submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

