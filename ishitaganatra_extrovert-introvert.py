import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
df_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


print("Train data shape : " , df_train.shape)
print("Test data shape : " , df_test.shape)


print("Train data head : \n" , df_train.head())
print("\nTest data head : \n" , df_test.head())


print("Train data info: \n")
print(df_train.info())
print("\nTest data info: \n")
print(df_test.info())


print("Train data: \n" , df_train.describe())
print("\nTest data: \n" , df_test.describe())


print("Train data: \n" , df_train.isnull().sum())
print("Test data: \n" , df_test.isnull().sum())


print("Duplicate rows in training set:", df_train.duplicated().sum())
print("Duplicate rows in testing set:", df_test.duplicated().sum())


print(df_train['Drained_after_socializing'].value_counts())
print(df_test['Drained_after_socializing'].value_counts())


print(df_train['Stage_fear'].value_counts())
print(df_test['Stage_fear'].value_counts())


outliers = []
for feature in df_train.select_dtypes(include=np.number).columns:
    Q1 = df_train[feature].quantile(0.25)
    Q3 = df_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    if df_train[(df_train[feature] < lower_bound) | (df_train[feature] > upper_bound)].any(axis=None):
        outliers.append(feature)
        df_train[feature] = df_train[feature].clip(lower=lower_bound, upper=upper_bound)
print("Attributes with outliers:", outliers)


outliers = []
for feature in df_test.select_dtypes(include=np.number).columns:
    Q1 = df_test[feature].quantile(0.25)
    Q3 = df_test[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    if df_test[(df_test[feature] < lower_bound) | (df_test[feature] > upper_bound)].any(axis=None):
        outliers.append(feature)
        df_test[feature] = df_test[feature].clip(lower=lower_bound, upper=upper_bound)
print("Attributes with outliers:", outliers)


numeric_cols = df_train.select_dtypes(include='number').columns
for col in numeric_cols:
    df_train[col].fillna(df_train[col].median(), inplace=True)


numeric_cols = df_test.select_dtypes(include='number').columns
for col in numeric_cols:
    df_test[col].fillna(df_test[col].median(), inplace=True)


categorical_cols = df_train.select_dtypes(include='object').columns
for col in categorical_cols:
    df_train[col].fillna(df_train[col].mode()[0], inplace=True)


categorical_cols = df_test.select_dtypes(include='object').columns
for col in categorical_cols:
    df_test[col].fillna(df_test[col].mode()[0], inplace=True)


df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})


df_train['Stage_fear'] = df_train['Stage_fear'].replace({'Yes': 1, 'No': 0})
df_test['Stage_fear'] = df_test['Stage_fear'].replace({'Yes': 1, 'No': 0})


df_train['Personality'] = df_train['Personality'].replace({'Extrovert': 1, 'Introvert': 0})


df_train["social_conflict"] = (df_train["Drained_after_socializing"] * df_train["Post_frequency"])
df_test["social_conflict"] = (df_test["Drained_after_socializing"] * df_test["Post_frequency"])


df_train["balanced_social_life"] = (df_train["Friends_circle_size"] * df_train["Social_event_attendance"])
df_test["balanced_social_life"] = (df_test["Friends_circle_size"] * df_test["Social_event_attendance"])


df_train["Social_boldness_score"] = (df_train["Social_event_attendance"] + df_train["Going_outside"] - df_train["Stage_fear"])
df_test["Social_boldness_score"] = (df_test["Social_event_attendance"] + df_test["Going_outside"] - df_test["Stage_fear"])


print(df_train['Personality'].value_counts())


X = df_train.drop(['id', 'Personality'], axis=1)
y = df_train['Personality']
X_test = df_test.drop(['id'], axis=1)


scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds_proba = np.zeros(len(X_test_scaled))
val_thresholds = []
fold = 1

for train_idx, val_idx in skf.split(X_scaled, y):
    print(f"\nTraining fold {fold}")
    X_tr, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=6000,
        learning_rate=0.013760092845998047,
        depth=5,
        l2_leaf_reg=4.728920684023544,
        random_strength=1.1784281467800652,
        bagging_temperature=0.24901940249126786,
        border_count=79,
        class_weights=[7.326837128363753, 7.197240945457829],
        loss_function="Logloss",
        eval_metric="Accuracy",
        verbose=0,
        task_type="GPU",
        random_state=42 + fold  
    )
    
    model.fit(X_tr, y_tr)

    val_probs = model.predict_proba(X_val_fold)[:, 1]
    
    best_acc = 0
    best_thresh = 0.5
    for t in np.arange(0.3, 0.71, 0.01):
        preds = (val_probs > t).astype(int)
        acc = accuracy_score(y_val_fold, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    print(f"Best threshold for fold {fold}: {best_thresh:.2f} with accuracy: {best_acc:.4f}")
    val_thresholds.append(best_thresh)
    
    test_preds_proba += model.predict_proba(X_test_scaled)[:, 1]
    
    fold += 1

test_preds_proba /= n_splits

final_threshold = np.mean(val_thresholds)
print(f"\nFinal threshold used on test set: {final_threshold:.2f}")

final_preds = (test_preds_proba > final_threshold).astype(int)

submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': final_preds
})
submission['Personality'] = submission['Personality'].replace({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv("submission.csv", index=False)
print("submission.csv created with threshold-tuned KFold predictions.")

