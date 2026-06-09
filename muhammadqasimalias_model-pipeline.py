import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
df.head()


df.describe()


df.info()


df.isna().sum()


print(df['Exited'].value_counts(normalize=True))  # Check target imbalance


sns.countplot(df,x="Gender")
plt.figure(figsize=(10, 6))
plt.show()


sns.countplot(df,x="Geography")


sns.boxplot(data=df, x='Geography', y='CreditScore')
plt.title("Boxplot of CreditScore by Geography")
plt.xticks(rotation=45)
plt.show()


sns.boxplot(data=df, x='Geography', y='EstimatedSalary')
plt.title("Boxplot of EstimatedSalary by Geography")
plt.xticks(rotation=45)
plt.show()


numerical_cols = ['Age', 'NumOfProducts','Tenure']

for col in numerical_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
for col in num_cols:
    sns.boxplot(x='Exited', y=col, data=df)
    plt.title(f"{col} distribution by Exited")
    plt.show()



corr = df.corr(numeric_only=True)
plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


df.corr(numeric_only=True)['Exited'].sort_values(ascending=False)



from sklearn.model_selection import train_test_split

X = df.drop(['Exited', 'id', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']

X = pd.get_dummies(X, drop_first=True)  # encode categoricals

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score, recall_score, roc_auc_score, classification_report, confusion_matrix

ratio = df['Exited'].value_counts()[0] / df['Exited'].value_counts()[1]


models = {
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=ratio, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(class_weight='balanced'),
    'CatBoost': CatBoostClassifier(auto_class_weights='Balanced', verbose=0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")
    print(f"Confusion Report:{classification_report(y_test,y_pred)}")



import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")


df['Surname'].value_counts()


# Age buckets
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 100], labels=['Young', 'Adult', 'Senior', 'Old'])

# Tenure group
df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1, 2, 5, 10], labels=['New', 'Mid', 'Loyal'])

# Balance to salary ratio
df['BalanceToSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1e-5)

# Age to tenure ratio
df['AgeTenureRatio'] = df['Tenure'] / (df['Age'] + 1e-5)

# High value customer: high balance and high credit score
df['HighValueCustomer'] = ((df['Balance'] > df['Balance'].median()) & (df['CreditScore'] > df['CreditScore'].median())).astype(int)

# Is senior citizen
df['IsSeniorCitizen'] = (df['Age'] > 60).astype(int)

# Is unengaged customer
df['IsUnengagedCustomer'] = ((df['IsActiveMember'] == 0) & (df['NumOfProducts'] <= 1)).astype(int)

# Has balance
df['HasBalance'] = (df['Balance'] > 0).astype(int)

# Frequency Encoding for Surname
surname_freq = df['Surname'].value_counts().to_dict()
df['SurnameFreq'] = df['Surname'].map(surname_freq)

# Frequency Encoding for CustomerId (if reused or patterned — rare)
custid_freq = df['CustomerId'].value_counts().to_dict()
df['CustomerIdFreq'] = df['CustomerId'].map(custid_freq)



df = df.drop(columns=['id'], errors='ignore')



df.head()


# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)



df.head()


df.info()


num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
                'BalanceToSalaryRatio', 'AgeTenureRatio','SurnameFreq','CustomerIdFreq']

scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])



# Separate target and features
X = df.drop(columns=['Exited'])
y = df['Exited']



len(df)


y.value_counts()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)



print(len(X_train))
print(len(X_test))


import lightgbm as lgb
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Split your data (you likely already have X_train, X_test, etc.)
X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Compute class imbalance ratio
scale_pos_weight = len(y_train_split[y_train_split == 0]) / len(y_train_split[y_train_split == 1])

# Define Optuna objective
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight": scale_pos_weight
    }

    model = lgb.LGBMClassifier(**params)

    # Use callback API for early stopping
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(stopping_rounds=30)]
    )

    y_pred = model.predict(X_valid)
    return f1_score(y_valid, y_pred)

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Use best params
print("Best trial:")
print(f"  F1 Score: {study.best_value}")
print(f"  Best Params: {study.best_params}")

best_params = study.best_params
best_params["objective"] = "binary"
best_params["boosting_type"] = "gbdt"
best_params["random_state"] = 42
best_params["scale_pos_weight"] = scale_pos_weight

# Train final model on full training set with best params
lgbm = lgb.LGBMClassifier(**best_params)
lgbm.fit(X_train, y_train)

# Feature importance
feature_importance = pd.Series(lgbm.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6), title='Feature Importances')
plt.tight_layout()
plt.show()

# Evaluation
y_pred = lgbm.predict(X_test)
print(classification_report(y_test, y_pred))



y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Plot the tradeoff
import matplotlib.pyplot as plt
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall vs Threshold")
plt.show()



y_pred = lgbm.predict(X_test)
new_threshold = 0.4
y_pred_thresh = (y_pred >= new_threshold).astype(int)
print(classification_report(y_test, y_pred_thresh))


df_c = pd.concat([X, y], axis=1)
df_c.head()


import pandas as pd
from sklearn.utils import resample
from collections import Counter

# Convert X and y to DataFrame (if not already)
df = pd.concat([X, y], axis=1)

# Split majority and minority
df_majority = df[df['Exited'] == 0]
df_minority = df[df['Exited'] == 1]

print("Before balancing:", Counter(df['Exited']))

# Choose medium target count (e.g., halfway between majority and minority)
target_samples = int((len(df_majority) + len(df_minority)) / 2)

# Upsample minority to target
df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=target_samples,
    random_state=42
)

# Downsample majority to same target
df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=target_samples,
    random_state=42
)

# Combine
df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
print("After balancing:", Counter(df_balanced['Exited']))

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Final X and y
X_balanced = df_balanced.drop(columns='Exited')
y_balanced = df_balanced['Exited']



X_balanced.head()


import lightgbm as lgb
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, classification_report

# Step 1: Split out 20% for testing (stratified)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_balanced, y_balanced,
    test_size=0.2,
    stratify=y_balanced,
    random_state=42
)

# Step 2: Optuna Objective with CV on 80% training data
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, valid_idx in skf.split(X_train_full, y_train_full):
        X_train_cv, X_valid_cv = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
        y_train_cv, y_valid_cv = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params)

        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_valid_cv, y_valid_cv)],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )

        y_pred = model.predict(X_valid_cv)
        f1_scores.append(f1_score(y_valid_cv, y_pred))

    return np.mean(f1_scores)

# Step 3: Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial:")
print(f"  F1 Score: {study.best_value}")
print(f"  Best Params: {study.best_params}")

# Step 4: Train final model on full 80% training data
best_params = study.best_params
best_params["objective"] = "binary"
best_params["boosting_type"] = "gbdt"
best_params["random_state"] = 42

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(X_train_full, y_train_full)

# Optional Step 5: Evaluate on held-out test set
y_pred_test = final_model.predict(X_test)
print("\nEvaluation on held-out 20% test set:")
print(classification_report(y_test, y_pred_test))



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load test data
df_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')



# --- Use training set statistics ---
balance_median = df['Balance'].median()
credit_median = df['CreditScore'].median()

# Age buckets
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=[18, 30, 45, 60, 100], labels=['Young', 'Adult', 'Senior', 'Old'])

# Tenure group
df_test['TenureGroup'] = pd.cut(df_test['Tenure'], bins=[-1, 2, 5, 10], labels=['New', 'Mid', 'Loyal'])

# Balance to salary ratio
df_test['BalanceToSalaryRatio'] = df_test['Balance'] / (df_test['EstimatedSalary'] + 1e-5)

# Age to tenure ratio
df_test['AgeTenureRatio'] = df_test['Tenure'] / (df_test['Age'] + 1e-5)

# High value customer
df_test['HighValueCustomer'] = ((df_test['Balance'] > balance_median) &
                                (df_test['CreditScore'] > credit_median)).astype(int)

# Is senior citizen
df_test['IsSeniorCitizen'] = (df_test['Age'] > 60).astype(int)

# Is unengaged customer
df_test['IsUnengagedCustomer'] = ((df_test['IsActiveMember'] == 0) &
                                  (df_test['NumOfProducts'] <= 1)).astype(int)

# Has balance
df_test['HasBalance'] = (df_test['Balance'] > 0).astype(int)

df_train = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')

# Step 2: Create frequency maps from df_train
surname_freq_map = df_train['Surname'].value_counts().to_dict()
customerid_freq_map = df_train['CustomerId'].value_counts().to_dict()

df_test['SurnameFreq'] = df_test['Surname'].map(surname_freq_map).fillna(1)
df_test['CustomerIdFreq'] = df_test['CustomerId'].map(customerid_freq_map).fillna(1)


# Drop unused columns
df_test.drop(['id','CustomerId', 'Surname'], axis=1, inplace=True)

# One-hot encode categorical columns
df_test = pd.get_dummies(df_test, drop_first=True)

# Standardize numerical features using the same scaler from training
num_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',
                'BalanceToSalaryRatio', 'AgeTenureRatio','SurnameFreq','CustomerIdFreq']

df_test[num_features] = scaler.transform(df_test[num_features])



X_balanced.head()


df_test.head()


# Add any missing columns in test set
missing_cols = set(df.columns) - set(df_test.columns)
missing_cols.discard('Exited')  # Don't include target

for col in missing_cols:
    df_test[col] = 0

# Reorder to match training columns (except target)
df_test = df_test[df.drop(columns='Exited').columns]



y_pred_Test = final_model.predict(df_test)


# Step 1: Get predictions
y_test_preds = final_model.predict(df_test)

# Step 2: Reload test file to access 'id' column
original_test = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')

# Step 3: Create submission DataFrame
submission = pd.DataFrame({
    'id': original_test['id'],
    'Exited': y_test_preds
})

# Step 4: Save to CSV
submission.to_csv('submission.csv', index=False)

print("✅ submission.csv created!")



df_test.info()


Y_test = model.predict(X_test)




