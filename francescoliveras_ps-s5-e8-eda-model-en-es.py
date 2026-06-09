# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


df_train.head(5)


df_test.head(5)


df_train.drop(columns=["id"], axis=1, inplace=True)
df_test.drop(columns=["id"], axis=1, inplace=True)


target_variable="y"


import seaborn as sns
import matplotlib.pyplot as plt
def eda_pipeline(df_train, df_test):
    
    # Display first few rows
    print("\n--- First few rows of train data ---")
    display(df_train.head())
    
    print("\n--- First few rows of test data ---")
    display(df_test.head())
    
    # Dataset info
    print("\n--- Train Data Info ---")
    print(df_train.info())
    
    print("\n--- Test Data Info ---")
    print(df_test.info())
    
    # Missing values
    print("\n--- Missing Values in Train Data ---")
    print(df_train.isnull().sum())
    
    print("\n--- Missing Values in Test Data ---")
    print(df_test.isnull().sum())
    
    print("\n--- Percentage of Missing Values in Train Data ---")
    print((df_train.isnull().sum() / len(df_train)) * 100)
    
    print("\n--- Percentage of Missing Values in Test Data ---")
    print((df_test.isnull().sum() / len(df_test)) * 100)
    
    # Summary statistics
    print("\n--- Train Data Summary Statistics ---")
    print(df_train.describe())
    
    print("\n--- Test Data Summary Statistics ---")
    print(df_test.describe())
    
    # Identify categorical columns
    train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
    test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']
    
    print("\n--- Categorical Columns in Train Data ---")
    print(train_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Train) ---")
    print(df_train[train_cat_columns].nunique())
    
    print("\n--- Categorical Columns in Test Data ---")
    print(test_cat_columns)
    
    print("\n--- Unique Values in Categorical Columns (Test) ---")
    print(df_test[test_cat_columns].nunique())
    
    # Identify numerical columns
    train_num_columns = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64']]
    test_num_columns = [col for col in df_test.columns if df_test[col].dtype in ['int64', 'float64']]
    
    print("\n--- Numerical Columns in Train Data ---")
    print(train_num_columns)
    
    print("\n--- Numerical Columns in Test Data ---")
    print(test_num_columns)
    
    # Check for duplicate rows
    print("\n--- Duplicate Rows in Train Data ---")
    print(df_train.duplicated().sum())
    
    print("\n--- Duplicate Rows in Test Data ---")
    print(df_test.duplicated().sum())
    
    # Correlation matrix (excluding non-numeric columns)
    print("\n--- Correlation Matrix ---")
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_train[train_num_columns].corr(), annot=True, cmap='coolwarm')
    plt.show()
       
    # Correlation with Target Variable
    print("\n--- Correlation with Target Variable ---")
    target_corr = df_train[train_num_columns].corr()[target_variable].sort_values(ascending=False)
    print(target_corr)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=target_corr.index, y=target_corr.values, palette='coolwarm')
    plt.xticks(rotation=90)
    plt.title(f'Feature Correlation with {target_variable}')
    plt.show()   
    
    # Distribution plots for numerical features
    print("\n--- Distribution of Numerical Features ---")
    df_train[train_num_columns].hist(figsize=(12, 10), bins=30)
    plt.show()
    
    # Box plots for outlier detection
    print("\n--- Box Plots for Outlier Detection ---")
    for col in train_num_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df_train[col])
        plt.title(f'Box plot of {col}')
        plt.show()
    
    # Value counts for categorical features
    print("\n--- Value Counts for Categorical Columns ---")
    for col in train_cat_columns:
        print(f"\nValue counts for {col}:")
        print(df_train[col].value_counts())


eda_pipeline(df_train, df_test)


# Target Distribution Check
print("\n--- Distribution of Target Variable for Class Balance Check ---\n")
df_train[target_variable].value_counts(normalize=True).plot(kind='barh')


df_train[target_variable].value_counts()


def data_preprocessing_pipeline_ohe(df_train, df_test, target_column='y'):
    """
    Preprocess data using One-Hot Encoding.
    Ensures same columns in train and test by combining before encoding.
    Excludes target column from test set during encoding.
    """

    # Drop target from test set if accidentally included
    if target_column in df_test.columns:
        df_test = df_test.drop(columns=[target_column])

    # Fill missing values
    # for column in df_train.columns:
    #     if column == target_column:
    #         continue
    #     if df_train[column].dtype == 'object':
    #         df_train[column].fillna(df_train[column].mode()[0], inplace=True)
    #     else:
    #         df_train[column].fillna(df_train[column].mean(), inplace=True)

    # for column in df_test.columns:
    #     if df_test[column].dtype == 'object':
    #         df_test[column].fillna(df_test[column].mode()[0], inplace=True)
    #     else:
    #         df_test[column].fillna(df_test[column].mean(), inplace=True)

    # Identify categorical columns (excluding target)
    cat_cols = df_train.drop(columns=[target_column]).select_dtypes(include=['object', 'category']).columns.tolist()

    # Combine train and test for consistent encoding
    df_train['__is_train__'] = 1
    df_test['__is_train__'] = 0
    combined = pd.concat([df_train, df_test], axis=0)

    # Apply One-Hot Encoding
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=False)

    # Split back
    df_train_encoded = combined[combined['__is_train__'] == 1].drop(columns=['__is_train__'])
    df_test_encoded = combined[combined['__is_train__'] == 0].drop(columns=['__is_train__', target_column], errors='ignore')

    return df_train_encoded, df_test_encoded



# df_train_processed, df_test_processed =data_preprocessing_pipeline_ohe(df_train, df_test, target_column=target_variable)


from sklearn.preprocessing import LabelEncoder

def data_preprocessing_pipeline(df_train, df_test):
   
    label_encoders = {}

    # # Fill missing values for training data
    # for column in df_train.columns:
    #     if df_train[column].dtype == 'object':
    #         mode_value = df_train[column].mode()[0]
    #         df_train[column].fillna(mode_value, inplace=True)
    #     elif df_train[column].dtype in ['int64', 'float64']:
    #         mean_value = df_train[column].mean()
    #         df_train[column].fillna(mean_value, inplace=True)

    # # Fill missing values for test data
    # for column in df_test.columns:
    #     if df_test[column].dtype == 'object':
    #         mode_value = df_test[column].mode()[0]
    #         df_test[column].fillna(mode_value, inplace=True)
    #     elif df_test[column].dtype in ['int64', 'float64']:
    #         mean_value = df_test[column].mean()
    #         df_test[column].fillna(mean_value, inplace=True)

    # Encode categorical features in training set
    for column in df_train.columns:
        if df_train[column].dtype == 'object':
            le = LabelEncoder()
            df_train[column] = le.fit_transform(df_train[column].astype(str))
            label_encoders[column] = le

    # Encode categorical features in test set using train encoders
    for column in df_test.columns:
        if column in label_encoders:
            le = label_encoders[column]
            df_test[column] = df_test[column].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        elif df_test[column].dtype == 'object':
            df_test[column] = -1  # default encoding for unknown categorical column

    return df_train, df_test, label_encoders



df_train,df_test,label_encoders=data_preprocessing_pipeline(df_train, df_test)


df_test.head(5)


df_train.head(5)


from sklearn.preprocessing import StandardScaler

def standardize_data(df_train, df_test):
    """
    Standardize all numerical features using StandardScaler,
    ensuring both train and test have the same columns, while preserving the target variable.
    """
    # Separate target column from train data
    target_values = df_train[target_variable]
    df_train = df_train.drop(columns=[target_variable])
    
    # Ensure both datasets have the same feature columns
    common_columns = df_train.columns.intersection(df_test.columns)
    df_train = df_train[common_columns]
    df_test = df_test[common_columns]
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit on train data and transform both train and test data
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=common_columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=common_columns)
    
    # Reattach the target column to the scaled train data
    df_train_scaled[target_variable] = target_values.reset_index(drop=True)
    
    return df_train_scaled, df_test_scaled


# df_train=df_train_processed
# df_test=df_test_processed


df_train_scaled, df_test_scaled = standardize_data(df_train, df_test)


df_train_scaled.head(5)


df_test_scaled.head(5)


X = df_train.drop(columns=[target_variable])
y = df_train[target_variable]


X_scalled = df_train_scaled.drop(columns=[target_variable])
y_scalled = df_train_scaled[target_variable]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, 
                                                    random_state=42)

X_train_scalled, X_test_scalled, y_train_scalled, y_test_scalled = train_test_split(
    X_scalled, y_scalled, test_size=0.2,stratify=y_scalled,random_state=42)


# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X_resampled, y_resampled = sm.fit_resample(X_train, y_train)


# cat_col=['job', 'marital', 'education', 'default', 'housing', 
#          'loan', 'contact', 'month', 'poutcome']


# import optuna
# from catboost import CatBoostClassifier, Pool
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split

# # Train-validation split
# X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
#     X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# # Define Optuna objective function
# def objective(trial):
#     params = {
#         'iterations': trial.suggest_int('iterations', 100, 1000),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
#         'border_count': trial.suggest_int('border_count', 32, 255),
#         'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'random_seed': 42,
#         'verbose': 0,
#         'task_type': 'CPU',
#         'cat_features': cat_col,
#     }

#     model = CatBoostClassifier(**params)
#     model.fit(X_train_split, y_train_split, eval_set=(X_valid_split, y_valid_split), early_stopping_rounds=50)
    
#     preds = model.predict(X_valid_split)
#     auc = roc_auc_score(y_valid_split, preds)
#     return auc

# # Run Optuna optimization
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=30)

# print("Best trial:")
# print(study.best_trial)


# best_params= {
#     'random_seed': 42,
#     'verbose': 100,
#     'cat_features': cat_col
# }


# Tuned_CatBoost_model = CatBoostClassifier(**best_params)
# Tuned_CatBoost_model.fit(X_train, y_train)
# Tuned_CatBoost_model.fit(X_train, y_train)
# Tuned_cat_pred = cat_model.predict(X_test)
# print("ROC AUC Score catboost is ",roc_auc_score(y_test,Tuned_cat_pred))


from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

cat_model = CatBoostClassifier(n_estimators=1500, verbose=100,random_state=42)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
print("ROC AUC Score catboost is ",roc_auc_score(y_test,cat_pred))

print("F1 Score Catboost is ",f1_score(y_test, cat_pred))



from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

cat_model_scalled_data = CatBoostClassifier(n_estimators=1500, verbose=100,
                                            random_state=42)
cat_model_scalled_data.fit(X_train_scalled, y_train_scalled)
cat_pred_scalled_data = cat_model_scalled_data.predict(X_test_scalled)
print("ROC AUC Score catboost on scalled data is ",
      roc_auc_score(y_test_scalled,cat_pred_scalled_data))

print("F1 Score Catboost on scalled data is is ",f1_score(y_test_scalled, 
                                                          cat_pred_scalled_data))


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_model = XGBClassifier(n_estimators=1500,random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("ROC_AUC Score xgb is ",roc_auc_score(y_test,xgb_pred))
print("F1 Score Xgboost is ",f1_score(y_test, xgb_pred))


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Initialize LGBM for multiclass classification
lgbm_model = LGBMClassifier(
    n_estimators=2500,
    random_state=42,verbose=-1
)

# Train
lgbm_model.fit(X_train, y_train)

# Predict
lgbm_pred = lgbm_model.predict(X_test)

# Accuracy
print("ROC AUC Score LGBM is", roc_auc_score(y_test, lgbm_pred))
print("F1 Score LGBM is ",f1_score(y_test, lgbm_pred))


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# Use 5-fold stratified cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
y_probs = np.zeros(len(df_test))
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=20000,
        learning_rate=0.06,
        num_leaves=100,
        max_depth=15,
        min_child_samples=9,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.78,
        reg_lambda=3.0,
        max_bin=4523,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=500)
        ]
    )

    models.append(model)
    
    # Average predictions across all folds
    y_probs += model.predict_proba(df_test)[:, 1] / n_splits


y_probs.shape, df_test.shape


from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Example classifiers
clf1 = CatBoostClassifier(n_estimators=1500, verbose=100,random_state=32)
clf2 = LGBMClassifier(n_estimators=1500,random_state=22,verbose=-1)
clf3 = XGBClassifier(n_estimators=1500,random_state=52)

# Voting classifier
voting_clf = VotingClassifier(
    estimators=[('cat', clf1), ('lgbm', clf2), ('xgb', clf3)],
    voting='soft'  # change to 'hard' for hard voting
)

# Fit and evaluate
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print("roc auc score VotingClassifier:", roc_auc_score(y_test, y_pred))

print("F1 Score VotingClassifier is ",f1_score(y_test, y_pred))


# import optuna
# from sklearn.ensemble import VotingClassifier
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# import numpy as np


# def objective(trial):
#     # Suggest hyperparameters for each base model
#     catboost_params = {
#         'learning_rate': trial.suggest_float("cat_learning_rate", 0.01, 0.3),
#         'depth': trial.suggest_int("cat_depth", 4, 10),
#         'l2_leaf_reg': trial.suggest_float("cat_l2_leaf_reg", 1, 10),
#         'iterations': trial.suggest_int("cat_n_estimators", 100, 3000),
#         'random_state': 42,
#         'verbose': 0
#     }

#     xgb_params = {
#         'learning_rate': trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
#         'max_depth': trial.suggest_int("xgb_max_depth", 3, 10),
#         'reg_lambda': trial.suggest_float("xgb_reg_lambda", 1, 10),
#         'n_estimators': trial.suggest_int("xgb_n_estimators", 100, 3000),
#         'use_label_encoder': False,
#         'eval_metric': 'logloss',
#         'random_state': 42
#     }

#     lgb_params = {
#         'learning_rate': trial.suggest_float("lgb_learning_rate", 0.01, 0.3),
#         'max_depth': trial.suggest_int("lgb_max_depth", 3, 10),
#         'reg_lambda': trial.suggest_float("lgb_reg_lambda", 1, 10),
#         'n_estimators': trial.suggest_int("lgb_n_estimators", 100, 3000),
#         'random_state': 42,
#         'verbosity': -1
#     }

#     # Suggest weights for each classifier in the ensemble
#     cat_weight = trial.suggest_float("cat_weight", 0.5, 3.0)
#     xgb_weight = trial.suggest_float("xgb_weight", 0.5, 3.0)
#     lgb_weight = trial.suggest_float("lgb_weight", 0.5, 3.0)

#     # Initialize base models
#     clf1 = CatBoostClassifier(**catboost_params)
#     clf2 = LGBMClassifier(**lgb_params)
#     clf3 = XGBClassifier(**xgb_params)

#     # VotingClassifier with soft voting
#     voting_clf = VotingClassifier(
#         estimators=[('cat', clf1), ('lgbm', clf2), ('xgb', clf3)],
#         voting='soft',
#         weights=[cat_weight, lgb_weight, xgb_weight],
#         n_jobs=-1
#     )

#     # Fit on training fold
#     voting_clf.fit(X_train, y_train)

#     # Predict on validation fold
#     y_pred = voting_clf.predict(X_test)

#     # ROC AUC on validation set
#     return roc_auc_score(y_test, y_pred)

# # Run Optuna study
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)  # Increase n_trials for better tuning

# # Best results
# print("âœ… Best AUC:", study.best_value)
# print("ğŸ“Œ Best parameters:")
# for k, v in study.best_params.items():
#     print(f"  {k}: {v}")



# best = study.best_params

# # Build models with best hyperparameters
# final_cat = CatBoostClassifier(
#     learning_rate=best["cat_learning_rate"],
#     depth=best["cat_depth"],
#     l2_leaf_reg=best["cat_l2_leaf_reg"],
#     iterations=best["cat_n_estimators"],
#     random_state=42,
#     verbose=0
# )

# final_xgb = XGBClassifier(
#     learning_rate=best["xgb_learning_rate"],
#     max_depth=best["xgb_max_depth"],
#     reg_lambda=best["xgb_reg_lambda"],
#     n_estimators=best["xgb_n_estimators"],
#     use_label_encoder=False,
#     eval_metric='logloss',
#     random_state=42
# )

# final_lgb = LGBMClassifier(
#     learning_rate=best["lgb_learning_rate"],
#     max_depth=best["lgb_max_depth"],
#     reg_lambda=best["lgb_reg_lambda"],
#     n_estimators=best["lgb_n_estimators"],
#     random_state=42,
#     verbosity=-1
# )

# # Final Voting Classifier
# final_voting_clf = VotingClassifier(
#     estimators=[('cat', final_cat), ('lgbm', final_lgb), ('xgb', final_xgb)],
#     voting='soft',
#     weights=[best["cat_weight"], best["lgb_weight"], best["xgb_weight"]],
#     n_jobs=-1
# )

# final_voting_clf.fit(X_train, y_train)

# # Final evaluation
# y_test_pred = final_voting_clf.predict(X_test)

# from sklearn.metrics import f1_score, roc_auc_score
# print("ğŸ”� Final ROC AUC:", roc_auc_score(y_test, y_test_pred))
# print("ğŸ�¯ Final F1 Score:", f1_score(y_test, y_test_pred))


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



from sklearn.metrics import f1_score

F1_Score = f1_score(y_test, y_pred)
print(F1_Score)


final_prediction=voting_clf.predict_proba(df_test)


final_prediction


final_prediction[:, 1]


sample_submission.head(5)


# sample_submission['y'] = final_prediction[:, 1]
# sample_submission.to_csv('submission.csv', index=False)
# print('Submission file saved.')


y_probs


sample_submission['y'] = y_probs
sample_submission.to_csv('submission.csv', index=False)
print('Submission file saved.')


sample_submission.head(5)




