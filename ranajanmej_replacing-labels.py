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


import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df['Personality'].value_counts()


train_df.iloc[[1873,16567]]


train_df['Social_event_attendance'].min()


categorical_cols = ['Stage_fear','Drained_after_socializing']


train_df = pd.get_dummies(train_df,columns = categorical_cols,drop_first = True)


train_df.head()


train_df['Stage_fear_Yes'] = train_df['Stage_fear_Yes'].astype(int)
train_df['Drained_after_socializing_Yes'] = train_df['Drained_after_socializing_Yes'].astype(int)


test_df = pd.get_dummies(test_df,columns = categorical_cols,drop_first = True)
test_df['Stage_fear_Yes'] = test_df['Stage_fear_Yes'].astype(int)
test_df['Drained_after_socializing_Yes'] = test_df['Drained_after_socializing_Yes'].astype(int)


test_df.head()


train_df[(train_df['Time_spent_Alone'] == 11) & (train_df['Social_event_attendance'] == 0)].isnull().sum()


from sklearn.experimental import enable_iterative_imputer  # Enable experimental feature
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import numpy as np


X =train_df.drop(columns = 'Personality',axis =1)
y = train_df['Personality']


# Create the Iterative Imputer
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)

# Fit and transform the data
train_df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test_df_imputed = pd.DataFrame(imputer.transform(test_df),columns = test_df.columns)


# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=3)
# # Fit and transform the data
# train_df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# test_df_imputed = pd.DataFrame(imputer.transform(test_df),columns = test_df.columns)


train_df_imputed.isnull().sum()


test_df_imputed.isnull().sum()


train_df_imputed.head()


columns_num = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size'
              ,'Post_frequency']



for cols in columns_num:
    print(f"{cols} --> {train_df[cols].unique()}")


for cols in columns_num:
    print(f"{cols} --> {train_df_imputed[cols].unique()}")
    train_df_imputed[cols] = train_df_imputed[cols].astype(int)
    test_df_imputed[cols] = test_df_imputed[cols].astype(int)


for cols in columns_num:
    print(f"{cols} --> {train_df_imputed[cols].unique()}")
    print(f"{cols} --> {test_df_imputed[cols].unique()}")


ind = train_df_imputed[(train_df_imputed['Time_spent_Alone'] == 11) & (train_df_imputed['Social_event_attendance'] == 0) &
                 (train_df_imputed['Going_outside'] == 0) & (train_df_imputed['Friends_circle_size'] == 0)].index



train_df.iloc[ind]


y.iloc[13225] = y.iloc[13225].replace('Extrovert','Introvert')


y.iloc[13225]


ind3 = train_df_imputed[(train_df_imputed['Time_spent_Alone'] >= 10) & (train_df_imputed['Social_event_attendance'] == 0) &
                 (train_df_imputed['Going_outside'] <= 1) & (train_df_imputed['Friends_circle_size'] == 0)].index



train_df_imputed.iloc[1041]


y.iloc[ind3][y.iloc[ind3] == 'Extrovert']


y.iloc[1041] = y.iloc[1041].replace('Extrovert','Introvert')


y.iloc[1041]


ind4 = train_df_imputed[(train_df_imputed['Time_spent_Alone'] >= 10) & (train_df_imputed['Social_event_attendance'] == 0) &
                 (train_df_imputed['Going_outside'] <= 1) & (train_df_imputed['Friends_circle_size'] <= 1)].index



y.iloc[ind4][y.iloc[ind4] == 'Extrovert']


train_df_imputed.iloc[[10374,12907]]


y.iloc[[10374,12907]] = y.iloc[[10374,12907]].replace('Extrovert','Introvert')


ind2 = train_df_imputed[(train_df_imputed['Time_spent_Alone'] <= 1) & (train_df_imputed['Social_event_attendance'] >= 9) &
                (train_df_imputed['Going_outside'] >= 6) & (train_df_imputed['Friends_circle_size'] >= 10) & (train_df_imputed['Post_frequency'] >= 9)].index


y.iloc[ind2][y.iloc[ind2] == 'Introvert']


y.iloc[[1873,16567]] = y.iloc[[1873,16567]].replace('Introvert','Extrovert')


y.iloc[[1873,16567]]


y.value_counts()


train_df['Personality'].value_counts()


train_df_imputed.head()


y.head()


cols_to_scale = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size',
                 'Post_frequency']
                


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
train_df_imputed[cols_to_scale] = scaler.fit_transform(train_df_imputed[cols_to_scale])
test_df_imputed[cols_to_scale] = scaler.transform(test_df_imputed[cols_to_scale])


train_df_imputed.head()


test_df_imputed.head()


y_encoded = y.replace({'Extrovert': 1,'Introvert' :0})


y_encoded


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train_df_imputed,y_encoded,test_size = 0.2,random_state = 42)


X_train.head()


# import optuna
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score

# # Optuna objective function using X_train and y_train
# def objective(trial):
#     model_name = trial.suggest_categorical('model', ['logistic_regression', 'svm', 'decision_tree', 'random_forest', 'xgboost'])
    
#     if model_name == 'logistic_regression':
#         C = trial.suggest_float('logreg_C', 1e-3, 100.0, log=True)
#         model = LogisticRegression(C=C, max_iter=1000)
    
#     elif model_name == 'svm':
#         C = trial.suggest_float('svm_C', 1e-3, 100.0, log=True)
#         kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf'])
#         model = SVC(C=C, kernel=kernel, probability=True)
    
#     elif model_name == 'decision_tree':
#         max_depth = trial.suggest_int('dt_max_depth', 2, 32)
#         min_samples_split = trial.suggest_int('dt_min_samples_split', 2, 10)
#         model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    
#     elif model_name == 'random_forest':
#         n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
#         max_depth = trial.suggest_int('rf_max_depth', 2, 32)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
#     elif model_name == 'xgboost':
#         n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
#         max_depth = trial.suggest_int('xgb_max_depth', 2, 32)
#         learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
#         model = XGBClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             use_label_encoder=False,
#             eval_metric='logloss'
#         )

#     # 5-fold cross-validation
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1).mean()
    
#     return score

# # Run Optuna study
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# # Best model and parameters
# print("Best Trial:", study.best_trial)

# best_params = study.best_trial.params
# model_name = best_params.pop('model')

# # Build best model
# if model_name == 'logistic_regression':
#     model = LogisticRegression(C=best_params['logreg_C'], max_iter=1000)
# elif model_name == 'svm':
#     model = SVC(C=best_params['svm_C'], kernel=best_params['svm_kernel'], probability=True)
# elif model_name == 'decision_tree':
#     model = DecisionTreeClassifier(
#         max_depth=best_params['dt_max_depth'],
#         min_samples_split=best_params['dt_min_samples_split']
#     )
# elif model_name == 'random_forest':
#     model = RandomForestClassifier(
#         n_estimators=best_params['rf_n_estimators'],
#         max_depth=best_params['rf_max_depth']
#     )
# elif model_name == 'xgboost':
#     model = XGBClassifier(
#         n_estimators=best_params['xgb_n_estimators'],
#         max_depth=best_params['xgb_max_depth'],
#         learning_rate=best_params['xgb_learning_rate'],
#         use_label_encoder=False,
#         eval_metric='logloss'
#     )

# # Fit best model on entire training data
# model.fit(X_train, y_train)

# # Optional: Evaluate on X_test (internal validation)
# test_preds_local = model.predict(X_test)
# print(f"Validation Accuracy: {accuracy_score(y_test, test_preds_local):.4f}")

# # ðŸ”® Make prediction on Kaggle test data (test_df_imputed)
# kaggle_preds = model.predict(test_df_imputed)


!pip install xgboost


!pip install lightgbm



!pip install optuna


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score
# import optuna

# # ----------------------------
# # STEP 1: Split the data
# # ----------------------------

# # X = train_new.drop(columns='Personality', axis=1)
# # y = train_new['Personality']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# # Optional: scale the data if needed
# # scaler = MinMaxScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)

# # ----------------------------
# # STEP 2: Define objective
# # ----------------------------

# def objective(trial):
#     classifier_name = trial.suggest_categorical(
#         "classifier", ["LogisticRegression", "SVC", "RandomForest", "XGBoost", "LightGBM", "DecisionTree"]
#     )

#     if classifier_name == "LogisticRegression":
#         C = trial.suggest_float("lr_C", 1e-4, 10.0, log=True)
#         model = LogisticRegression(C=C, solver='liblinear')

#     elif classifier_name == "SVC":
#         C = trial.suggest_float("svc_C", 1e-4, 10.0, log=True)
#         kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf", "poly"])
#         model = SVC(C=C, kernel=kernel)

#     elif classifier_name == "RandomForest":
#         n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
#         max_depth = trial.suggest_int("rf_max_depth", 2, 20)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

#     elif classifier_name == "XGBoost":
#         n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
#         learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
#         max_depth = trial.suggest_int("xgb_max_depth", 2, 10)
#         model = XGBClassifier(n_estimators=n_estimators,
#                               learning_rate=learning_rate,
#                               max_depth=max_depth,
#                               use_label_encoder=False,
#                               eval_metric='logloss')

#     elif classifier_name == "LightGBM":
#         n_estimators = trial.suggest_int("lgb_n_estimators", 50, 300)
#         learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.3)
#         num_leaves = trial.suggest_int("lgb_num_leaves", 15, 150)
#         max_depth = trial.suggest_int("lgb_max_depth", 3, 15)
#         model = LGBMClassifier(n_estimators=n_estimators,
#                                learning_rate=learning_rate,
#                                num_leaves=num_leaves,
#                                max_depth=max_depth,
#                               verbose=-1)

#     else:  # DecisionTree
#         max_depth = trial.suggest_int("dt_max_depth", 2, 20)
#         min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 10)
#         model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

#     # Cross-validation
#     cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#     scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
#     return scores.mean()

# # ----------------------------
# # STEP 3: Run Optuna
# # ----------------------------

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# # ----------------------------
# # STEP 4: Best results
# # ----------------------------

# print("Best trial:")
# trial = study.best_trial

# print(f"  Accuracy: {trial.value}")
# print("  Best hyperparameters:")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

# # ----------------------------
# # STEP 5: Train best model
# # ----------------------------

# params = trial.params
# clf_type = params['classifier']

# if clf_type == "LogisticRegression":
#     best_model = LogisticRegression(C=params['lr_C'], solver='liblinear')

# elif clf_type == "SVC":
#     best_model = SVC(C=params['svc_C'], kernel=params['svc_kernel'])

# elif clf_type == "RandomForest":
#     best_model = RandomForestClassifier(
#         n_estimators=params['rf_n_estimators'],
#         max_depth=params['rf_max_depth']
#     )

# elif clf_type == "XGBoost":
#     best_model = XGBClassifier(
#         n_estimators=params['xgb_n_estimators'],
#         learning_rate=params['xgb_learning_rate'],
#         max_depth=params['xgb_max_depth'],
#         use_label_encoder=False,
#         eval_metric='logloss'
#     )

# elif clf_type == "LightGBM":
#     best_model = LGBMClassifier(
#         n_estimators=params['lgb_n_estimators'],
#         learning_rate=params['lgb_learning_rate'],
#         num_leaves=params['lgb_num_leaves'],
#         max_depth=params['lgb_max_depth'],
#         verbose=-1
#     )

# else:  # DecisionTree
#     best_model = DecisionTreeClassifier(
#         max_depth=params['dt_max_depth'],
#         min_samples_split=params['dt_min_samples_split']
#     )

# # ----------------------------
# # STEP 6: Evaluate on test set
# # ----------------------------

# best_model.fit(X_train, y_train)
# y_pred = best_model.predict(X_test)

# final_accuracy = accuracy_score(y_test, y_pred)
# print(f"Final Test Accuracy: {final_accuracy:.4f}")



from xgboost import XGBClassifier


m_xgb = XGBClassifier(
    n_estimators = 163,
    learning_rate = 0.13993199123471584,
    max_depth = 9
)


m_xgb.fit(X_train,y_train)


y_pred = m_xgb.predict(X_test)


from sklearn.metrics import accuracy_score


print(accuracy_score(y_test,y_pred))


y_fin_pred = m_xgb.predict(test_df_imputed)


submission = pd.DataFrame({
    'id': test_df_imputed['id'].astype(int),  # or use another column like df_test['Id'] if available
    'Personality': y_fin_pred        # replace 'target' with actual target column name if needed
})


submission.head()


submission['Personality'] = submission['Personality'].replace({1: 'Extrovert', 0: 'Introvert'})


submission.head()


# # âœ… Save submission file (modify according to Kaggle submission format)
# submission = pd.DataFrame({
#     'Id': test_df_imputed['id'].astype(int),  # or use another column like df_test['Id'] if available
#     'Personality': kaggle_preds        # replace 'target' with actual target column name if needed
# })
submission.to_csv('submission1.csv', index=False)
print("Submission file saved as submission.csv")


submission['Personality'].value_counts()




