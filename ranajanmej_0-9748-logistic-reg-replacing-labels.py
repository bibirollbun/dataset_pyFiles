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
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head()


train_df.isnull().sum()


train_df['Time_spent_Alone'].unique()


train_df['Social_event_attendance'].unique()


train_df['Friends_circle_size'].unique()


train_df['Post_frequency'].unique()


ind1 = train_df[(train_df['Time_spent_Alone'] <= 1) & (train_df['Social_event_attendance'] >=9) & 
        (train_df['Friends_circle_size'] >=13) & (train_df['Post_frequency'] > 5) & (train_df['Personality'] == 'Introvert')].index


print(ind1)


train_df.loc[ind1, 'Personality'] = 'Extrovert'


train_df.iloc[ind1]


train_df['Time_spent_Alone'].unique()


train_df['Social_event_attendance'].unique()


train_df['Friends_circle_size'].unique()


train_df['Post_frequency'].unique()


ind2 = train_df[(train_df['Time_spent_Alone'] >= 10) & (train_df['Social_event_attendance'] <= 1) & 
        (train_df['Friends_circle_size'] <= 1) & (train_df['Post_frequency'] <= 1) & (train_df['Personality'] =='Extrovert')].index


train_df.loc[ind2,['Personality']] = 'Introvert'


train_df.iloc[ind2]


train_df.head()


train_df['Stage_fear'] = train_df['Stage_fear'].replace({'No' : 0,'Yes':1})
train_df['Drained_after_socializing'] = train_df['Drained_after_socializing'].replace({'No' :0,'Yes':1})


test_df['Stage_fear'] = test_df['Stage_fear'].replace({'No' : 0,'Yes':1})
test_df['Drained_after_socializing'] = test_df['Drained_after_socializing'].replace({'No' :0,'Yes':1})


X = train_df.drop(columns = 'Personality',axis = 1)
y = train_df['Personality']


from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


# Initialize IterativeImputer with RandomForest (optional, default is BayesianRidge)
imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)

# Fit and transform
train_df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test_df_imputed = pd.DataFrame(imputer.transform(test_df),columns = test_df.columns)


# from sklearn.impute import KNNImputer
# import pandas as pd

# # Initialize the KNN Imputer (you can change n_neighbors as needed)
# imputer = KNNImputer(n_neighbors=5)

# # Fit and transform the training data
# train_df_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# # Transform the test data
# test_df_imputed = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)


train_df_imputed.head()


train_df_imputed = train_df_imputed.drop(columns = 'id',axis = 1)
test_df_imputed = test_df_imputed.drop(columns = 'id',axis = 1)


train_df_imputed.head()


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


cols_to_scale = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']


train_df_imputed.columns


test_df_imputed.columns


scaler.fit_transform(train_df_imputed)


train_df_imputed[cols_to_scale] = scaler.fit_transform(train_df_imputed[cols_to_scale])
train_df_imputed.head()


test_df_imputed[cols_to_scale] = scaler.transform(test_df_imputed[cols_to_scale])


train_df_imputed.isnull().sum()


y_enc = y.replace({'Extrovert':1,'Introvert':0})


# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train_df_imputed,y_enc,test_size = 0.25,random_state = 42)


# import optuna
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# from sklearn.metrics import accuracy_score
# import numpy as np


# # Use Stratified K-Fold CV
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# def objective(trial):
#     classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "SVC", "DecisionTree", "RandomForest", "XGBoost"])

#     if classifier_name == "LogisticRegression":
#         C = trial.suggest_loguniform("lr_C", 1e-4, 10)
#         penalty = trial.suggest_categorical("lr_penalty", ["l2"])
#         model = make_pipeline(StandardScaler(), 
#                               LogisticRegression(C=C, penalty=penalty, solver="lbfgs", max_iter=1000))

#     elif classifier_name == "SVC":
#         C = trial.suggest_loguniform("svc_C", 1e-4, 10)
#         kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf"])
#         gamma = trial.suggest_categorical("svc_gamma", ["scale", "auto"])
#         model = make_pipeline(StandardScaler(), 
#                               SVC(C=C, kernel=kernel, gamma=gamma))

#     elif classifier_name == "DecisionTree":
#         max_depth = trial.suggest_int("dt_max_depth", 1, 20)
#         min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 20)
#         model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

#     elif classifier_name == "RandomForest":
#         n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
#         max_depth = trial.suggest_int("rf_max_depth", 3, 20)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

#     elif classifier_name == "XGBoost":
#         n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
#         max_depth = trial.suggest_int("xgb_max_depth", 3, 15)
#         learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
#         model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
#                               learning_rate=learning_rate, use_label_encoder=False,
#                               eval_metric='logloss', verbosity=0)

#     # Use cross-validation to get average accuracy
#     score = cross_val_score(model, train_df_imputed, y_enc, cv=cv, scoring="accuracy").mean()
#     return score

# # Run Optuna study
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)

# # Show best result
# print("Best Model:", study.best_trial.params["classifier"])
# print("Best Score (Accuracy):", study.best_value)
# print("Best Hyperparameters:")
# for key, value in study.best_trial.params.items():
#     print(f"  {key}: {value}")


# import warnings
# warnings.filterwarnings("ignore")

# import optuna
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier

# # Cross-validation strategy
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# def objective(trial):
#     classifier_name = trial.suggest_categorical("classifier", [
#         "LogisticRegression", "SVC", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"
#     ])

#     if classifier_name == "LogisticRegression":
#         C = trial.suggest_loguniform("lr_C", 1e-4, 10)
#         penalty = "l2"
#         model = make_pipeline(StandardScaler(),
#                               LogisticRegression(C=C, penalty=penalty, solver="lbfgs", max_iter=1000))

#     elif classifier_name == "SVC":
#         C = trial.suggest_loguniform("svc_C", 1e-4, 10)
#         kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf"])
#         gamma = trial.suggest_categorical("svc_gamma", ["scale", "auto"])
#         model = make_pipeline(StandardScaler(),
#                               SVC(C=C, kernel=kernel, gamma=gamma))

#     elif classifier_name == "DecisionTree":
#         max_depth = trial.suggest_int("dt_max_depth", 1, 20)
#         min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 20)
#         model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

#     elif classifier_name == "RandomForest":
#         n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
#         max_depth = trial.suggest_int("rf_max_depth", 3, 20)
#         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

#     elif classifier_name == "XGBoost":
#         n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
#         max_depth = trial.suggest_int("xgb_max_depth", 3, 15)
#         learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
#         model = XGBClassifier(n_estimators=n_estimators,
#                               max_depth=max_depth,
#                               learning_rate=learning_rate,
#                               use_label_encoder=False,
#                               eval_metric='logloss',
#                               verbosity=0)

#     elif classifier_name == "LightGBM":
#         n_estimators = trial.suggest_int("lgb_n_estimators", 50, 300)
#         max_depth = trial.suggest_int("lgb_max_depth", 3, 15)
#         learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.3)
#         num_leaves = trial.suggest_int("lgb_num_leaves", 20, 150)
#         model = LGBMClassifier(n_estimators=n_estimators,
#                                max_depth=max_depth,
#                                learning_rate=learning_rate,
#                                num_leaves=num_leaves,
#                                verbosity=-1)

#     score = cross_val_score(model, train_df_imputed, y_enc, cv=cv, scoring="accuracy").mean()
#     return score

# # Run Optuna
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=75)

# # Print best trial
# print("Best Model:", study.best_trial.params["classifier"])
# print("Best Accuracy Score:", study.best_value)
# print("Best Parameters:")
# for key, value in study.best_trial.params.items():
#     print(f"  {key}: {value}")

# # =====================================
# # Train best model on full train data
# # =====================================
# best_params = study.best_trial.params
# classifier_name = best_params["classifier"]

# if classifier_name == "LogisticRegression":
#     best_model = make_pipeline(
#         StandardScaler(),
#         LogisticRegression(
#             C=best_params["lr_C"],
#             penalty="l2",
#             solver="lbfgs",
#             max_iter=1000
#         )
#     )

# elif classifier_name == "SVC":
#     best_model = make_pipeline(
#         StandardScaler(),
#         SVC(
#             C=best_params["svc_C"],
#             kernel=best_params["svc_kernel"],
#             gamma=best_params["svc_gamma"]
#         )
#     )

# elif classifier_name == "DecisionTree":
#     best_model = DecisionTreeClassifier(
#         max_depth=best_params["dt_max_depth"],
#         min_samples_split=best_params["dt_min_samples_split"]
#     )

# elif classifier_name == "RandomForest":
#     best_model = RandomForestClassifier(
#         n_estimators=best_params["rf_n_estimators"],
#         max_depth=best_params["rf_max_depth"]
#     )

# elif classifier_name == "XGBoost":
#     best_model = XGBClassifier(
#         n_estimators=best_params["xgb_n_estimators"],
#         max_depth=best_params["xgb_max_depth"],
#         learning_rate=best_params["xgb_learning_rate"],
#         use_label_encoder=False,
#         eval_metric='logloss',
#         verbosity=0
#     )

# elif classifier_name == "LightGBM":
#     best_model = LGBMClassifier(
#         n_estimators=best_params["lgb_n_estimators"],
#         max_depth=best_params["lgb_max_depth"],
#         learning_rate=best_params["lgb_learning_rate"],
#         num_leaves=best_params["lgb_num_leaves"],
#         verbosity=-1
#     )

# # Fit the best model on training data
# best_model.fit(train_df_imputed, y_enc)

# # Make predictions on test set
# test_predictions = best_model.predict(test_df_imputed)

# # Show a sample of predictions
# print("Test Predictions (first 10):", test_predictions[:10])



import warnings
warnings.filterwarnings("ignore")

import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", [
        "LogisticRegression", "SVC", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"
    ])

    if classifier_name == "LogisticRegression":
        C = trial.suggest_loguniform("lr_C", 1e-4, 10)
        penalty = "l2"
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=C, penalty=penalty, solver="lbfgs", max_iter=1000))

    elif classifier_name == "SVC":
        C = trial.suggest_loguniform("svc_C", 1e-4, 10)
        kernel = trial.suggest_categorical("svc_kernel", ["linear", "rbf"])
        gamma = trial.suggest_categorical("svc_gamma", ["scale", "auto"])
        model = make_pipeline(StandardScaler(),
                              SVC(C=C, kernel=kernel, gamma=gamma))

    elif classifier_name == "DecisionTree":
        max_depth = trial.suggest_int("dt_max_depth", 1, 20)
        min_samples_split = trial.suggest_int("dt_min_samples_split", 2, 20)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

    elif classifier_name == "RandomForest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        max_depth = trial.suggest_int("rf_max_depth", 3, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif classifier_name == "XGBoost":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 15)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
        model = XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_depth,
                              learning_rate=learning_rate,
                              use_label_encoder=False,
                              eval_metric='logloss',
                              verbosity=0)

    elif classifier_name == "LightGBM":
        n_estimators = trial.suggest_int("lgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("lgb_max_depth", 3, 15)
        learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.3)
        num_leaves = trial.suggest_int("lgb_num_leaves", 20, 150)
        model = LGBMClassifier(n_estimators=n_estimators,
                               max_depth=max_depth,
                               learning_rate=learning_rate,
                               num_leaves=num_leaves,
                               verbosity=-1)

    score = cross_val_score(model, train_df_imputed, y_enc, cv=cv, scoring="accuracy").mean()
    return score

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)

# Print best trial
print("Best Model:", study.best_trial.params["classifier"])
print("Best Accuracy Score:", study.best_value)
print("Best Parameters:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")

# =====================================
# Train best model on full train data
# =====================================
best_params = study.best_trial.params
classifier_name = best_params["classifier"]

if classifier_name == "LogisticRegression":
    best_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=best_params["lr_C"],
            penalty="l2",
            solver="lbfgs",
            max_iter=1000
        )
    )

elif classifier_name == "SVC":
    best_model = make_pipeline(
        StandardScaler(),
        SVC(
            C=best_params["svc_C"],
            kernel=best_params["svc_kernel"],
            gamma=best_params["svc_gamma"]
        )
    )

elif classifier_name == "DecisionTree":
    best_model = DecisionTreeClassifier(
        max_depth=best_params["dt_max_depth"],
        min_samples_split=best_params["dt_min_samples_split"]
    )

elif classifier_name == "RandomForest":
    best_model = RandomForestClassifier(
        n_estimators=best_params["rf_n_estimators"],
        max_depth=best_params["rf_max_depth"]
    )

elif classifier_name == "XGBoost":
    best_model = XGBClassifier(
        n_estimators=best_params["xgb_n_estimators"],
        max_depth=best_params["xgb_max_depth"],
        learning_rate=best_params["xgb_learning_rate"],
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )

elif classifier_name == "LightGBM":
    best_model = LGBMClassifier(
        n_estimators=best_params["lgb_n_estimators"],
        max_depth=best_params["lgb_max_depth"],
        learning_rate=best_params["lgb_learning_rate"],
        num_leaves=best_params["lgb_num_leaves"],
        verbosity=-1
    )

# Fit the best model on training data
best_model.fit(train_df_imputed, y_enc)

# Make predictions on test set
test_predictions = best_model.predict(test_df_imputed)

# Show a sample of predictions
print("Test Predictions (first 10):", test_predictions[:10])



fin_preds = best_model.predict(test_df_imputed)


fin_df = pd.DataFrame({
    'id' : test_df['id'],
    'Personality' : fin_preds
})


fin_df['Personality'].value_counts()


fin_df['Personality'] = fin_df['Personality'].replace({1:'Extrovert',0:'Introvert'})


fin_df.head()


fin_df.to_csv('hyper_tunn3.csv',index = False)


# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score,classification_report


# lgr_model = LogisticRegression(penalty='l1', solver='liblinear')


# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # model = RandomForestClassifier(random_state=42)

# accuracies = []
# X =train_df_imputed
# y=y_enc
# for train_index, test_index in skf.split(X, y):
#     X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
#     lgr_model.fit(X_train_cv, y_train_cv)
#     y_pred = lgr_model.predict(X_test_cv)
    
#     acc = accuracy_score(y_test_cv, y_pred)
#     accuracies.append(acc)

# print("Cross-validated accuracies:", accuracies)
# print("Mean accuracy:", np.mean(accuracies))



# lgr_model.fit(X_train,y_train)


# y_lgr_pred = lgr_model.predict(X_test)


# print(accuracy_score(y_test,y_lgr_pred))


# print(classification_report(y_test,y_lgr_pred))


# test_df_lgr = lgr_model.predict(test_df_imputed)


# pred_lgr = pd.DataFrame({
#     'id':test_df['id'],
#     'Personality' : test_df_lgr
# })


# pred_lgr['Personality'].value_counts()


# Personality
# 1    4621
# 0    1554


# pred_lgr['Personality']  = pred_lgr['Personality'].replace({1:'Extrovert',0:'Introvert'})


# pred_lgr.to_csv('Lgr_labels.csv',index = False)


# from sklearn.svm import SVC


# model_svc = SVC()


# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # model = RandomForestClassifier(random_state=42)

# accuracies = []
# X =train_df_imputed
# y=y_enc
# for train_index, test_index in skf.split(X, y):
#     X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
#     model_svc.fit(X_train_cv, y_train_cv)
#     y_pred = model_svc.predict(X_test_cv)
    
#     acc = accuracy_score(y_test_cv, y_pred)
#     accuracies.append(acc)

# print("Cross-validated accuracies:", accuracies)
# print("Mean accuracy:", np.mean(accuracies))



# model_svc.fit(X_train,y_train)


# y_svc_preds = model_svc.predict(X_test)


# print(accuracy_score(y_test,y_svc_preds))


# test_pred_svc = model_svc.predict(test_df_imputed)


# pred_svc = pd.DataFrame({
#     'id' : test_df['id'],
#     'Personality' : test_pred_svc
# })


# pred_svc['Personality'].value_counts()


# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # model = RandomForestClassifier(random_state=42)

# accuracies = []
# X =train_df_imputed
# y=y_enc
# for train_index, test_index in skf.split(X, y):
#     X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
#     rf_model.fit(X_train_cv, y_train_cv)
#     y_pred = rf_model.predict(X_test_cv)
    
#     acc = accuracy_score(y_test_cv, y_pred)
#     accuracies.append(acc)

# print("Cross-validated accuracies:", accuracies)
# print("Mean accuracy:", np.mean(accuracies))



# rf_model.fit(X_train, y_train)

# # Predict
# y_pred_rf = rf_model.predict(X_test)


# print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# test_pred_rfc = rf_model.predict(test_df_imputed)


# model_rfc = pd.DataFrame({
#     'id' : test_df['id'],
#     'Personality' : test_pred_rfc
# })


# model_rfc['Personality'].value_counts()


# model_rfc['Personality'] = model_rfc['Personality'].replace({1:'Extrovert',0:'Introvert'})


# model_rfc.head()


# model_rfc.to_csv('rfc_preds.csv',index= False)


from xgboost import XGBClassifier


# xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')


# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # model = RandomForestClassifier(random_state=42)

# accuracies = []
# X =train_df_imputed
# y=y_enc
# for train_index, test_index in skf.split(X, y):
#     X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
#     xgb_model.fit(X_train_cv, y_train_cv)
#     y_pred = xgb_model.predict(X_test_cv)
    
#     acc = accuracy_score(y_test_cv, y_pred)
#     accuracies.append(acc)

# print("Cross-validated accuracies:", accuracies)
# print("Mean accuracy:", np.mean(accuracies))



# xgb_model.fit(X_train,y_train)


# y_pred_xgb = xgb_model.predict(X_test)


# print(accuracy_score(y_test,y_pred_xgb))


# y_xgb_pred = xgb_model.predict(test_df_imputed)


# xgb_md = pd.DataFrame({
#     'id' : test_df['id'],
#     'Personality' : y_xgb_pred
# })


# xgb_md['Personality'].value_counts()


# import lightgbm as lgb


# lgb_model = lgb.LGBMClassifier(
#     n_estimators=150,
#     learning_rate=0.1,
#     random_state=42,
#     verbose=-1  # suppresses warnings
# )

# # Fit model
# lgb_model.fit(X_train, y_train)


# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# import numpy as np

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # model = RandomForestClassifier(random_state=42)

# accuracies = []
# X =train_df_imputed
# y=y_enc
# for train_index, test_index in skf.split(X, y):
#     X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#     y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
#     lgb_model.fit(X_train_cv, y_train_cv)
#     y_pred = lgb_model.predict(X_test_cv)
    
#     acc = accuracy_score(y_test_cv, y_pred)
#     accuracies.append(acc)

# print("Cross-validated accuracies:", accuracies)
# print("Mean accuracy:", np.mean(accuracies))



# y_pred_lgb = lgb_model.predict(X_test)

# # Evaluate
# print("LightGBM Accuracy:", accuracy_score(y_test, y_pred_lgb))


# lgb_pred = lgb_model.predict(test_df_imputed)


# lgb_sub = pd.DataFrame({
#     'id' : test_df['id'],
#     'Personality' : lgb_pred
# })


# lgb_sub['Personality'].value_counts()


# lgb_sub['Personality'] = lgb_sub['Personality'].replace({0:'Introvert',1:'Extrovert'})
# lgb_sub.head()


# lgb_sub.to_csv('fin_pred_lgb.csv',index = False)


# pred_lgr['p_logi'] = pred_lgr['Personality'].replace({'Extrovert':1,'Introvert':0})


# pred_lgr['p_lgb'] = lgb_sub['Personality']


# pred_lgr.head()


# pred_lgr['p_lgb'] = pred_lgr['p_lgb'].replace({'Extrovert' :1,'Introvert':0})


# pred_lgr['diff'] = pred_lgr['p_logi'] - pred_lgr['p_lgb']


# ind_val = pred_lgr[pred_lgr['diff'] != 0].index


# test_df.iloc[ind_val]




