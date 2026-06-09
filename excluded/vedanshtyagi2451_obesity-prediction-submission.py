import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import numpy as np 
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,  confusion_matrix, make_scorer
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

!pip install optuna



%%capture
!pip install openfe


from openfe import OpenFE, tree_to_formula, transform


train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv', index_col='id')
train.head(3)


test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv', index_col='id')
test.head(3)


submission_df = pd.read_csv('/kaggle/input/playground-series-s4e2/sample_submission.csv')
submission_df.head(3)


# List of object features
object_cols = test.select_dtypes(include=['object']).columns
object_cols


def one_hot_encoding(X):
    object_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=object_cols, prefix=object_cols, dtype=int)
    return X


train_conv = one_hot_encoding(train[train.columns[:-1]])
train_conv["target"] = train[train.columns[-1]]
train = train_conv.copy()
train.head(3)


test = one_hot_encoding(test)
test = test.drop('CALC_Always', axis=1)
test.head(3)


for df in [train, test]:
    df['BMI'] = df['Weight']/df['Height']**2


int_columns = train.select_dtypes(include='float').columns
int_columns


def remove_zscore_outliers(df, col, threshold=3):
    m = np.mean(df[col])
    sd = np.std(df[col])
    z_scores = (df[col] - m) / sd
    return df[np.abs(z_scores) <= threshold]

cols = int_columns
train_ = train.copy()
for col in cols:
    train_ = remove_zscore_outliers(train_, col)
train = train_.copy()
train.head(3)


X = train.copy()
y = X.pop('target')


# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(train[train.columns[-1]])
# X = train[train.columns[:-1]]


# import optuna

# def objective(trial):
#     x = trial.suggest_int('x', -10, 10)
#     return (np.exp(x)/2) ** 2

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

# study.best_params 


LGBMClassifier()


# 'n_estimators': [n for n in range(100, 1000, 100)],
#               'max_depth': [n for n in range(3, 30, 1)],
#               'min_child_samples': [n for n in range(2, 100, 2)],
#               'learning_rate': [0.001, 0.01, 0.1],
#               'num_leaves': [n for n in range(2, 50, 1)],
#               'min_split_gain': [0, 0.01, 0.001, 0.5, 0.05, 0.0001, 1], 
#               'objective': ['binary'],
#               'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#               'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#               'boosting_type': ['gbdt'],
#               'subsample_freq': [n for n in range(0, 100, 1)]


# import optuna
# def objective(trial):
#     lgbmc = LGBMClassifier(num_leaves = trial.suggest_int('num_leaves', 4, 50),
#                            max_depth = trial.suggest_int('max_depth', 4, 50),
#                            learning_rate = trial.suggest_float('learning_rate', 0.0001, 1),
#                            n_estimators = trial.suggest_int('n_estimators',100, 2000),
#                            min_split_gain = trial.suggest_float('min_split_gain', 0, 1),
#                            min_child_weight = trial.suggest_float('min_child_weight', 0.0, 1.0),
#                            subsample = trial.suggest_float('subsample', 0.3, 1),
#                            min_child_samples = trial.suggest_int('min_child_samples', 4, 50),
#                            colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1),
#                            reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0),
#                            colsample_bynode = trial.suggest_float('colsample_bynode', 0.3, 1.0),
#                            subsample_freq = trial.suggest_int('subsample_freq', 0, 100),
#                            verbose=1
#                           )
    
#     score = cross_val_score(lgbmc, X, y, n_jobs=-1, cv=3)
#     accuracy = score.mean()
#     return accuracy

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=20)


# best_trial = study.best_trial


# optuna.visualization.plot_optimization_history(study)


# optuna.visualization.plot_slice(study)


# best_trial.value


# best_trial.params


# study.best_params


# study.best_params


# def objective(trial):
#     model = CatBoostClassifier(
#         iterations=trial.suggest_int("iterations", 100, 1000),
#         learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
#         depth=trial.suggest_int("depth", 4, 10),
#         l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
#         bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
#         random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
#         bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
#         od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
#         od_wait=trial.suggest_int("od_wait", 10, 50),
#         verbose=False
#     )
# #     model.fit(X, y)
# #     y_pred = model.predict(X_t)
# #     return accuracy_score(y_test, y_pred)

#     score = cross_val_score(model, X, y, n_jobs=-1, cv=3)
#     accuracy = score.mean()
# #     return accuracy


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=10)


# best_params = study.best_trial.params
# best_params


from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

# Define the best parameters obtained from optimization
best_lgbm_params = {
                    'boosting_type':"gbdt",
                     'objective':"multiclass",
                     'metric':"multi_logloss",
                     'num_class':7,
                     'learning_rate':0.025,
                     'n_estimators':500,
                     'lambda_l1':0.06,
                     'lambda_l2':0.3,
                     'max_depth':9,
                     'colsample_bytree':0.40,
                     'subsample':0.85,
                     'min_child_samples':15,
                     'verbosity':-11,
                    "verbose":0
}



best_xgboost_params = {'grow_policy': 'lossguide',
                       'n_estimators': 679,
                       'learning_rate': 0.021466845326995,
                       'gamma': 0.10314733748964279,
                       'subsample': 0.5717293764790643,
                       'colsample_bytree': 0.35831057532679117,
                       'max_depth': 11,
                       'min_child_weight': 7,
                       'reg_lambda': 2.190155853764884,
                       'reg_alpha': 0.9232583576479211
                      }
xgboost_params_2 = {
        'n_estimators': 1314,
    'learning_rate': 0.0182795207589753689,
    'gamma': 0.0024196352444278957,
    'reg_alpha': 0.9025936864379646,
    'reg_lambda': 0.06835589643789557,
    'max_depth': 5,
    'min_child_weight': 5,
    'subsample': 0.8832747986546899,
    'colsample_bytree': 0.65798288646896546
}

# Initialize the models with the best parameters
lgbm_model = LGBMClassifier(**best_lgbm_params)
xgboost_model = xgb.XGBClassifier(**best_xgboost_params)
xgboost_model_2 = xgb.XGBClassifier(**xgboost_params_2)

# Store the models in a list
models = [lgbm_model, xgboost_model,xgboost_model_2]
models_name = ['lgbm','xbg','xgb_2']


class_dict = {'Insufficient_Weight': 0, 
              'Normal_Weight': 1, 
              'Overweight_Level_I': 2, 
              'Overweight_Level_II': 3,
              'Obesity_Type_I': 4,
              'Obesity_Type_II': 5,
              'Obesity_Type_III': 6
              }


X_train_prep, X_valid_prep, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# fit the model
lgbm_model.fit(X_train_prep, y_train)

# predict the test dataset
y_preds = lgbm_model.predict(X_valid_prep)

# get the confusion matrix
conf_matrix = confusion_matrix(y_valid, y_preds)

# plot the conf_matrix heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            cmap='Blues', cbar=False,
#             xticklabels= class_dict, 
#             yticklabels= class_dict
           )
plt.show()


# # import sys
# # sys.path.append('../')
# import pandas as pd
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# import lightgbm as lgb
# from sklearn.metrics import mean_squared_error
# def get_score(train_x, test_x, train_y, test_y):
#     train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
#     params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
#     gbm = lgb.LGBMRegressor(**params)
#     gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
#     pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
#     score = mean_squared_error(test_y, pred)
#     return score
# if __name__ == '__main__':
#     n_jobs = 4
#     data = fetch_california_housing(as_frame=True).frame
#     label = data[['MedHouseVal']]
#     del data['MedHouseVal']
#     train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
#     # get baseline score
#     score = get_score(train_x, test_x, train_y, test_y)
#     print("The MSE before feature generation is", score)
#     # feature generation
#     ofe = OpenFE()
#     ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)
#     # OpenFE recommends a list of new features. We include the top 10
#     # generated features to see how they influence the model performance
#     train_x, test_x = transform(train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
#     score = get_score(train_x, test_x, train_y, test_y)
#     print("The MSE after feature generation is", score)
#     print("The top 10 generated features are")
#     for feature in ofe.new_features_list[:20]:
#         print(tree_to_formula(feature))


# # feature generation
# ofe = OpenFE()
# ofe.fit(data=X[:, :6], label=y, n_jobs=n_jobs)
# # OpenFE recommends a list of new features. We include the top 10
# # generated features to see how they influence the model performance
# train_x, test_x = transform(train, test, ofe.new_features_list[:10], n_jobs=n_jobs)

# train_x.head(3)


lgbm_model.fit(X,y)
preds = lgbm_model.predict(test)
# a = label_encoder.inverse_transform(pred)

submission_df['NObeyesdad'] = preds
submission_df.to_csv("submission.csv",index=False)
submission_df


    

