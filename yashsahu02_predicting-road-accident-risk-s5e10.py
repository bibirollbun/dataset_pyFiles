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


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


## list of columns 
train.columns


train.head()


test.head()


sample_submission.head()


## checking is there any null value in train
train.isnull().sum()


## checking is there any null value in test
test.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt


train['road_type'].value_counts()


train['road_type'].value_counts().values


train['road_type'].value_counts().index


train['curvature'].describe()


# ## function that plots the graphs and basic details 
# def show_details_and_plots(df, feature):
#     target_feature = "accident_risk"
    
#     # if the feature is object type (categorical)
#     if(df[feature].dtype=='O'):
#         print("Feature Name:", feature)
#         print(f"Total Unique Categories: {df[feature].nunique()}")
#         print("Value Counts:")
#         display(df[feature].value_counts())
#         print("*"*40)
        
#         plt.figure(figsize=(18,13))
#         plt.subplot(2,2,1)
#         sns.countplot(x=feature, data=df)
#         plt.title(f"Count Plot for {feature}")

#         plt.subplot(2,2,2)
#         # df[feature].value_counts().plot().pie(autopct='%1.1f%%')
#         plt.pie(data=df,x=df[feature].value_counts().values, labels=df[feature].value_counts().index, autopct='%.1f%%')
#         plt.title(f"Pie Chart for {feature}")

#         plt.subplot(2,2,3)
#         sns.boxplot(x=feature, y=target_feature, data=df)
#         plt.title(f"Boxplot for {feature}")

#         plt.subplot(2,2,4)
#         sns.barplot(x=feature, y=target_feature, data=df)
#         plt.title(f"Bar Plot for {feature}")
#         plt.show()
        
#     # for the numerical features
#     elif(df[feature].dtype!='O'):
#         train['curvature'].describe()
#         print("*"*40)
#         plt.figure(figsize=(20,20))
#         plt.subplot(2,2,1)
#         sns.histplot(data=df, x=feature, kde=True)
#         plt.title(f"Histplot for {feature}")

#         plt.subplot(2,2,2)
#         sns.kdeplot(data=df, x=feature)
#         plt.title(f"kde Plot for {feature}")


#         plt.subplot(2,2,3)
#         sns.boxplot(x=df[feature])
#         plt.title(f"Box Plot for {feature}")

#         plt.subplot(2,2,4)
#         sns.scatterplot(data=df, x=feature, y=target_feature)
#         plt.title(f"Scatter Plot between {feature} and {target_feature}")
#         plt.show()
        
#     else:
#         print(f"{feature} is neither Numeric nor Categorical...")

#     print("*"*200)
#     print()
        


train.dtypes


# Convert infinite values to NaN
train.replace([np.inf, -np.inf], np.nan, inplace=True)


train.dtypes


# train 
train['road_signs_present'] = train['road_signs_present'].replace({True:"Yes",False:"No"})
train['public_road'] = train['public_road'].replace({True:"Yes",False:"No"})
train['holiday'] = train['holiday'].replace({True:"Yes",False:"No"})
train['school_season'] = train['school_season'].replace({True:"Yes",False:"No"})

# test
test['road_signs_present'] = test['road_signs_present'].replace({True:"Yes",False:"No"})
test['public_road'] = test['public_road'].replace({True:"Yes",False:"No"})
test['holiday'] = test['holiday'].replace({True:"Yes",False:"No"})
test['school_season'] = test['school_season'].replace({True:"Yes",False:"No"})


train.dtypes


# for feature in train.columns:
#     show_details_and_plots(train, feature)


train.head(10)


categorical_features = [feature for feature in train.columns if train[feature].dtype=='O']


from sklearn.preprocessing import LabelEncoder


## applying LabelEncoder
for feature in categorical_features:
    le = LabelEncoder()
    train[feature] = le.fit_transform(train[feature])
    test[feature] = le.transform(test[feature])


## dropping the id feature because it is not useful in predction 
train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


print("Shape of train: ",train.shape)
print("Shape of test: ",test.shape)


X = train.drop('accident_risk', axis=1)
y = train['accident_risk']


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

# External libraries
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


from sklearn.metrics import mean_squared_error


from sklearn.model_selection import KFold
num_fold=5


# models = {
#     "LinearRegression": LinearRegression(),
#     "Ridge": Ridge(),
#     "Lasso": Lasso(),
#     "ElasticNet": ElasticNet(),
#     "DecisionTreeRegressor": DecisionTreeRegressor(),
#     # "RandomForestRegressor": RandomForestRegressor(),
#     # GPU-enabled gradient boosting libraries
#     "XGBRegressor": XGBRegressor(
#         verbosity=0,
#         tree_method='gpu_hist',    # enables GPU training
#         predictor='gpu_predictor'  # uses GPU for prediction
#     ),
    
#     "LGBMRegressor": LGBMRegressor(
#         verbose=-1,
#         device='gpu',              # enables GPU
#         gpu_platform_id=0,
#         gpu_device_id=0
#     ),
    
#     "CatBoostRegressor": CatBoostRegressor(
#         verbose=0,
#         task_type='GPU',           # enables GPU
#         devices='0'                # specify GPU id if needed
#     )
# }

# num_folds = 5
# model_list = []
# avg_rmse_list = []

# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# for model_name, model in models.items():
#     print(model_name, "================>\n")

#     rmse_score_list = []
#     test_pred = np.zeros(test.shape[0])

#     for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
#         model.fit(X.iloc[train_idx], y.iloc[train_idx])
#         y_test_pred = model.predict(X.iloc[test_idx])

#         score = np.sqrt(mean_squared_error(y.iloc[test_idx], y_test_pred))
#         print(f"Fold {i+1} RMSE: {score:.4f}")
#         rmse_score_list.append(score)

#         # Prediction on test data (for ensemble averaging)
#         test_pred += model.predict(test)

#     avg_score = np.mean(rmse_score_list)
#     print(f"Average RMSE: {avg_score:.4f}\n")
#     print('-'*60, '\n')

#     model_list.append(model_name)
#     avg_rmse_list.append(avg_score)

#     prediction = test_pred / num_folds
#     sample_submission['accident_risk'] = prediction
#     sample_submission.to_csv(f"{model_name}_prediction.csv", index=False)
#     print(f"File saved as {model_name}_prediction.csv")
#     display(sample_submission.head())
#     print("*"*60)

# # Performance tracking
# performance_df = pd.DataFrame({
#     "Model Name": model_list,
#     "AVG RMSE Score": avg_rmse_list
# })


# display(performance_df)


catboost_best_params = {'iterations': 1100, 
               'depth': 8, 
               'learning_rate': 0.2534517792765171, 
               'l2_leaf_reg': 9.721060107731173, 
               'border_count': 254, 
               'random_strength': 6.126661058319514, 
               'boosting_type': 'Ordered', 
               'bootstrap_type': 'Bernoulli', 
               'leaf_estimation_iterations': 8}


print(study_lgbm.best_params)


lgbm_best_params = {'learning_rate': 0.061436227425352374, 'num_leaves': 189, 'max_depth': 11, 'min_child_samples': 16, 'subsample': 0.8718221021766972, 'colsample_bytree': 0.9737278808826064, 'reg_alpha': 1.7764872231225977, 'reg_lambda': 4.495223215871367}


from lightgbm import LGBMRegressor


# # best_params = study.best_params
# lgbm_best_params.update({
#     'device_type': 'gpu',       # ✅ Enables GPU
#     'gpu_platform_id': 0,
#     'gpu_device_id': 0,
#     'verbose': -1,
#     'random_state': 42
# })

# print(lgbm_best_params)

# model = LGBMRegressor(**lgbm_best_params)

# num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# rmse_score_list = []
# test_pred = np.zeros(test.shape[0])

# for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
#     model.fit(X.iloc[train_idx], y.iloc[train_idx])
#     y_test_pred = model.predict(X.iloc[test_idx])

#     score = np.sqrt(mean_squared_error(y.iloc[test_idx], y_test_pred))
#     print(f"Fold {i+1} RMSE: {score:.4f}")
#     rmse_score_list.append(score)

#     test_pred += model.predict(test)

# avg_rmse = np.mean(rmse_score_list)
# print(f"\nAverage RMSE: {avg_rmse:.4f}")

# # Save predictions
# sample_submission["accident_risk"] = test_pred / num_folds
# sample_submission.to_csv("LGBMRegressor_prediction.csv", index=False)
# print("File saved as LGBMRegressor_prediction.csv")


# num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
# rmse_score_list = []
# test_pred = np.zeros(test.shape[0])

# for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
#     model = LGBMRegressor(**lgbm_best_params)
#     model.fit(X.iloc[train_idx], y.iloc[train_idx])
#     y_test_pred = model.predict(X.iloc[test_idx])

#     score = np.sqrt(mean_squared_error(y.iloc[test_idx], y_test_pred))
#     print(f"Fold {i+1} RMSE: {score:.4f}")
#     rmse_score_list.append(score)

#     test_pred += model.predict(test)

# avg_rmse = np.mean(rmse_score_list)
# print(f"\nAverage RMSE: {avg_rmse:.4f}")

# # Save predictions
# sample_submission["accident_risk"] = test_pred / num_folds
# sample_submission.to_csv("LGBMRegressor_prediction2.csv", index=False)
# print("File saved as LGBMRegressor_prediction2.csv")


import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Split once for consistent validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        'n_estimators': 10000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'gpu_hist',    # ✅ Enables GPU training
        'predictor': 'gpu_predictor', # ✅ Uses GPU for prediction too
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42
    }

    model = XGBRegressor(**params, verbosity=0)

    # Fit with early stopping for faster Optuna search
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False
    )

    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse


# ✅ Create or resume Optuna study
study_xgb = optuna.create_study(
    study_name="xgb_regression_rmse_study",
    direction="minimize",
    storage="sqlite:///xgb_regression.db",  # save progress
    load_if_exists=True
)

# ✅ Run optimization
study_xgb.optimize(objective, n_trials=50, show_progress_bar=True)

# ✅ Best results
print("Best Params:", study_xgb.best_params)
print("Best RMSE:", study_xgb.best_value)


best_params_xgb = study_xgb.best_params


best_params_xgb


from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# ✅ After Optuna tuning, load your best params
xgb_best_params = study_xgb.best_params  # Assuming you named your study 'study_xgb'

# ✅ Add fixed parameters
xgb_best_params.update({
    'tree_method': 'gpu_hist',    # ✅ Enables GPU acceleration
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'n_estimators': 10000,
    'verbosity': 0
})

print("Final XGBoost Parameters:")
print(xgb_best_params)

# ✅ Initialize model
model = XGBRegressor(**xgb_best_params)

# ✅ K-Fold Cross Validation
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
rmse_score_list = []
test_pred = np.zeros(test.shape[0])

for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[test_idx], y.iloc[test_idx])],
        eval_metric='rmse',
        early_stopping_rounds=100,
        verbose=False
    )
    
    y_test_pred = model.predict(X.iloc[test_idx])
    score = np.sqrt(mean_squared_error(y.iloc[test_idx], y_test_pred))
    print(f"Fold {i+1} RMSE: {score:.4f}")
    rmse_score_list.append(score)

    # ✅ Predict test set
    test_pred += model.predict(test)

# ✅ Average results
avg_rmse = np.mean(rmse_score_list)
print(f"\nAverage RMSE: {avg_rmse:.4f}")

# ✅ Save predictions
sample_submission["accident_risk"] = test_pred / num_folds
sample_submission.to_csv("XGBRegressor_prediction.csv", index=False)
print("File saved as XGBRegressor_prediction.csv")





