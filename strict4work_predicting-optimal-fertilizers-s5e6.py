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


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV

import optuna
import numpy as np
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train.describe()


train.info()


train.isnull().sum()


test.isnull().sum()


numerical = train.select_dtypes(include=["int64", "float64"])
categorical = train.select_dtypes(exclude=["int64", "float64"])


# Check Distribution

# Check Distribution
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

numerical_noid = numerical.drop(columns=['id'])
# train.select_dtypes(include=["number"]).columns
for i, col in enumerate(numerical_noid.columns):
    ax = axes.flat[i]
    sns.kdeplot(data=numerical, x=col, ax=ax)  # KDE plot (no histogram)
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15,15))
c = numerical.corr()
sns.heatmap(c, cmap="coolwarm", annot=True)
plt.show()


# Create the scatter plot
for var in ["Humidity", "Temparature", "Moisture"]:
    for fertilizer in train["Fertilizer Name"].unique():
        sns.scatterplot(x="Fertilizer Name", y=var, data=train[train["Fertilizer Name"] == fertilizer], color="royalblue")

    # Add labels and title
    plt.title(f"Fertilizer Name vs {var}")
    plt.xlabel("Fertilizer Name")
    plt.ylabel(f"{var}")
    plt.tight_layout()
    plt.show()


# A more specific plot to make sure my assumption make sense.
# Subplot
fig, axes = plt.subplots(1, 2, figsize=(8, 8))

# Scatter plot
sns.scatterplot(x="Temparature", y="Humidity", data=train[(train["Temparature"] == 27) | (train["Temparature"] == 33)], color="royalblue", ax=axes[0])

# Add labels and titles
axes[0].set_title("Temparature vs Moisture")
axes[0].set_xlabel("Temparature (°C)")
axes[0].set_ylabel("Humidity (%)")

# Scatter plot
sns.scatterplot(x="Humidity", y="Moisture", data=train[(train["Moisture"] == 29) | (train["Moisture"] == 62)], color="royalblue", ax=axes[1])

# Add labels and titles
axes[1].set_title("Humidity vs Moisture")
axes[1].set_xlabel("Humidity (%)")
axes[1].set_ylabel("Moisture (%)")

# Layout adjustments
plt.tight_layout()
plt.suptitle("Fertilizer Conditions by Temp & Moisture", fontsize=16)
plt.subplots_adjust(top=0.93)  # Prevent overlap with suptitle
plt.show()


# Reclassify
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjGvODG3NSNAxVScmwGHX0FLJ8QFnoECBkQAQ&url=https%3A%2F%2Fwww.fao.org%2Ffileadmin%2Ftemplates%2Fess%2Fdocuments%2Fworld_census_of_agriculture%2Fappendix3_r7.pdf&usg=AOvVaw3lGR3a_-fOtV2PNI2aHZY3&opi=89978449

# Paddy : Cereals
# Pulses : Leguminous crops
# Cotton : Other crops
# Tobacco : Other crops
# Wheat : Cereals
# Millets : Cereals
# Barley : Cereals
# Sugarcane : sugar crop
# Oil seeds : Oilseed crops
# Maize :  Cereals
# Ground nuts: Oilseed crops

# general_crop_type_mapping = {
#     "Paddy":"Cereals",
#     "Pulses":"Leguminous Crops",
#     "Cotton":"Other Crops",
#     "Tobacco":"Other Crops",
#     "Wheat":"Cereals",
#     "Millets":"Cereals",
#     "Barley":"Cereals",
#     "Sugarcane":"Sugar Crops",
#     "Oil seeds":"Oilseed Crops",
#     "Maize":"Cereals",
#     "Ground Nuts":"Oilseed Crops"
# }


# Exclude ID
train = train.drop(columns=["id"], axis=1)
train.rename(columns={"Temparature": "Temperature"}, inplace=True)

labelEncoder_ct = LabelEncoder()
labelEncoder_st = LabelEncoder()
labelEncoder_fn = LabelEncoder()

train["Crop Type"] = labelEncoder_ct.fit_transform(train["Crop Type"])
train["Soil Type"] = labelEncoder_st.fit_transform(train["Soil Type"])
train["Fertilizer Name"] = labelEncoder_fn.fit_transform(train["Fertilizer Name"])

for col in train.select_dtypes(include=["bool"]):
    train[col] = train[col].map({True: 1, False: 0})

# Td = T- (100 - Relative Humidity) / 5 ~ approximation of 3 deg difference
# train["Dew Point Temperature"] = train["Temperature"] - ((100 - train["Humidity"]) / 5.)

X = train.drop(columns=["Fertilizer Name"], axis=1)
y = train["Fertilizer Name"]


# # MAP Explained https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code/notebook

# import numpy as np

# def apk(actual, predicted, k=10):
#     """
#     Computes the average precision at k.
#     This function computes the average prescision at k between two lists of
#     items.
#     Parameters
#     ----------
#     actual : list
#              A list of elements that are to be predicted (order doesn't matter)
#     predicted : list
#                 A list of predicted elements (order does matter)
#     k : int, optional
#         The maximum number of predicted elements
#     Returns
#     -------
#     score : double
#             The average precision at k over the input lists
#     """
#     if not actual:
#         return 0.0

#     if len(predicted)>k:
#         predicted = predicted[:k]

#     score = 0.0
#     num_hits = 0.0

#     for i,p in enumerate(predicted):
#         # first condition checks whether it is valid prediction
#         # second condition checks if prediction is not repeated
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)

#     return score

# def mapk(actual, predicted, k=10):
#     """
#     Computes the mean average precision at k.
#     This function computes the mean average prescision at k between two lists
#     of lists of items.
#     Parameters
#     ----------
#     actual : list
#              A list of lists of elements that are to be predicted 
#              (order doesn't matter in the lists)
#     predicted : list
#                 A list of lists of predicted elements
#                 (order matters in the lists)
#     k : int, optional
#         The maximum number of predicted elements
#     Returns
#     -------
#     score : double
#             The mean average precision at k over the input lists
#     """
#     return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# MAP Explained https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code/notebook

def mapk(actual, predicted, k=3):
    def apk(actual, predicted, k):

        if not actual:
            return 0.0

        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                if p == actual:
                    score += 1.0 / (i+1)
                    break

        return score
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# def objective(trial):
#     params = {
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
#         "max_depth": trial.suggest_int("max_depth", 6, 12),
#         "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
#     }

#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     xgbmodel = xgb.XGBClassifier(**params, 
#                                 objective="multi:softprob",
#                                 booster="gbtree",
#                                 device="gpu",
#                                 tree_method="gpu_hist",
#                                 random_state=42,
#                                 metric_name="mlogloss",
#                                 early_stopping_rounds=200,
#                                 verbosity=0
#                                 )

#     kf_score = []
#     for fold, (xdx, ydx) in enumerate(kf.split(X, y)):

#         X_train_fold, X_test_fold = X.iloc[xdx], X.iloc[ydx]
#         y_train_fold, y_test_fold = y.iloc[xdx], y.iloc[ydx]

#         print(f"\n{'*' * 20} FOLD: {fold+1} {'*' * 20}")

#         xgbmodel.fit(X_train_fold, y_train_fold, eval_set=[(X_test_fold, y_test_fold)], verbose=100)

#         y_pred = xgbmodel.predict_proba(X_test_fold)
#         top3_preds = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
#         top3_preds = [labelEncoder_fn.inverse_transform(row) for row in top3_preds]
#         y_true_label = labelEncoder_fn.inverse_transform(y_test_fold)
#         map3 = mapk(y_true_label, top3_preds)

#         kf_score.append(map3)
#         fold +=1

#     print(f"Score: {kf_score}")
#     print(f"Mean score: {np.mean(kf_score):.6f}")
#     return np.mean(kf_score)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)
# print("Best trial:")
# print(study.best_trial.params)


best_params={"learning_rate": 0.05094167382251042,
             "max_depth": 6,
             "n_estimators": 1545,
             "reg_alpha": 0.3040596266944266,
             "reg_lambda": 0.034673978288113666
            }

X = train.drop(columns=["Fertilizer Name"], axis=1)
y = train["Fertilizer Name"]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgbmodel = xgb.XGBClassifier(**best_params, 
                            objective="multi:softprob",
                            booster="gbtree",
                            device="cpu",
                            tree_method="hist",
                            random_state=42,
                            metric_name="mlogloss",
                            early_stopping_rounds=200,
                            verbosity=0
                            )

kf_score = []
for fold, (xdx, ydx) in enumerate(kf.split(X, y)):

    X_train_fold, X_test_fold = X.iloc[xdx], X.iloc[ydx]
    y_train_fold, y_test_fold = y.iloc[xdx], y.iloc[ydx]

    print(f"\n{'*' * 20} FOLD: {fold+1} {'*' * 20}")

    xgbmodel.fit(X_train_fold, y_train_fold, eval_set=[(X_test_fold, y_test_fold)], verbose=100)

    y_pred = xgbmodel.predict_proba(X_test_fold)
    top3_preds = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
    top3_preds = [labelEncoder_fn.inverse_transform(row) for row in top3_preds]
    y_true_label = labelEncoder_fn.inverse_transform(y_test_fold)
    map3 = mapk(y_true_label, top3_preds)

    kf_score.append(map3)
    fold +=1

print(f"Score: {kf_score}")
print(f"Mean score: {np.mean(kf_score):.6f}")


# Submission
test["Crop Type"] = labelEncoder_ct.fit_transform(test["Crop Type"])
test["Soil Type"] = labelEncoder_st.fit_transform(test["Soil Type"])

for col in test.select_dtypes(include=["bool"]):
    test[col] = test[col].map({True: 1, False: 0})

# Td = T- (100 - Relative Humidity) / 5 ~ approximation of 3 deg difference

test.rename(columns={"Temparature": "Temperature"}, inplace=True)
# test["Dew Point Temperature"] = test["Temperature"] - ((100 - test["Humidity"]) / 5.)

test_submission = test[[col for col in test.columns if col != 'id']]
prediction = xgbmodel.predict_proba(test_submission)
top3_prediction = np.argsort(prediction, axis=1)[:, -3:][:, ::-1]
top3_prediction = [labelEncoder_fn.inverse_transform(row) for row in top3_prediction]
submission = pd.DataFrame(data={"id":test["id"],"Fertilizer Name":[" ".join(pred) for pred in top3_prediction]})
submission.to_csv("submission.csv", index=False)


# from sklearn.model_selection import train_test_split

# X = train.drop(columns=["Fertilizer Name"], axis=1)
# y = train["Fertilizer Name"]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# import lightgbm as lgb


# params = {
#     'boosting_type': 'gbdt',
#     'class_weight': None,
#     'colsample_bytree': 1.0,
#     'importance_type': 'split',
#     'learning_rate': 0.1,
#     'max_depth': -1,
#     'min_child_samples': 20,
#     'min_child_weight': 0.001,
#     'min_split_gain': 0.0,
#     'n_estimators': 100,
#     'n_jobs': None,
#     'num_leaves': 31,
#     'objective': None,
#     'random_state': 42,
#     'reg_alpha': 0.0,
#     'reg_lambda': 0.0,
#     'subsample': 1.0,
#     'subsample_for_bin': 200000,
#     'subsample_freq': 0
# }

# lgbm = lgb.LGBMClassifier(**params)

# lgbm.fit(X_train, y_train, eval_set=(X_test, y_test), callbacks=[lgb.early_stopping(stopping_rounds=50)])

# y_pred = lgbm.predict_proba(X_test)
# top3_preds = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
# top3_preds = [labelEncoder.inverse_transform(row) for row in top3_preds]
# y_true_label = labelEncoder.inverse_transform(y_test)
# map3 = mapk(y_true_label, top3_preds)
# print(map3)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.impute import SimpleImputer

# # Load the dataset
# df = pd.read_csv('your_train_dataset.csv')

# # Retrieve simple information of the dataset
# print("First few rows of the dataframe:")
# print(df.head())

# print("\nSummary statistics of the numeric columns:")
# print(df.describe())

# print("\nColumn names and data types:")
# print(df.info())

# # Handle missing data
# # Identify missing values
# missing_values = df.isnull().sum()
# print("\nMissing Values Count per Column:")
# print(missing_values)

# threshold = len(df) * 0.50  # Adjust the threshold as needed
# missing_percentage = (df.isnull().sum() / len(df)) * 100
# high_missing_cols = missing_percentage[missing_percentage > threshold].index
# print("\nColumns with high missing values:")
# print(high_missing_cols)

# # Impute missing data using SimpleImputer
# imputer = SimpleImputer(strategy='mean')  # You can choose other strategies like 'median' or 'most_frequent'
# df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# # Verify the imputation
# print("\nMissing Values after Imputation:")
# missing_values_after_imputation = df_imputed.isnull().sum()
# print(missing_values_after_imputation)

# # Plot charts

# # Scatter Plot
# sns.pairplot(df_imputed)  # For numeric features, this will plot pairwise relationships
# plt.show()

# # Box Plot
# sns.boxplot(data=df_imputed)  # For numeric features, this will create a box plot for each column
# plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap
# plt.title('Box Plot of Numerical Features')
# plt.show()

# # Correlation Heatmap
# correlation_matrix = df_imputed.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

# # Distribution of Numerical Features
# for col in df_imputed.select_dtypes(include=['float64', 'int64']).columns:
#     sns.histplot(df_imputed[col], kde=True)
#     plt.title(f'Histogram of {col}')
#     plt.show()

# # Count Plot for Categorical Features
# for col in df_imputed.select_dtypes(include=['object', 'category']).columns:
#     sns.countplot(data=df_imputed, x=col)
#     plt.title(f'Count Plot of {col}')
#     plt.show()

# # Correlation with Target Variable (if available)
# target = 'target'
# correlations_numeric = df_imputed.corrwith(df[target])
# print("\nCorrelation of Numerical Features with the Target Variable:")
# print(correlations_numeric)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=correlations_numeric.index, y=correlations_numeric.values)
# plt.title('Correlation of Numeric Features with the Target')
# plt.show()


# # Exclude ID
# train = train.drop(columns=["id"], axis=1)
# train.rename(columns={"Temparature": "Temperature"}, inplace=True)

# labelEncoder_ct = LabelEncoder()
# labelEncoder_st = LabelEncoder()
# labelEncoder_fn = LabelEncoder()

# train["Crop Type"] = labelEncoder_ct.fit_transform(train["Crop Type"])
# train["Soil Type"] = labelEncoder_st.fit_transform(train["Soil Type"])
# train["Fertilizer Name"] = labelEncoder_fn.fit_transform(train["Fertilizer Name"])

# for col in train.select_dtypes(include=["bool"]):
#     train[col] = train[col].map({True: 1, False: 0})

# # Td = T- (100 - Relative Humidity) / 5 ~ approximation of 3 deg difference
# # train["Dew Point Temperature"] = train["Temperature"] - ((100 - train["Humidity"]) / 5.)

# X = train.drop(columns=["Fertilizer Name"], axis=1)
# y = train["Fertilizer Name"]
# num_class = y.nunique()

# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_dmtrain = xgb.DMatrix(X_train, label=y_train)
# X_dmtest = xgb.DMatrix(X_test, label=y_test)

# # params = {
# #     "objective": ["multi:softprob"],
# #     "booster": ["gbtree"],
# #     "early_stopping_rounds":[200],
# #     # "colsample_bylevel": None,
# #     # "colsample_bynode": None,
# #     # "colsample_bytree": None,
# #     "device": ["cpu"],
# #     "eval_metric": ["mlogloss"],
# #     "gamma": [0, 2, 4],
# #     "learning_rate": [0.01, 0.1, 0.3],
# #     "max_bin": [256],
# #     # "max_cat_threshold": None,
# #     # "max_cat_to_onehot": None,
# #     # "max_delta_step": None,
# #     "max_depth": [6, 8, 10],
# #     # "max_leaves": None,
# #     # "min_child_weight": None,
# #     "tree_method": ["hist"],
# #     "n_estimators": [1000, 2000],
# #     "random_state": [42],
# #     "reg_alpha": [0],
# #     "reg_lambda": [0],
# #     "verbosity": [0]
# # }

# xgb_dmatrix_params = {
#     "learning_rate": 0.1,
#     "gamma": 0,
#     "max_depth": 7,
#     "objective": "multi:softprob",
#     "lambda": 0,
#     "alpha": 0,
#     "tree_method": "hist",
#     "num_class" : num_class,
# }

# # xgbRSCV = xgb.XGBClassifier(**params)
# # xgbRSCV = RandomizedSearchCV(xgbmodel, param_distributions=params)

# # early_stop = xgb.callback.EarlyStopping(rounds=2, data_name="validation_0", metric_name="mlogloss")

# # xgbRSCV.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
# xgbRSCV = xgb.train(params=xgb_dmatrix_params, dtrain=X_dmtrain, num_boost_round=1000, evals=[(X_dmtest, "test")], early_stopping_rounds=100, verbose_eval=100)


# # bestRSCV = xgbRSCV.best_estimator_

# y_pred = xgbRSCV.predict(X_dmtest)
# top3_preds = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
# top3_preds = [labelEncoder_fn.inverse_transform(row) for row in top3_preds]
# y_true_label = labelEncoder_fn.inverse_transform(y_test)
# map3 = mapk(y_true_label, top3_preds)

# print(f"Score: {map3}")


# params_grid = {
#     'eta': [0.1, 0.01],
#     'max_depth': [3, 6, 9],
#     'subsample': [0.5, 1.0],
#     'colsample_bytree': [0.5, 1.0]
# }

# def run_cv(params):
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=100,
#         nfold=5,
#         metrics='rmse',
#         as_pandas=True,
#         seed=42
#     )
    
#     return cv_results

# # Run cross-validation for each combination of parameters
# best_params = None
# best_rmse = float('inf')

# from itertools import product

# for params in list(product(*params_grid.values())):
#     param_dict = dict(zip(params_grid.keys(), params))
#     cv_results = run_cv(param_dict)
    
#     mean_rmse = cv_results['test-rmse-mean'].iloc[-1]
#     if mean_rmse < best_rmse:
#         best_rmse = mean_rmse
#         best_params = param_dict

# print(f"Best parameters: {best_params}")
# print(f"Best RMSE: {best_rmse}")

