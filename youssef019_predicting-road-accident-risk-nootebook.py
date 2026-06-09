import math
import re
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.linear_model import Lasso, Ridge


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import KFold


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor


import pickle


df_train = pd.read_csv(r"/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv(r"/kaggle/input/playground-series-s5e10/test.csv")

df_train['Type'] = 'Train'
df_test['Type'] = 'Test'
df_test['accident_risk'] = 0.0

df = pd.concat([df_train, df_test], ignore_index=True)


list(df.columns)


pd.concat([df.head(5), df.sample(5), df.tail(5)])


df.describe(include='all')


df.info()


pd.DataFrame([ df.nunique(), df.dtypes ])


TARGET = 'accident_risk'

INPUT_FEATURES = [
 'road_type',
 'num_lanes',
 'curvature',
 'speed_limit',
 'lighting',
 'weather',
 'road_signs_present',
 'public_road',
 'time_of_day',
 'holiday',
 'school_season',
 'num_reported_accidents'
]

CATEGORICAL_ORD_FEATURES = ['road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

CATEGORICAL_NOM_FEATURES = ['road_type', 'lighting', 'weather']

NUMERICAL_CON_FEATURES = ['curvature', ]

NUMERICAL_DIS_FEATURES = ['num_lanes', 'num_reported_accidents', 'speed_limit']


df[CATEGORICAL_ORD_FEATURES] = df[CATEGORICAL_ORD_FEATURES].astype(str)
df[CATEGORICAL_NOM_FEATURES] = df[CATEGORICAL_NOM_FEATURES].astype(str)


df[NUMERICAL_CON_FEATURES] = df[NUMERICAL_CON_FEATURES].astype(float)
df[NUMERICAL_DIS_FEATURES] = df[NUMERICAL_DIS_FEATURES].astype(float)

df[TARGET] = df[TARGET].astype(float)


pd.DataFrame([ df.nunique(), df.dtypes ])


100.0 * df.isna().sum() / len(df)


# df.dropna(inplace=True)


df.duplicated().sum()


# df.drop_duplicates(inplace=True)


# df.duplicated().sum()


plt.style.use(plt.style.available[11])


COLS = 3
ROWS = math.ceil(len(CATEGORICAL_ORD_FEATURES) / COLS)

for i in range(ROWS):
    plt.figure(figsize=(10,3))

    for j in range(COLS):
        idx = i * COLS + j   

        if idx >= len(CATEGORICAL_ORD_FEATURES):
            break
            
        cat_feature_i = CATEGORICAL_ORD_FEATURES[idx]

        plt.subplot(1, COLS, j + 1)
        plt.title(f"Count plot: {cat_feature_i}")
        sns.countplot(df, x=cat_feature_i)
        plt.ylabel("")
        plt.xlabel("")
        
    plt.show()


COLS = 3
ROWS = math.ceil(len(NUMERICAL_DIS_FEATURES) / COLS)

for i in range(ROWS):
    plt.figure(figsize=(10,3))

    for j in range(COLS):
        idx = i * COLS + j   

        if idx >= len(NUMERICAL_DIS_FEATURES):
            break
            
        feature_i = NUMERICAL_DIS_FEATURES[idx]

        plt.subplot(1, COLS, j + 1)
        plt.title(f"Count plot: {feature_i}")
        sns.countplot(df, x=feature_i)
        plt.ylabel("")
        plt.xlabel("")
        
    plt.show()


plt.figure(figsize=(15,5))
plt.title(f"Hist plot: {TARGET}")
sns.histplot(df[df['Type'] == 'Train'], x=TARGET, kde=True)
plt.show()


COLS = 1
ROWS = math.ceil(len(NUMERICAL_CON_FEATURES) / COLS)

for i in range(ROWS):
    plt.figure(figsize=(10,3))

    for j in range(COLS):
        idx = i * COLS + j   

        if idx >= len(NUMERICAL_CON_FEATURES):
            break
            
        feature_i = NUMERICAL_CON_FEATURES[idx]

        plt.subplot(1, COLS, j + 1)
        plt.title(f"histplot: {feature_i}")
        sns.histplot(df, x=feature_i, kde=True)
        plt.ylabel("")
        plt.xlabel("")
        
    plt.show()


LI_TEMP = NUMERICAL_CON_FEATURES

COLS = 2
ROWS = math.ceil(len(LI_TEMP) / COLS)

for i in range(ROWS):
    plt.figure(figsize=(10,3))

    for j in range(COLS):
        idx = i * COLS + j   

        if idx >= len(LI_TEMP):
            break
            
        feature_i = LI_TEMP[idx]

        plt.subplot(1, COLS, j + 1)
        plt.title(f"Scatter plot: {feature_i}")
        sns.scatterplot(data=df[df['Type'] == 'Train'], x = LI_TEMP[idx], y = TARGET)

    plt.show()


LI_TEMP = NUMERICAL_DIS_FEATURES + CATEGORICAL_NOM_FEATURES + CATEGORICAL_ORD_FEATURES

COLS = 2
ROWS = math.ceil(len(LI_TEMP) / COLS)

for i in range(ROWS):
    plt.figure(figsize=(10,3))

    for j in range(COLS):
        idx = idx = i * COLS + j   

        if idx >= len(LI_TEMP):
            break
            
        feature_i = LI_TEMP[idx]

        plt.subplot(1, COLS, j + 1)
        plt.title(f"Scatter plot: {feature_i}")
        sns.boxplot(data=df[df['Type'] == 'Train'], x = LI_TEMP[idx], y = TARGET)

    plt.show()


from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def detect_lof_outliers(df: pd.DataFrame, numeric_cols: list,new_col_name, contamination=0.01, n_neighbors=20, ):
    # Work on a copy to avoid modifying the original df
    df_out = df.copy()

    # Extract numeric data
    X = df_out[numeric_cols]

    # Initialize and fit LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_labels = lof.fit_predict(X)

    # Convert labels: -1 -> 1 (outlier), 1 -> 0 (normal)
    df_out[new_col_name] = (outlier_labels == -1).astype(str)

    return df_out


df = detect_lof_outliers(df,
                         NUMERICAL_CON_FEATURES, 
                         contamination=0.02, 
                         new_col_name='lof_outlier_v1')

CATEGORICAL_ORD_FEATURES.append('lof_outlier_v1')
INPUT_FEATURES.append('lof_outlier_v1')


# df = detect_lof_outliers(df,
#                          NUMERICAL_DIS_FEATURES, 
#                          contamination=0.02, 
#                          new_col_name='lof_outlier_v2')


# CATEGORICAL_ORD_FEATURES.append('lof_outlier_v2')
# INPUT_FEATURES.append('lof_outlier_v2')


df.columns


INPUT_FEATURES


def register_feature(f_name, list_type, f_input, f_callback):
    if f_name in INPUT_FEATURES:
        return

    if isinstance(f_input, list):
        df[f_name] = df[f_input].apply(lambda row: f_callback(row.values), axis=1)
    else:
        df[f_name] = df[f_input].apply(f_callback)

    INPUT_FEATURES.append(f_name)
    list_type.append(f_name)


register_feature(
    "curvature_M_speed_limit",
    NUMERICAL_CON_FEATURES,
    ['curvature', 'speed_limit'],
    lambda x: x[0] * x[1]
)


register_feature(
    "curvature_M_num_lanes",
    NUMERICAL_CON_FEATURES,
    ['curvature', 'num_lanes'],
    lambda x: x[0] * x[1]
)


register_feature(
    "curvature_power_2",
    NUMERICAL_CON_FEATURES,
    ['curvature'],
    lambda x: x[0] ** 2
)


register_feature(
    "num_reported_accidents_sigmoid",
    NUMERICAL_CON_FEATURES,
    ['num_reported_accidents'],
    lambda x: 1 / (1 + math.e ** (-1 * x[0] + 3.0))
)


register_feature(
    "speed_limit>= 50.0",
    NUMERICAL_CON_FEATURES,
    ['speed_limit'],
    lambda x: float(x[0] >= 50.0)
)


register_feature(
    "num_reported_accidents_>= 5.0",
    NUMERICAL_CON_FEATURES,
    ['num_reported_accidents'],
    lambda x: float(x[0] >= 5.0)
)


register_feature(
    "curvature>= 0.5",
    NUMERICAL_CON_FEATURES,
    ['curvature'],
    lambda x: float(x[0] >= 0.5)
)


df.columns


plt.title("Input Features Correlation")

sns.heatmap( 
    df[NUMERICAL_CON_FEATURES + NUMERICAL_DIS_FEATURES].corr(),
    annot=True,
    cmap='rocket_r'
    )


plt.title("Input Features vs Target Correlation")

sns.heatmap( 
    df[ [TARGET] + NUMERICAL_CON_FEATURES + NUMERICAL_DIS_FEATURES].corr().iloc[:,[0]],
    annot=True,
    cmap='rocket_r'
    )


encoded_df = df.copy()


encoded_df = pd.get_dummies(encoded_df, columns=CATEGORICAL_NOM_FEATURES, drop_first=True, dtype=int)


encoded_df = pd.get_dummies(encoded_df, columns=CATEGORICAL_ORD_FEATURES, drop_first=True, dtype=int)


min_max_scaller = MinMaxScaler()

min_max_scaller.fit(encoded_df[NUMERICAL_CON_FEATURES + NUMERICAL_DIS_FEATURES])


encoded_df.loc[:,NUMERICAL_CON_FEATURES + NUMERICAL_DIS_FEATURES] = min_max_scaller.transform(encoded_df[NUMERICAL_CON_FEATURES + NUMERICAL_DIS_FEATURES])


df.head()


encoded_df.head()


df_train = encoded_df[ encoded_df['Type']  == 'Train']
df_test = encoded_df[ encoded_df['Type']  == 'Test']


X_train,X_val, y_train, y_val = train_test_split( 
    df_train.drop([TARGET, 'Type', 'id'], axis=1), 
    df_train.loc[:, TARGET],
    test_size=0.3,
    random_state=41 
  )


X_test = df_test.drop([TARGET, 'Type', 'id'], axis=1)
X_test_idx = df_test['id']


print(f"{X_train.shape= }")
print(f"{y_train.shape= }")
print(f"{X_val.shape= }")
print(f"{y_val.shape= }")
print(f"{X_test.shape= }")
print(f"{len(X_test_idx)= }")


CACHE_MODELS_DIR_NAME = 'models_cache' 
os.makedirs(CACHE_MODELS_DIR_NAME, exist_ok=True)


class CustomModel:
    def __init__(self, name, model, extra_train_param = None ):

        self.name = str(name)
        self.model = model
        self.extra_train_param = extra_train_param

        self.y_train_hat = None
        self.y_test_hat = None
        
        self.load()


    def fit(self, x_train, y_train):
        if self.trained == False:
            
            if self.extra_train_param is None:
                self.model.fit(x_train, y_train)
            else:
                self.model.fit(x_train, y_train, **self.extra_train_param)

            self.trained = True
            self.save()
    
    def prdict_on_train(self, x_train):
        if self.y_train_hat is None:
            self.y_train_hat = self.model.predict(x_train)
    
    def prdict_on_test(self, x_test):
        if self.y_test_hat is None:
            self.y_test_hat = self.model.predict(x_test)
    
    def save(self):
        file_name = re.sub('', '', str(self.name).lower())
        file_path = CACHE_MODELS_DIR_NAME + '/' + file_name + '.pickle'
        with open(file_path, 'wb') as f:
           pickle.dump(self.model, f)        

    def load(self):
        file_name = re.sub('', '', str(self.name).lower())
        file_path = CACHE_MODELS_DIR_NAME + '/' + file_name + '.pickle'

        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)
            self.trained = True
        else:
            self.trained = False


models_list = []


# # Define search space
# search_space = {
#     'max_depth': Integer(2, 20),
#     'min_samples_split': Integer(2, 10),
#     'min_samples_leaf': Integer(1, 5),
# }

# # Define base model
# base_dt = DecisionTreeRegressor(random_state=42)

# # Bayesian optimizer
# dt_bayes = BayesSearchCV(
#     estimator=base_dt,
#     search_spaces=search_space,
#     n_iter=20,                 # number of optimization steps
#     cv=3,                      # 3-fold cross-validation
#     scoring='neg_mean_squared_error',              # or 'neg_mean_squared_error'
#     random_state=42,
#     n_jobs=-1,
#     verbose=0
# )

# # Wrap in your CustomModel
# models_list.append(CustomModel("Decision Tree Bayesian Opt v2", dt_bayes))


# # Search space for regularization strength
# search_space = {
#     'alpha': Real(1e-4, 100.0, prior='log-uniform'),
# }

# ridge = Ridge(random_state=42)

# ridge_bayes = BayesSearchCV(
#     estimator=ridge,
#     search_spaces=search_space,
#     n_iter=20,              # number of optimization trials
#     cv=3,                   # 3-fold cross validation
#     scoring='neg_mean_squared_error',           # or 'neg_mean_squared_error'
#     n_jobs=-1,
#     verbose=0,
#     random_state=42
# )

# models_list.append(CustomModel("Ridge Regression Bayesian Opt v2", ridge_bayes))


# # Define search space
# search_space = {
#     'n_estimators': Integer(10, 200),                     # number of weak learners
#     'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
#     'estimator__max_depth': Integer(1, 10),               # depth of base tree
# }

# # Define base AdaBoost model
# base_adaboost = AdaBoostRegressor(
#     estimator=DecisionTreeRegressor(),
#     random_state=42
# )

# # Define Bayesian optimizer
# adaboost_bayes = BayesSearchCV(
#     estimator=base_adaboost,
#     search_spaces=search_space,
#     n_iter=30,                  # number of optimization steps
#     cv=3,                       # 3-fold cross-validation
#     scoring='neg_mean_squared_error',               # or 'neg_mean_squared_error'
#     n_jobs=-1,
#     verbose=0,
#     random_state=42
# )

# # Wrap in your CustomModel
# models_list.append(CustomModel("AdaBoost Bayesian Opt", adaboost_bayes))


# # Define search space
# search_space = {
#     'n_estimators': Integer(50, 500),
#     'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
#     'max_depth': Integer(2, 10),
#     'min_samples_split': Integer(2, 20),
#     'min_samples_leaf': Integer(1, 10)
# }

# # Base model
# gbr = GradientBoostingRegressor(random_state=42)

# # Bayesian optimizer
# gbr_bayes = BayesSearchCV(
#     estimator=gbr,
#     search_spaces=search_space,
#     n_iter=30,                   # Number of optimization steps
#     cv=3,                        # 3-fold cross validation
#     scoring='neg_mean_squared_error',                # You can also use 'neg_mean_squared_error'
#     n_jobs=-1,
#     verbose=0,
#     random_state=42
# )

# # Add to your models list
# models_list.append(CustomModel("Gradient Boosting Bayesian Opt", gbr_bayes))


# # Define search space
# search_space = {
#     'n_estimators': Integer(100, 300),
#     'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#     'max_depth': Integer(3, 12),
#     'subsample': Real(0.5, 1.0),
#     'colsample_bytree': Real(0.5, 1.0),
#     'gamma': Real(0, 5.0),
#     'reg_alpha': Real(1e-5, 10.0, prior='log-uniform'),
#     'reg_lambda': Real(1e-5, 10.0, prior='log-uniform')
# }

# # Base XGBoost model
# xgb = XGBRegressor(
#     random_state=42,
#     objective='reg:squarederror',
#     n_jobs=-1,
#     tree_method='hist',     # faster training
#     eval_metric='rmse'
# )

# # Bayesian optimizer
# xgb_bayes = BayesSearchCV(
#     estimator=xgb,
#     search_spaces=search_space,
#     n_iter=15,                  # number of optimization trials
#     cv=3,                       # 3-fold cross-validation
#     scoring='neg_mean_squared_error',             # or 'neg_mean_squared_error'
#     n_jobs=-1,
#     verbose=0,
#     random_state=42
# )

# # Wrap in your CustomModel
# models_list.append(CustomModel("XGBoost Bayesian Opt v1", xgb_bayes))


# # Define search space for CatBoost
# search_space = {
#     'iterations': Integer(100, 600),
#     'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#     'depth': Integer(3, 10),
#     'l2_leaf_reg': Real(1, 10),
#     'bagging_temperature': Real(0.0, 1.0),
#     'border_count': Integer(32, 255)
# }

# # Base model
# cat = CatBoostRegressor(
#     loss_function='RMSE',
#     random_state=42,
#     verbose=0
# )

# # Define Bayesian optimizer
# cat_bayes = BayesSearchCV(
#     estimator=cat,
#     search_spaces=search_space,
#     n_iter=40,             # number of optimization trials
#     cv=3,
#     scoring='neg_mean_squared_error',          # or 'neg_mean_squared_error'
#     n_jobs=-1,
#     verbose=0,
#     random_state=42
# )

# # Add to list
# models_list.append(CustomModel("CatBoost Bayesian Opt", cat_bayes))


# stacked_v1 = StackingRegressor(
#     estimators=[
#         ('ridge', LinearRegression()),
#         ('tree1', DecisionTreeRegressor(max_depth=10)),
#         ('tree2', DecisionTreeRegressor(max_depth=5)),
#         ('tree3', DecisionTreeRegressor(max_depth=15))
#     ], 
#     passthrough=False,
#     final_estimator= GradientBoostingRegressor())

# models_list.append(CustomModel("Stacked Model v1", stacked_v1))





# stacked_v2 = StackingRegressor(
#     estimators=[
#         ('XGBoost_1', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)),
#         ('LightGBM_1', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1)),
#         ('CatBoost_1', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0)),
#         ('tree_1', DecisionTreeRegressor(max_depth=10))
#     ], 
#     passthrough=False,
#     final_estimator= LinearRegression())
# models_list.append(CustomModel("Stacked Model v2", stacked_v2))





# stacked_v3 = StackingRegressor(
#     estimators=[
#         ('XGBoost_1', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)),
#         ('LightGBM_1', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1)),
#         ('CatBoost_1', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0)),
#         ('tree_1', DecisionTreeRegressor(max_depth=10)),
#         ("MLP_1",  MLPRegressor(
#                         hidden_layer_sizes=(100, 50), 
#                         learning_rate_init=0.001, 
#                         max_iter=1000, 
#                         random_state=42),
#         )
#     ], 
#     passthrough=False,
#     final_estimator= LinearRegression())

# models_list.append(CustomModel("Stacked Model v3", stacked_v3))










# stacked_v4 = StackingRegressor(
#     estimators=[
#         ('XGBoost_1', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)),
#         ('LightGBM_1', LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1)),
#         ('CatBoost_1', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0)),
#         ('tree_1', DecisionTreeRegressor(max_depth=10)),
#     ], 
#     passthrough=True,
#     final_estimator= MLPRegressor(
#                         hidden_layer_sizes=(50, 20), 
#                         learning_rate_init=0.001, 
#                         max_iter=1000, 
#                         random_state=42))
# models_list.append(CustomModel("Stacked Model v4", stacked_v4))



from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import (
    LinearRegression,
    TheilSenRegressor,
    TweedieRegressor,
    QuantileRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# ==========================================================
# Define Stacked Regressor with classic + modern models
# ==========================================================
stacked_v4_opt = StackingRegressor(
    estimators=[
        ('XGBoost_1', XGBRegressor(random_state=42)),
        ('LightGBM_1', LGBMRegressor(random_state=42)),
        ('CatBoost_1', CatBoostRegressor(verbose=0, random_state=42)),
        #('Tree_1', DecisionTreeRegressor(random_state=42)),
        ('MLP_1', MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)),
        #('Tweedie_1', TweedieRegressor(power=1.5, alpha=0.1)),  # for skewed/positive data
        #('TheilSen_1', TheilSenRegressor(random_state=42)),       # robust to outliers
        ('Quantile_1', QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs'))
    ],
    final_estimator=LinearRegression(),  # could also use Ridge or LassoCV
    passthrough=False,
    n_jobs=-1
)

# ==========================================================
# Define Search Space for Bayesian Optimization
# ==========================================================
param_space = {
    # Tree-based
    'XGBoost_1__max_depth': Integer(3, 10),
    'XGBoost_1__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'LightGBM_1__num_leaves': Integer(20, 156),
    'CatBoost_1__depth': Integer(4, 10),
    #'Tree_1__max_depth': Integer(3, 12),

    # New models (safe small tuning range)
    # 'Tweedie_1__power': Real(1.0, 2.0),           # 1=Poisson, 2=Gamma
    # 'Tweedie_1__alpha': Real(1e-4, 1.0, prior='log-uniform'),  # FIXED ✅
    'Quantile_1__quantile': Real(0.1, 0.9),       # test lower/mid/upper quantiles
}

# ==========================================================
# Define Bayesian Optimizer
# ==========================================================
stacked_v4_opt_bayes = BayesSearchCV(
    estimator=stacked_v4_opt,
    search_spaces=param_space,
    n_iter=15,
    cv=KFold(5, shuffle=True, random_state=42),
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# ==========================================================
# Register Custom Model
# ==========================================================
models_list.append(CustomModel("Stacked Model v4 (Modern + Robust)", stacked_v4_opt_bayes))


# models_list.append(CustomModel(
#     "MLP v1",
#     MLPRegressor(
#         hidden_layer_sizes=(100, 50), 
#         learning_rate_init=0.001, 
#         max_iter=1000, 
#         random_state=42)
# ))


# ## Take too much time to train
# # models_list.append(CustomModel("SVC linear kernel", SVR(kernel='linear')))
# # models_list.append(CustomModel("SVC poly kernel", SVR(kernel='poly')))
# # models_list.append(CustomModel("SVC rbf kernel", SVR(kernel='rbf')))


# ## overfitting
# # models_list.append(CustomModel("Random Forest, trees=3", RandomForestRegressor(n_estimators=3)))
# # models_list.append(CustomModel("Random Forest, trees=10", RandomForestRegressor(n_estimators=10)))
# models_list.append(CustomModel("Random Forest, trees=25", RandomForestRegressor(n_estimators=25)))
# # models_list.append(CustomModel("Random Forest, trees=50", RandomForestRegressor(n_estimators=50)))
# # models_list.append(CustomModel("Random Forest, trees=100", RandomForestRegressor(n_estimators=100)))
# # models_list.append(CustomModel("Random Forest, trees=200", RandomForestRegressor(n_estimators=200)))


for i, model in enumerate(models_list):
    print(f'\n{i+1:3d}/{len(models_list)}. Train {model.name}')
    model.fit(X_train,y_train)


for i, model in enumerate(models_list):
    print(f'{i+1:3d}/{len(models_list)}. Predict {model.name} on train data')
    model.prdict_on_train(X_train)
    
    model.y_train_hat[ model.y_train_hat > 1] = 1
    model.y_train_hat[ model.y_train_hat < 0] = 0


for i, model in enumerate(models_list):
    print(f'{i+1:3d}/{len(models_list)}. Predict {model.name} on val data')
    model.prdict_on_test(X_val)

    model.y_test_hat[ model.y_test_hat > 1] = 1
    model.y_test_hat[ model.y_test_hat < 0] = 0


evaluation_dataset = []

for i, model in enumerate(models_list):
    print(f'{i+1:02d}/{len(models_list)}. Evaluate {model.name}')

    mse_train = mean_squared_error( y_train, model.y_train_hat )
    mse_val = mean_squared_error( y_val, model.y_test_hat )

    mae_train = mean_absolute_error( y_train, model.y_train_hat )
    mae_val = mean_absolute_error( y_val, model.y_test_hat )

    rmse_train = mean_squared_error( y_train, model.y_train_hat, squared=False )
    rmse_val = mean_squared_error( y_val, model.y_test_hat, squared=False)
    
    r2_score_train = r2_score( y_train, model.y_train_hat )
    r2_score_val = r2_score( y_val, model.y_test_hat )

    print(f" Model: {model.name :20s}")
    
    print('\n')

    evaluation_dataset.append(
        {"model": model.name, 'mse': mse_train, 'mae': mae_train, 'rmse' : rmse_train, 'r2_score':r2_score_train, 'data':'train'}
    )

    evaluation_dataset.append(
        {"model": model.name, 'mse': mse_val, 'mae': mae_val, 'rmse' : rmse_val, 'r2_score':r2_score_val, 'data':'val'}
    )

evaluation_dataset = pd.DataFrame(evaluation_dataset)


evaluation_dataset.sort_values('rmse')


sns.barplot(evaluation_dataset, x='model',y='rmse', hue='data')
plt.xticks(rotation = 90)
plt.show()


best_model_name = str(evaluation_dataset[evaluation_dataset['data'] == 'val'].sort_values('rmse').iloc[0,0])

best_model = list(filter(lambda x: x.name == best_model_name , models_list))[0]

print(f"Best Model Name: {best_model_name}")
print(f"{best_model}")


X_test.shape, X_test_idx.shape


Y_test = best_model.model.predict(X_test)
Y_test.shape


print(f"{Y_test.min()=}, {Y_test.max()=}")
Y_test[ Y_test > 1] = 1
Y_test[ Y_test < 0] = 0
print(f"{Y_test.min()=}, {Y_test.max()=}")


submission_df = pd.DataFrame({"id": X_test_idx, "accident_risk": Y_test})
submission_df.head()


submission_df.to_csv('submission.csv', index=False)

