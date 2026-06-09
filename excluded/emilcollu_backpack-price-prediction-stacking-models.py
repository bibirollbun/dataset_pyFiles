import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import shap

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

warnings.simplefilter(action='ignore', category=Warning)


df_ =  pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df = df_.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head())

    print("##################### Tail #####################")
    print(dataframe.tail())

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    numeric_cols = dataframe.select_dtypes(include=['number'])  # Select numeric columns
    print(numeric_cols.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_cols(dataframe, cat_th = 10, car_th = 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


check_df(df)


df.columns = [col.upper() for col in df.columns]


cat_cols, num_cols, cat_but_car = grab_cols(df)
num_cols = [col for col in num_cols if col not in "ID"]


for col in cat_cols:
    cat_summary(df, col)


df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot = True)


correlation_matrix(df, num_cols)


for col in cat_cols:
    target_summary_with_cat(df, "PRICE", col)


for col in num_cols:
    target_summary_with_num(df, "PRICE", col)


#Defining outlier handling functions
def outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    Q1 = dataframe[variable].quantile(q1)
    Q3 = dataframe[variable].quantile(q3)
    IQR = Q3 - Q1
    upper_limit = q3 + 1.5 * IQR
    lower_limit = q1 - 1.5 * IQR

    return upper_limit, lower_limit

def check_outlier(dataframe, variable):
    upper_limit, lower_limit = outlier_thresholds(dataframe, variable)
    outliers = dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)]
    print(f"{variable}: {len(outliers)} outliers found.")

def replace_outliers(dataframe, variable):
    upper_limit, lower_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] > upper_limit, variable] = upper_limit
    dataframe.loc[dataframe[variable] < lower_limit, variable] = lower_limit

#Defining encoder functions
def label_encoder(dataframe, binary_col):
    encoder = LabelEncoder()
    dataframe[binary_col] = encoder.fit_transform(dataframe[binary_col])

    return dataframe

def one_hot_encoder(dataframe, non_ordinal_cat_col, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=non_ordinal_cat_col, drop_first=drop_first)

    return dataframe


# NEW_WEIGHT_PER_COMPARTMENT
df["NEW_CMPRT_DENSITY"] = df["COMPARTMENTS"] / df["WEIGHT CAPACITY (KG)"]

# NEW_IS_LUXURY_MATERIAL
df["NEW_IS_LUXURY_MATERIAL"] = np.where(df["MATERIAL"].isin(["Leather", "Canvas"]), 1, 0)

# NEW_IS_PREMIUM_BAG
df["NEW_IS_PREMIUM_BAG"] = np.where((df["NEW_IS_LUXURY_MATERIAL"] == 1) & (df["LAPTOP COMPARTMENT"] == "Yes") & (df["WATERPROOF"] == "Yes"),1 ,0)

cat_cols, num_cols, cat_but_car = grab_cols(df, cat_th=10, car_th=20)
num_cols = [col for col in num_cols if col not in "ID"]


for col in cat_cols:
    cat_summary(df, col)


for col in cat_cols:
    target_summary_with_cat(df, "PRICE", col)


for col in num_cols:
    target_summary_with_num(df, "PRICE", col)


#Selecting numerical null columns
na_num_cols = [col for col in df.columns if df[col].dtypes != "O" and df[col].isnull().sum() > 0]

#Defining the function to fill null values in numerical columns with KNN imputer
def knn_imputation(dataframe, cols):
    scaler = StandardScaler()
    dataframe_scaled = pd.DataFrame(scaler.fit_transform(dataframe[cols]), columns=cols)

    #Applying the KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    dataframe_imputed = pd.DataFrame(imputer.fit_transform(dataframe_scaled), columns=cols)

    #Reversing the scaling
    dataframe[cols] = scaler.inverse_transform(dataframe_imputed)

    return dataframe

#Filling null values
knn_imputation(df, na_num_cols)

#Filling cat_cols with mode imputation
na_cat_cols = [col for col in df.columns if df[col].dtype == "O"]

for col in na_cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#Checking for outliers in numerical columns
for col in num_cols:
    check_outlier(df, col)

#Handling outliers found
for col in num_cols:
    replace_outliers(df, col)

df.isnull().sum()


binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]
ordinal_cols = ["SIZE"]

label_cols = binary_cols + ordinal_cols

for col in label_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if df[col].dtype == "O" and 10>= df[col].nunique() > 2 and col not in ordinal_cols]

def one_hot_encoder(dataframe, non_ordinal_cat_col, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=non_ordinal_cat_col, drop_first=drop_first)

    return dataframe

df = one_hot_encoder(df, ohe_cols)


num_cols = [col for col in num_cols if col not in "PRICE"]

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


def backpack_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # NEW_WEIGHT_PER_COMPARTMENT
    dataframe["NEW_CMPRT_DENSITY"] = dataframe["COMPARTMENTS"] / dataframe["WEIGHT CAPACITY (KG)"]

    # NEW_IS_LUXURY_MATERIAL
    dataframe["NEW_IS_LUXURY_MATERIAL"] = np.where(dataframe["MATERIAL"].isin(["Leather", "Canvas"]), 1, 0)

    # NEW_IS_PREMIUM_BAG
    dataframe["NEW_IS_PREMIUM_BAG"] = np.where(
        (dataframe["NEW_IS_LUXURY_MATERIAL"] == 1) & (dataframe["LAPTOP COMPARTMENT"] == "Yes") & (dataframe["WATERPROOF"] == "Yes"), 1, 0)

    cat_cols, num_cols, cat_but_car = grab_cols(dataframe)
    num_cols = [col for col in num_cols if col not in ["ID", "PRICE"]]

    # Selecting numerical null columns
    na_num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].isnull().sum() > 0]

    # Filling null values
    knn_imputation(dataframe, na_num_cols)

    # Filling cat_cols with mode imputation
    na_cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    for col in na_cat_cols:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    # Handling outliers found (no outliers so deactivated here)
    # for col in num_cols:
    #     replace_outliers(dataframe, col)

    #Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() == 2]
    ordinal_cols = ["SIZE"]

    label_cols = binary_cols + ordinal_cols

    for col in label_cols:
        label_encoder(dataframe, col)

    ohe_cols = [col for col in dataframe.columns if
                dataframe[col].dtype == "O" and 10 >= dataframe[col].nunique() > 2 and col not in ordinal_cols]

    df = one_hot_encoder(dataframe, ohe_cols)

    # Scaling
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Converting True/False values to 1 and 0 for best practice
    df = df.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)

    df["PRICE"] = np.log1p(df["PRICE"])
    y = df["PRICE"]
    X = df.drop(["PRICE", "ID"], axis=1)

    return X, y

####################################################################################################

def backpack_test_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # NEW_WEIGHT_PER_COMPARTMENT
    dataframe["NEW_CMPRT_DENSITY"] = dataframe["COMPARTMENTS"] / dataframe["WEIGHT CAPACITY (KG)"]

    # NEW_IS_LUXURY_MATERIAL
    dataframe["NEW_IS_LUXURY_MATERIAL"] = np.where(dataframe["MATERIAL"].isin(["Leather", "Canvas"]), 1, 0)

    # NEW_IS_PREMIUM_BAG
    dataframe["NEW_IS_PREMIUM_BAG"] = np.where(
        (dataframe["NEW_IS_LUXURY_MATERIAL"] == 1) & (dataframe["LAPTOP COMPARTMENT"] == "Yes") & (dataframe["WATERPROOF"] == "Yes"), 1, 0)

    cat_cols, num_cols, cat_but_car = grab_cols(dataframe)
    num_cols = [col for col in num_cols if col not in "ID"]

    # Selecting numerical null columns
    na_num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].isnull().sum() > 0]

    # Filling null values
    knn_imputation(dataframe, na_num_cols)

    # Filling cat_cols with mode imputation
    na_cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    for col in na_cat_cols:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    # Handling outliers found (no outliers so deactivated here)
    # for col in num_cols:
    #     replace_outliers(dataframe, col)

    #Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O" and dataframe[col].nunique() == 2]
    ordinal_cols = ["SIZE"]

    label_cols = binary_cols + ordinal_cols

    for col in label_cols:
        label_encoder(dataframe, col)

    ohe_cols = [col for col in dataframe.columns if
                dataframe[col].dtype == "O" and 10 >= dataframe[col].nunique() > 2 and col not in ordinal_cols]

    df = one_hot_encoder(dataframe, ohe_cols)

    # Scaling
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Converting True/False values to 1 and 0 for best practice
    df = df.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)

    X = df.drop("ID", axis=1)

    return X


df = df_.copy()

X, y = backpack_data_prep(df)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)


xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
lgb_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
cat_model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, random_seed=42, verbose=0)

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgb_model),
        ('cat', cat_model)
    ],
    final_estimator=LinearRegression(),  # Meta-model (takes base model predictions)
    passthrough=True  # Use original features along with model predictions
).fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Stacked Model RMSE: {rmse:.4f}")


def optimize_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }
    model = XGBRegressor(**params)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -score.mean()

# Run optimization
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(optimize_xgb, n_trials=20, show_progress_bar=True)

# Best params for XGBoost
best_xgb_params = study_xgb.best_params


def optimize_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }

    model = LGBMRegressor(**params)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -score.mean()

# Run optimization
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(optimize_lgb, n_trials=20, show_progress_bar=True)

# Best params for LightGBM
best_lgb_params = study_lgb.best_params


def optimize_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42
    }

    model = CatBoostRegressor(**params, verbose=0)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -score.mean()

# Run optimization
study_cat = optuna.create_study(direction="minimize")
study_cat.optimize(optimize_cat, n_trials=20, show_progress_bar=True)

# Best params for CatBoost
best_cat_params = study_cat.best_params


xgb_best_model = XGBRegressor(**best_xgb_params).fit(X_train, y_train)
lgb_best_model = LGBMRegressor(**best_lgb_params).fit(X_train, y_train)
cat_best_model = CatBoostRegressor(**best_cat_params, verbose=0).fit(X_train, y_train)

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_best_model),
        ('lgbm', lgb_best_model),
        ('cat', cat_best_model)
    ],
    final_estimator=LinearRegression(),  # Meta-model (takes base model predictions)
    passthrough=True  # Use original features along with model predictions
).fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Stacked Model RMSE: {rmse:.4f}")


cv_results = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring = "neg_mean_squared_error")

cv_results = -cv_results
print(cv_results.mean())

