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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

plt.style.use('dark_background')


class BackpackForcaster:

    train_data = "/kaggle/input/playground-series-s5e2/train.csv"
    test_data = "/kaggle/input/playground-series-s5e2/test.csv"

    barColor = "#212121"
    
    ### Get the count of all the unique values in the categorical columns
    ### Parameters:- dataframe: Dataframe in question, categorical_col: list of all the categporical columns names.
    def categorical_counts(self, dataframe, categorical_cols: []):
        columns = ["Column_Name", "Counts"]
        data_arr = []
        
        for col in categorical_cols:
            data_arr.append((col, dataframe[col].value_counts().count()))

        resultant_df = pd.DataFrame(data_arr, columns = columns)
        print(resultant_df.to_markdown())

    def findNullCol(self, df):
        return df.columns[df.isnull().any()].tolist()
        
        
    ### Fill the null values in the dataset
    ### Parameters:- df: Dataframe in question, col: column which has missing values
    def fill_na(self, df, cols: []):
        for col in cols:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df
            
            
    ### Draws a count plot of categorical columns
    ### Parameters:- df: Dataframe in question, r,c: numner of rows and columns respectively, categorical_cols: list of all the categporical columns names.
    def draw_count_plot(self, df,r,c,categorical_cols = []):
        plt.figure(figsize=(15,15))
        i=1
        for col in categorical_cols:
            plt.subplot(r,c,i)
            plt.title(col)
            df[col].value_counts(sort = True).plot(kind = "bar", color = self.barColor)
            
            plt.xticks(rotation = 90)
            i+=1
        plt.tight_layout()
        plt.show()
        
    ### Draws a hist plot of numerical columns
    ### Parameters:- df: Dataframe in question, r,c: numner of rows and columns respectively, numerical_cols: list of all the numerical columns names.
    def draw_hist_plot(self, df,r,c,numerical_cols = []):
        plt.figure(figsize=(14,12))
        i=1
        for col in numerical_cols:
            plt.subplot(r,c,i)
            plt.title(col)
            plt.hist(df[col], color = self.barColor, edgecolor = "white")
            plt.xticks(rotation = 0)
            i+=1
        plt.tight_layout()
        plt.show()
        
    ### Draws a box plot of numerical columns
    ### Parameters:- df: Dataframe in question, r,c: numner of rows and columns respectively, numerical_cols: list of all the numerical columns names.
    def draw_box_plot(self, df,r,c,numerical_cols = []):
        plt.figure(figsize=(14,12))
        i=1
        for col in numerical_cols:
            plt.subplot(r,c,i)
            plt.title(col)
            sns.boxplot(df[col])
            plt.xticks(rotation = 0)
            i+=1
        plt.tight_layout()
        plt.show()
        
        
    ### Draws a heatmap of correlation of the data.
    ### Parameters:- dataframe: Dataframe in question.
    def heatmap(self, dataframe):
        corr = dataframe.corr()
        fig,ax= plt.subplots(figsize=(20,20))
        sns.heatmap(corr,annot=True,ax=ax)
        plt.show()

    def date_feature_engineering(self, df):
        pass
        
    def encode_categorical_data(self, dataframe, categorical_cols):
        le = LabelEncoder()
        for col in categorical_cols:
            if col in dataframe.columns:
                dataframe[col] = le.fit_transform(dataframe[col])
        return dataframe
    
    def encoding_numerical_columns(self, dataframe, cols):
        scaler = StandardScaler()
        dataframe[cols] = scaler.fit_transform(dataframe[cols])
        return dataframe
        
        
backpackForcaster = BackpackForcaster()


train = pd.read_csv(backpackForcaster.train_data)
train.head()


test = pd.read_csv(backpackForcaster.test_data)
test.head()


train.shape


train.info()


null_cols = backpackForcaster.findNullCol(train)
print(null_cols)


train = backpackForcaster.fill_na(train, null_cols)


train.duplicated().sum()


categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_cols = ["Compartments", "Weight Capacity (kg)", "Price"]


train[numerical_cols].describe([0.25, 0.5, 0.9])


backpackForcaster.draw_count_plot(train, 3, 3, categorical_cols)


backpackForcaster.draw_hist_plot(train, 2,2, numerical_cols)


backpackForcaster.draw_box_plot(train, 2,2, numerical_cols)


train = backpackForcaster.encode_categorical_data(train, categorical_cols)
train.head()


numerical_cols.remove("Price")
train = backpackForcaster.encoding_numerical_columns(train, numerical_cols)
train.head()


x = train.drop(["id", "Price"], axis = 1)
y = train["Price"]
x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size = 0.2,random_state = 2)


train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)


params = {
    "objective": "regression",
    "boosting_type": "rf",
    "num_leaves": 8,
    "force_row_wise": True,
    "learning_rate": 0.03,
    "metric": "rmse",
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8
}
# Train the LightGBM model
num_round = 500
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])


def root_mean_squared_error(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


import optuna
import lightgbm as lgb

# Set random seed
SEED = 42

# Define the Optuna objective function
def lgbm_objective(trial):
    lgbm_params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.3),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1.0),
        "random_state": SEED,
        "verbose": -1,
    }

    # Initialize model with the parameters suggested by Optuna
    lgbm_model = LGBMRegressor(**lgbm_params)
    
    # Fit the model
    lgbm_model.fit(x_train, y_train)

    # Predictions
    y_pred = np.expm1(lgbm_model.predict(x_test))

    # Return MAPE as the objective metric
    return root_mean_squared_error(y_test, y_pred)


study = optuna.create_study(direction='minimize', study_name="LGBM Hyperparameter Tuning")
study.optimize(lgbm_objective, n_trials= 50)


print("Best hyperparameters: ", study.best_params)
print("Best RSME: ", study.best_value)


# Train final model with best parameters
best_params = study.best_params
# best_params['n_estimators'] = 2000  # Keep as per original setting
final_model = LGBMRegressor(**best_params, random_state=SEED, verbose=-1)
final_model.fit(x_train, y_train)


test.head()


test = backpackForcaster.encode_categorical_data(test, categorical_cols)
test.head()


test = backpackForcaster.encoding_numerical_columns(test, numerical_cols)
test.head()


test_data = test.drop(["id"], axis = 1)
predictions = final_model.predict(test_data)


predictions


result_df = pd.DataFrame({
    'id': test['id'],
    'loan_status': predictions
})


result_df.to_csv("submission_lgb_optuna.csv", index = False)




