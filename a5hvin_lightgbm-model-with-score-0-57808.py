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

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

plt.style.use('dark_background')


class StickerSalesForcaster:

    train_data = "/kaggle/input/playground-series-s5e1/train.csv"
    test_data = "/kaggle/input/playground-series-s5e1/test.csv"
    
    ### Get the count of all the unique values in the categorical columns
    ### Parameters:- dataframe: Dataframe in question, categorical_col: list of all the categporical columns names.
    def categorical_counts(self, dataframe, categorical_cols: []):
        columns = ["Column_Name", "Counts"]
        data_arr = []
        
        for col in categorical_cols:
            data_arr.append((col, dataframe[col].value_counts().count()))

        resultant_df = pd.DataFrame(data_arr, columns = columns)
        print(resultant_df.to_markdown())
        
        
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
        plt.figure(figsize=(12,10))
        i=1
        for col in categorical_cols:
            plt.subplot(r,c,i)
            plt.title(col)
            df[col].value_counts(sort = True).plot(kind = "bar", color = "#212121")
            
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
            plt.hist(df[col], color = "#212121", edgecolor = "white")
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
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.day_name()
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365.0)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0)
        df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)

        df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
        return df
        
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
        
        
salesForecaster = StickerSalesForcaster()


train_df = pd.read_csv(salesForecaster.train_data)
train_df.sample(5)


train_df = salesForecaster.date_feature_engineering(train_df)


train_df.info()


categorical_col = ["country", "store", "product", "day_of_week"]
numerical_col = ["num_sold", "year", "quarter", "month", "day", "week_of_year", "day_sin", "day_cos", "month_sin", "month_cos", "year_sin", "year_cos", "quarter_sin", "quarter_cos", "group"]


len(categorical_col) + len(numerical_col)


train_df['num_sold'] = np.log1p(train_df['num_sold'])


train_df.sample(5)


salesForecaster.categorical_counts(train_df, categorical_col)


train_df["country"].value_counts()


train_df.isnull().sum()


null_cols = train_df.columns[train_df.isnull().any()].tolist()
print(null_cols)


train_df.duplicated().sum()


train_df = salesForecaster.fill_na(train_df, null_cols)
train_df.sample(5)


train_df.isnull().sum()


train_df.describe()


cols = categorical_col.copy()
print(cols)

salesForecaster.draw_count_plot(train_df, 2, 2, cols)


rows = math.ceil(len(numerical_col) / 3)
salesForecaster.draw_hist_plot(train_df,rows,3, numerical_col)


rows = math.ceil(len(numerical_col) / 3)
salesForecaster.draw_box_plot(train_df, rows,3, numerical_col)


train_df = salesForecaster.encode_categorical_data(train_df, categorical_col)


train_df.head()


salesForecaster.heatmap(train_df)


x = train_df.drop(['num_sold', "date", "id"], axis=1)
y = train_df['num_sold']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


test_df = pd.read_csv(salesForecaster.test_data)
test_df.head()


### Feature Engineering of the Test data
test_df = salesForecaster.date_feature_engineering(test_df)

### Encoding Categorical columns of the Test data
test_df = salesForecaster.encode_categorical_data(test_df, categorical_col)


test = test_df.drop(["id", "date"], axis = 1)


paramsxgb = {
    'n_estimators': 702, 
    'learning_rate': 0.01316466004260925, 
    'max_depth': 6, 
    'min_child_weight': 5, 
    'subsample': 0.7085976110203339, 
    'colsample_bytree': 0.9306214290853707, 
    'gamma': 0.0006666226876864524, 
    'reg_alpha': 4.783736210532281, 
    'reg_lambda': 5.162091591648421
}


# Define MAPE metric
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# Cross-validation for XGBRegressor
def cross_val_xgbr_mape(X, y, test, n_splits=6, **paramsxgb):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mape_scores = []
    preds = []

    for train_index, valid_index in kf.split(X):
        # Ensure data types for indexing
        if isinstance(X, pd.DataFrame):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        else:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

        # Initialize and train the model
        model = XGBRegressor(random_state=42, **paramsxgb)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_valid)
        score = mape(y_valid, y_pred)
        mape_scores.append(score)

        # Predict on the test set
        preds.append(model.predict(test))

    # Average predictions over all folds
    test_preds_mean = np.mean(preds, axis=0)

    return np.mean(mape_scores), test_preds_mean

# print(f"Average MAPE across folds: {average_mape:.4f}")


average_mape, xgb_preds = cross_val_xgbr_mape(x, y, test, n_splits=5, **paramsxgb)


# Save predictions for submission
submission = pd.DataFrame({'id': test_df['id'], 'num_sold': np.expm1(xgb_preds).round()})
print(submission.head())
submission.to_csv('submission_xgb1.csv', index=False)


train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)


params = {
    "objective": "mape",
    "boosting_type": "gbdt",
    "metric": "mape",
    "n_estimators": 1000,
    "verbosity": -1,
    "bagging_freq": 1,
    "learning_rate": 0.034276812942016774, 
    "num_leaves": 850, 
    "subsample": 0.9921322730910768, 
    "colsample_bytree": 0.9868424650530208, 
    "min_data_in_leaf": 29
}

# Train the LightGBM model
num_round = 500
lgb_model = lgb.train(params, train_data, num_round, valid_sets=[test_data])


testData = test_df.drop(columns = ["id", "date"], axis = 1)
predictions = lgb_model.predict(testData)


# Save predictions for submission
submission = pd.DataFrame({'id': test_df['id'], 'num_sold': np.expm1(predictions).round()})
print(submission.head())
submission.to_csv('submission_lgboptuna.csv', index=False)


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
        "metric": "mape",
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1.0),
        "random_state": SEED,
        "verbose": -1,
        "device": "gpu"
    }

    # Initialize model with the parameters suggested by Optuna
    lgbm_model = LGBMRegressor(**lgbm_params)
    
    # Fit the model
    lgbm_model.fit(x_train, y_train)

    # Predictions
    y_pred = np.expm1(lgbm_model.predict(x_test))

    # Return MAPE as the objective metric
    return mean_absolute_percentage_error(np.expm1(y_test), y_pred)


# Run Optuna optimization
study = optuna.create_study(direction='minimize', study_name="LGBM Hyperparameter Tuning")
study.optimize(lgbm_objective, n_trials= 50)



# Display the best hyperparameters and the best value
print("Best hyperparameters: ", study.best_params)
print("Best MAPE: ", study.best_value)


# Train final model with best parameters
best_params = study.best_params
# best_params['n_estimators'] = 2000  # Keep as per original setting
final_model = LGBMRegressor(**best_params, random_state=SEED, verbose=-1)
final_model.fit(x_train, y_train)


final_preds = np.expm1(final_model.predict(x_test))
final_mape = mean_absolute_percentage_error(np.expm1(y_test), final_preds)
print(f"Final Model MAPE: {final_mape}")


test = test_df.drop(["id", "date"], axis = 1)


test_preds = np.expm1(final_model.predict(test))


test_preds


submission = pd.DataFrame({'id': test_df['id'], 'num_sold': test_preds.round()})
print(submission.head())


submission.to_csv('submission_lgboptuna_2.csv', index=False)


# Set random seed
SEED = 42

# Define the Optuna objective function
def xgb_objective(trial):
    xgb_params = {
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "max_depth": trial.suggest_int("max_depth", 4, 25),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
        "objective": "reg:squarederror",
        "random_state": SEED,
        "device": "gpu"
        "tree_method"='gpu_hist',
        "predictor"='gpu_predictor',
        "eval_metric"='mape'
    }

    # Initialize model with the parameters suggested by Optuna
    xgb_model = XGBRegressor(**xgb_params)
    
    # Fit the model
    xgb_model.fit(x_train, y_train)

    # Predictions
    y_pred = np.expm1(xgb_model.predict(x_test))

    # Return MAPE as the objective metric
    return mean_absolute_percentage_error(np.expm1(y_test), y_pred)


xgb_study = optuna.create_study(direction='minimize', study_name="XGB Hyperparameter Tuning")
xgb_study.optimize(xgb_objective, n_trials= 50,)



# Train final model with best parameters
best_params = xgb_study.best_params
# best_params['n_estimators'] = 2000  # Keep as per original setting
xgb_model = XGBRegressor(**best_params, random_state=SEED)
xgb_model.fit(x_train, y_train)


test_preds = np.expm1(xgb_model.predict(test))


submission = pd.DataFrame({'id': test_df['id'], 'num_sold': test_preds.round()})
print(submission.head())


submission.to_csv('submission_xgboptuna.csv', index=False)


meta_model = LinearRegression()

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', final_model)
    ],
    final_estimator=meta_model,
    n_jobs=-1
)


stacking_model.fit(x_train, y_train)


predictions = np.expm1(stacking_model.predict(test))


predictions


submission = pd.DataFrame({'id': test_df['id'], 'num_sold': predictions.round()})
print(submission.head())


submission.to_csv('submission_stack.csv', index=False)

