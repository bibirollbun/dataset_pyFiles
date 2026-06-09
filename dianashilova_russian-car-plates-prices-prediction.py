import pandas as pd
import numpy as np
from supplemental_english import *
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")


train_sales_df = pd.read_csv(r'/kaggle/input/russian-car-plates-prices-prediction/train.csv')


train_sales_df.info()
train_sales_df.describe()


train_sales_df.head()


def prepare_data(df):
    df = df.sort_values(by='date')
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["year"] = df["date"].dt.year.astype('int64')
    df["month"] = df["date"].dt.month.astype('int64')
    df["day"] = df["date"].dt.day.astype('int64')
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["day_of_week"] = df["date"].dt.dayofweek.astype('int64')
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype('int64')  
    df["is_start_of_month"] = (df["day"] == 1).astype('int64')

    df["plate_number"] = df["plate"].apply(lambda plate: plate[1:4]).astype(str)
    df["plate_series"] = df["plate"].apply(lambda plate: plate[0] + plate[4:6]).astype(str)

    df['region'] = df['plate'].str.extract(r"(\d{2,3})$")[0].astype(str)

    df = df.sort_values(by=['region', 'date'])

    df['price'] = np.log1p(df['price'])

    df['is_palindrome_letters'] = df['plate_series'].apply(lambda x: x == x[::-1])
    df['is_palindrome_numbers'] = df['plate_number'].apply(lambda x: x == x[::-1])
    df['unique_letters_count'] = df['plate_series'].apply(lambda x: len(set(x)))
    df['unique_numbers_count'] = df['plate_number'].apply(lambda x: len(set(x)))
    df["has_repeating_digits"] = df["plate_number"].apply(lambda x: len(set(x)) < len(x))

    digits = df['plate_number'].apply(lambda s: list(map(int, s)))
    df[['d1', 'd2', 'd3']] = pd.DataFrame(digits.tolist(), index=df.index)
    df['sum_numbers'] = df['d1'] + df['d2'] + df['d3']
    df['product_numbers'] = df['d1'] * df['d2'] * df['d3']

    df = df.drop(columns=['id', 'plate', 'date', 'd1', 'd2', 'd3', 'sum_numbers'])

    return df

def get_region_code(plate):
    region_code = str(int(plate[6:]))
    for region, codes in REGION_CODES.items():
        if region_code in codes:
            return region
    return "Unknown"

def prepare_df(df):
    df = prepare_data(df)

    return df


def convert_data(df):
    le = LabelEncoder()

    for col in [
                'region', 
                'is_palindrome_letters', 
                'is_palindrome_numbers', 
                'has_repeating_digits', 
                'plate_number', 
                'plate_series'
                ]:
        df[col] = le.fit_transform(df[col])
        
    return df


def plot_corr_matrix(sales):
    plt.figure(figsize=(20,20))
    sns.heatmap(sales.corr(), annot=True, cmap="mako", fmt='.3f')
    plt.show()


def split_data(sales):
    X = sales.drop(columns=['price'])

    y = sales['price']

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


train_sales = prepare_df(train_sales_df)
train_sales = convert_data(train_sales)
train_sales = train_sales.dropna()

plot_corr_matrix(train_sales)

x_train, x_test, y_train, y_test = split_data(train_sales)


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 1, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
#         "subsample": trial.suggest_float("subsample", 0.2, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
#         "tree_method": "auto",  
#         "n_jobs": -1,
#         "random_state": 42,
#     }

#     model = XGBRegressor(**params)
#     score = cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)
#     return -score.mean()

# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=100, timeout=600)

# print("\nBest trial:")
# print(study.best_trial)

# best_model = XGBRegressor(**study.best_params, tree_method="auto", n_jobs=-1, random_state=42)
# best_model.fit(x_train, y_train)
# preds = best_model.predict(x_test)
# rmse = mean_squared_error(y_test, preds)

# print(f"Validation RMSE: {rmse:.4f}")


params={'n_estimators': 738, 
        'max_depth': 8, 
        'learning_rate': 0.04617186074132262, 
        'subsample': 0.8299455281377266, 
        'colsample_bytree': 0.7378508474001586, 
        'gamma': 0.13271078122997157, 
        'min_child_weight': 5, 
        'reg_alpha': 2.063511174604212, 
        'reg_lambda': 1.4099841043286725}


# fig_1 = optuna.visualization.plot_slice(study)
# fig_1.update_layout(template='plotly_dark', title='<b>Slice Plot', title_x=0.2)


# fig_2 = optuna.visualization.plot_optimization_history(study)
# fig_2.update_layout(template='plotly_dark', title='<b>Optimization History Plot', title_x=0.5)


# fig_3 = optuna.visualization.plot_param_importances(study)
# fig_3.update_layout(template='plotly_dark', title='<b>Hyperparameter Importances', title_x=0.5)


def fit_predict_model(x_train, x_test, y_train, y_test):
    model_xgb = XGBRegressor(**params)

    model_xgb.fit(x_train, y_train)

    pred_test_xgb = model_xgb.predict(x_test)

    mae_test = mean_absolute_error(y_test, pred_test_xgb)
    mse_test = mean_squared_error(y_test, pred_test_xgb)

    print(f'MAE Test {mae_test:.2f}\n')
    print(f'MSE Test {mse_test:.2f}\n')

    print(f'Test R2 {r2_score(y_test, pred_test_xgb):.4f}\n')

    print(f'Feature importances:\n{model_xgb.feature_importances_}')

    return model_xgb, pred_test_xgb


def plot_feature_importance(model):
    FEATURES = train_sales.drop(columns=[
         'price'
        ]).columns

    plt.figure(figsize=(16,6))
    imp = model.feature_importances_
    plt.barh(FEATURES, imp)
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def smape(y_true, y_pred):
    mean = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff  = np.abs(y_true - y_pred)
    
    return np.mean(diff / mean) * 100


def predict_score(pred_test_xgb, y_test):
    y_val_pred = np.expm1(pred_test_xgb)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_val_pred)
    smape_score = smape(y_true, y_val_pred)

    print(f'MAE: {mae}\n')
    print(f'SMAPE: {smape_score}')


model, predict_values = fit_predict_model(x_train, x_test, y_train, y_test)

plot_feature_importance(model)

predict_score(predict_values, y_test)


model.fit(x_test, y_test)


test_sales_df = pd.read_csv(r'/kaggle/input/russian-car-plates-prices-prediction/test.csv')
test_sales_df.head()


test_sales_df.info()


test_sales = prepare_df(test_sales_df)
test_sales = convert_data(test_sales)
test_sales = test_sales.drop(columns=['price'])

predict_values_test = model.predict(test_sales)

values = np.expm1(predict_values_test)


submission = pd.DataFrame({'id': test_sales_df['id'], 'price': values})
submission.head()


submission.to_csv('submission.csv',index=False)

