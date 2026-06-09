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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


base_features = [
    "buy_qty", "sell_qty", "volume", "bid_qty", "ask_qty",
]


# XGB feature importance

def get_xgb_top_features(train_df, label_column, top_k=50):
    X = train_df.drop(columns=[label_column]).values
    y = train_df[label_column].values
    model = XGBRegressor(n_estimators=100, random_state=42, tree_method="hist")
    model.fit(X, y)
    imp = model.feature_importances_
    names = train_df.drop(columns=[label_column]).columns
    top_features = pd.DataFrame({"feature": names, "importance": imp})\
                        .sort_values("importance", ascending=False)\
                        .head(top_k)["feature"].tolist()
    print(f"XGB Importance Top-{top_k}:", top_features)
    return top_features


# corr
def get_top_corr_features(train_df, label_column, top_k=50):
    corrs = train_df.corr(numeric_only=True)[label_column].abs().sort_values(ascending=False)
    top_features = corrs.index[1:top_k+1].tolist()
    print(f"Correlation Top-{top_k}:", top_features)
    return top_features


def get_all_features(train_df, label_column):
    return [col for col in train_df.columns if col != label_column]


# try 1:
selected_features = [
    "buy_qty", "sell_qty", "volume", "bid_qty", "ask_qty",
    'X22', 'X28', 'X40', 'X52', 'X55', 'X97', 'X137', 'X138', 'X168', 'X169', 'X174', 'X175', 'X178',
    'X179', 'X180', 'X181', 'X173', 'X197', 'X198', 'X272', 'X288', 'X297', 'X302', 'X321', 'X333',
    'X338', 'X341', 'X343', 'X344', 'X345', 'X363', 'X379', 'X385', 'X386', 'X415', 'X421', 'X427',
    'X428', 'X435', 'X438', 'X444', 'X445', 'X450', 'X452', 'X459', 'X466', 'X586', 'X587', 'X593',
    'X598', 'X572', 'X603', 'X605', 'X612', 'X674', 'X680', 'X683', 'X686', 'X692', 'X695', 'X696', 'X532'
]


def add_features(df):
    data = df.copy()
    features_df = pd.DataFrame(index=data.index)
    
    features_df['bid_ask_spread_proxy'] = data['ask_qty'] - data['bid_qty']
    features_df['total_liquidity'] = data['bid_qty'] + data['ask_qty']
    features_df['trade_imbalance'] = data['buy_qty'] - data['sell_qty']
    features_df['total_trades'] = data['buy_qty'] + data['sell_qty']
    
    features_df['volume_per_trade'] = data['volume'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['buy_volume_ratio'] = data['buy_qty'] / (data['volume'] + 1e-8)
    features_df['sell_volume_ratio'] = data['sell_qty'] / (data['volume'] + 1e-8)
    
    features_df['buying_pressure'] = data['buy_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['selling_pressure'] = data['sell_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    
    features_df['order_imbalance'] = (data['bid_qty'] - data['ask_qty']) / (data['bid_qty'] + data['ask_qty'] + 1e-8)
    features_df['order_imbalance_abs'] = np.abs(features_df['order_imbalance'])
    features_df['bid_liquidity_ratio'] = data['bid_qty'] / (data['volume'] + 1e-8)
    features_df['ask_liquidity_ratio'] = data['ask_qty'] / (data['volume'] + 1e-8)
    features_df['market_depth'] = data['bid_qty'] + data['ask_qty']
    features_df['depth_imbalance'] = features_df['market_depth'] - data['volume']
    
    features_df['buy_sell_ratio'] = data['buy_qty'] / (data['sell_qty'] + 1e-8)
    features_df['bid_ask_ratio'] = data['bid_qty'] / (data['ask_qty'] + 1e-8)
    features_df['volume_liquidity_ratio'] = data['volume'] / (data['bid_qty'] + data['ask_qty'] + 1e-8)

    features_df['buy_volume_product'] = data['buy_qty'] * data['volume']
    features_df['sell_volume_product'] = data['sell_qty'] * data['volume']
    features_df['bid_ask_product'] = data['bid_qty'] * data['ask_qty']
    
    features_df['market_competition'] = (data['buy_qty'] * data['sell_qty']) / ((data['buy_qty'] + data['sell_qty']) + 1e-8)
    features_df['liquidity_competition'] = (data['bid_qty'] * data['ask_qty']) / ((data['bid_qty'] + data['ask_qty']) + 1e-8)
    
    total_activity = data['buy_qty'] + data['sell_qty'] + data['bid_qty'] + data['ask_qty']
    features_df['market_activity'] = total_activity
    features_df['activity_concentration'] = data['volume'] / (total_activity + 1e-8)
    
    features_df['info_arrival_rate'] = (data['buy_qty'] + data['sell_qty']) / (data['volume'] + 1e-8)
    features_df['market_making_intensity'] = (data['bid_qty'] + data['ask_qty']) / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['effective_spread_proxy'] = np.abs(data['buy_qty'] - data['sell_qty']) / (data['volume'] + 1e-8)
    
    lambda_decay = 0.95
    ofi = data['buy_qty'] - data['sell_qty']
    features_df['order_flow_imbalance_ewm'] = ofi.ewm(alpha=1-lambda_decay).mean()

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    return features_df


class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = []
    # train_df = pd.read_parquet(Config.TRAIN_PATH)
    # top_features = get_top_corr_features(train_df, Config.LABEL_COLUMN, top_k=50)
    # Config.FEATURES = top_features
    
    LABEL_COLUMN = "label"
    RANDOM_STATE = 42

    RIDGE_PARAMS = {'alpha': 1.0}

MODELS = [
    ("ridge", Ridge, Config.RIDGE_PARAMS),
]


def load_train_data(features):
    df = pd.read_parquet(Config.TRAIN_PATH)
    derived = add_features(df)
    df = pd.concat([df, derived], axis=1)
    X = df[Config.FEATURES].values
    y = df[Config.LABEL_COLUMN].values
    return X, y


def split_and_scale(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=Config.RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, y_train, y_val, scaler


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    corr = pearsonr(y_val, y_pred)[0]
    return {"R2": r2, "RMSE": rmse, "Corr": corr}


def load_test_data(scaler):
    test_df = pd.read_parquet(Config.TEST_PATH)
    derived = add_features(test_df)
    test_df = pd.concat([test_df, derived], axis=1)
    X_test = test_df[Config.FEATURES].values
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled


def create_submission(model, X_test_scaled, filename="submission.csv"):
    submission = pd.read_csv(Config.SUBMISSION_PATH)
    preds = model.predict(X_test_scaled)
    submission["prediction"] = preds
    submission.to_csv(filename, index=False)
    print("완")


# X, y = load_train_data()
# X_train, X_val, y_train, y_val, scaler = split_and_scale(X, y)

# for model_name, ModelClass, params in MODELS:
#     print(f"\n모델: {model_name.upper()}")
#     model = ModelClass(**params)
#     model.fit(X_train, y_train)

#     scores = evaluate_model(model, X_val, y_val)
#     print(f"{model_name.upper()} 평가 결과: {scores}")


def train_and_evaluate(model_class, model_params, feature_selector_fn, top_k=50):
    print("\n[학습/검증 시작]")
    train_df = pd.read_parquet(Config.TRAIN_PATH)

    features = feature_selector_fn(train_df, Config.LABEL_COLUMN, top_k = top_k)
    features = list(dict.fromkeys(base_features + features))
    Config.FEATURES = features

    X, y = load_train_data(features=Config.FEATURES)
    X_train, X_val, y_train, y_val, scaler = split_and_scale(X, y)

    model = model_class(**model_params)
    model.fit(X_train, y_train)

    scores = evaluate_model(model, X_val, y_val)
    print(f"평가 결과 (R2={scores['R2']:.4f}, RMSE={scores['RMSE']:.4f}, Corr={scores['Corr']:.4f})")
    return model, scaler, Config.FEATURES


def train_and_evaluate_stacking(feature_selector_fn, top_k=50):
    print("\n[Stacking 학습/검증 시작]")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    features = feature_selector_fn(train_df, Config.LABEL_COLUMN, top_k=top_k)
    features = list(dict.fromkeys(base_features + features))
    Config.FEATURES = features # + base_features

    X, y = load_train_data(features=Config.FEATURES)
    X_train, X_val, y_train, y_val, scaler = split_and_scale(X, y)

    # Base models
    ridge = Ridge(alpha=1.0, random_state=Config.RANDOM_STATE)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=Config.RANDOM_STATE, tree_method="hist")
    lgbm = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=Config.RANDOM_STATE)

    stack_model = StackingRegressor(
        estimators=[
            ('ridge', ridge),
            ('xgb', xgb),
            ('lgbm', lgbm)
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )

    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_val)
    scores = evaluate_model(stack_model, X_val, y_val)
    print(f"Stacking 평가 결과 (R2={scores['R2']:.4f}, RMSE={scores['RMSE']:.4f}, Corr={scores['Corr']:.4f})")
    return stack_model, scaler, Config.FEATURES


# selected feature 를 사용했을 때,
def train_with_selected_features(model_class, model_params, features):
    print("\n[selected feature 학습]")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    Config.FEATURES = features

    X, y = load_train_data(features=Config.FEATURES)
    X_train, X_val, y_train, y_val, scaler = split_and_scale(X, y)

    model = model_class(**model_params)
    model.fit(X_train, y_train)

    scores = evaluate_model(model, X_val, y_val)
    print(f"평가 결과 (R2={scores['R2']:.4f}, RMSE={scores['RMSE']:.4f}, Corr={scores['Corr']:.4f})")
    return model, scaler, features



def predict_and_submit(model, scaler, features, filename="submission.csv"):
    Config.FEATURES = features
    X_test = load_test_data(scaler)
    create_submission(model, X_test, filename=filename)
    print(f"완: {filename}")


model, scaler, features = train_and_evaluate(
    model_class=Ridge,
    model_params=Config.RIDGE_PARAMS,
    # feature들을 바꿀 때,
    feature_selector_fn=get_top_corr_features
    # feature_selector_fn=get_all_features,
    # top_k=50
)

predict_and_submit(model, scaler, features, filename="ridge_corr_submission.csv")
# predict_and_submit(model, scaler, features, filename="ridge_full_submission.csv")


stack_model, scaler, features = train_and_evaluate_stacking(
    feature_selector_fn=get_top_corr_features,
    top_k=50
)
predict_and_submit(stack_model, scaler, features, filename="stacking_submission.csv")


# selected_feature 사용했을 때,
# model, scaler, features = train_with_selected_features(Ridge, Config.RIDGE_PARAMS, selected_features)

# predict_and_submit(model, scaler, features, filename="ridge_selected_features.csv")




