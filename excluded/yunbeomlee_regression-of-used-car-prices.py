!pip install -U autogluon


from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import re
train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
Original = pd.read_csv('/kaggle/input/used-car-price-prediction-dataset/used_cars.csv')
Original[['milage', 'price']] = Original[['milage', 'price']].map(
    lambda x: int(''.join(re.findall(r'\d+', x)))
)
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor

import random

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from autogluon.tabular import TabularPredictor
# 다시 불러온거라 생략해도됨


train.shape


train.info()


train.describe()


train.isnull().sum().sort_values


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train['price'],kde=True)
plt.title('Price Distribution of Used Cars')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.xlim(-100000,500000)
plt.show()


train['price_log'] = np.log(train['price'] + 1)

sns.distplot(train['price_log'], kde=True)
plt.title('Log-Transformed Price Distribution of Used Cars')
plt.xlabel('Log-Transformed Price')
plt.ylabel('Frequency')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

numeric_df = train.select_dtypes(include=['number'])
categorical_df = train.select_dtypes(include=['object'])

encoder = OrdinalEncoder()
categorical_encoded = pd.DataFrame(encoder.fit_transform(categorical_df), columns=categorical_df.columns)

full_df = pd.concat([numeric_df, categorical_encoded], axis=1)

correlation_matrix = full_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
Original = pd.read_csv('/kaggle/input/used-car-price-prediction-dataset/used_cars.csv')
Original[['milage', 'price']] = Original[['milage', 'price']].map(
    lambda x: int(''.join(re.findall(r'\d+', x)))
)

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

train = pd.concat([train, Original], ignore_index=True)


def extract_age_features(df):
    current_year = 2024

    df['Vehicle_Age'] = current_year - df['model_year']

    df['Mileage_per_Year'] = df['milage'] / df['Vehicle_Age']
    df['milage_with_age'] = df.groupby('Vehicle_Age')['milage'].transform('mean')

    df['Mileage_per_Year_with_age'] = df.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')

    return df

def extract_other_features(df):

    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)


    return df

train = extract_age_features(train)
test = extract_age_features(test)

train = extract_other_features(train)
test = extract_other_features(test)


def update(df):

    t = 100

    cat_c = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    re_ = ['model', 'engine', 'transmission', 'ext_col', 'int_col']

    for col in re_:
        df.loc[df[col].value_counts(dropna=False)[df[col]].values < t, col] = "noise"

    for col in cat_c:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')

    return df

train = update(train)
test = update(test)

X = train.drop('price', axis = 1)
y = train['price']


import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


callbacks = [log_evaluation(period=300), early_stopping(stopping_rounds=200)]

cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"cat_cols--------{cat_cols}")


def get_MAE_oof(df, target, lgb_params, cat_params=None, model_type='LGBM'):


    oof_predictions = np.zeros(len(df))
    kf = KFold(n_splits=5, shuffle=True, random_state = 1)
    models = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"Training fold {fold + 1}/{5} with {model_type}")

        X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        if model_type == 'LGBM':
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference = train_data)

            model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )

        elif model_type == 'CAT':
            train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
            val_data = Pool(data=X_val, label = y_val, cat_features=cat_cols)

            model = CatBoostRegressor(**cat_params)
            model.fit(train_data, eval_set=val_data, verbose = 150, early_stopping_rounds=200)

        models.append(model)

        if model_type == 'LGBM':
            pred = model.predict(X_val, num_iteration=model.best_iteration)
        elif model_type == 'CAT':
            pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, pred))
        rmse_scores.append(rmse)

        print(f'{model_type} Fold RMSE: {rmse}')

        oof_predictions[val_idx] = pred

    print(f'Mean RMSE: {np.mean(rmse_scores)}')
    return oof_predictions, models


lgb_params = {
    'objective' : 'MAE',
    'n_estimators' : 1000,
    'random_state': 1,
}

oof_predictions_lgbm, models_lgbm = get_MAE_oof(X, y, lgb_params, model_type = 'LGBM')
X['LGBM_MAE'] = oof_predictions_lgbm


LGBM_preds = np.zeros(len(test))
for model in models_lgbm:
    LGBM_preds += model.predict(test) / len(models_lgbm)
test['LGBM_MAE'] = LGBM_preds



lgb_params = {
    'objective': 'MSE',
    'n_estimators' : 1000,
    'random_state': 1,
}

oof_predictions_lgbm, models_lgbm = get_MAE_oof(X, y, lgb_params, model_type='LGBM')

X['LGBM_MSE_diff'] = oof_predictions_lgbm - X['LGBM_MAE']


LGBM_preds = np.zeros(len(test))
for model in models_lgbm:
    LGBM_preds += model.predict(test) / len(models_lgbm)
test['LGBM_MSE_diff'] = LGBM_preds - test['LGBM_MAE']

test.head()



X['price'] = y

predictor = TabularPredictor(label='price',
                            eval_metric = 'rmse',
                            problem_type = 'regression').fit(X,
                                                            presets = 'best_quality',
                                                            time_limit = 3600 * 1,
                                                            verbosity=2,
                                                            num_gpus=0,
                                                            included_model_types = ['GBM', 'CAT']
                                                            )


y_pred = predictor.predict(test)

sub_blend = pd.read_csv('/kaggle/input/top-5-blended-car-prices/submission_9.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')
sample_sub['price'] = y_pred * 0.55 + sub_blend['price'] * 0.45
sample_sub.to_csv("submission.csv", index=False)
sample_sub.head()




