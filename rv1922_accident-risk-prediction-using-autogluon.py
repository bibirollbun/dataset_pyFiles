%%time
!pip install autogluon==1.1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from autogluon.tabular import TabularPredictor
import os
import sklearn
print("AutoGluon ready, sklearn version:", sklearn.__version__)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


if train.duplicated().any():
    train = train.drop_duplicates()


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
target_col = 'accident_risk'


for c in bool_cols:
    if c in train.columns or test.columns:
        train[c] = train[c].astype(int)
        test[c] = test[c].astype(int)


encoder = TargetEncoder(cols=cat_cols)
train[cat_cols] = encoder.fit_transform(train[cat_cols], train[target_col])
test[cat_cols] = encoder.transform(test[cat_cols])


def feature_engineering(df):
    df = df.copy()

    # --- Core transforms ---
    if 'num_reported_accidents' in df:
        df['accidents_log'] = np.log1p(df['num_reported_accidents'])
        df['accidents_sqrt'] = np.sqrt(df['num_reported_accidents'])

    if 'curvature' in df:
        df['curvature_squared'] = df['curvature'] ** 2
        df['curv_log'] = np.log1p(df['curvature'])
        df['curv_inv'] = 1 / (df['curvature'] + 1e-5)

    if 'speed_limit' in df:
        df['speed_log'] = np.log1p(df['speed_limit'])
        df['inv_speed'] = 1 / (df['speed_limit'] + 1)

    if 'num_lanes' in df:
        df['lanes_log'] = np.log1p(df['num_lanes'])
        df['lanes_inv'] = 1 / (df['num_lanes'] + 1)

    # --- Targeted interactions (validated by importance > 0.001) ---
    if {'speed_limit', 'curvature'} <= set(df.columns):
        df['speed_x_curvature'] = df['speed_limit'] * df['curvature']
        df['danger_score'] = (df['speed_limit'] / 100) * (df['curvature'] ** 2)

    if {'weather', 'lighting'} <= set(df.columns):
        df['env_risk'] = df['weather'] * df['lighting']

    if {'weather', 'curvature'} <= set(df.columns):
        df['weather_curv_interact'] = df['weather'] * df['curvature']

    if {'weather', 'speed_limit'} <= set(df.columns):
        df['weather_speed_interact'] = df['weather'] * df['speed_limit']

    if {'env_risk', 'curvature'} <= set(df.columns):
        df['total_env_exposure'] = df['env_risk'] * df['curvature']

    # --- Temporal encodings ---
    if 'time_of_day' in df:
        df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)

    # --- Structural / categorical simplifications ---
    if 'num_lanes' in df:
        df['tight_lane'] = (df['num_lanes'] <= 2).astype(int)
    if 'curvature' in df:
        df['sharp_curve'] = (df['curvature'] > 0.6).astype(int)

    # --- Model handles NaN natively ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


train = feature_engineering(train)
test = feature_engineering(test)


train.head()


features = [col for col in train.columns if col not in ['id', target_col]]
train_data = train[features + [target_col]].copy()
test_data = test[features].copy()


save_path = 'saved_models'
random_state = 42
time_limit = 3600


assert target_col in train_data.columns, "Target column missing in train_data"

if os.path.exists(save_path):
    import shutil
    shutil.rmtree(save_path)

predictor = TabularPredictor(
    label=target_col,
    problem_type='regression',
    eval_metric='rmse',
    path=save_path
)

predictor.fit(
    train_data=train_data,
    presets='best_quality',
    num_bag_folds=5,
    num_stack_levels=2,
    excluded_model_types=['KNN'],
    fit_weighted_ensemble=True,
    time_limit=time_limit,
    verbosity=3
)


leaderboard = predictor.leaderboard(silent=True)
print("Leaderboard:")
print(leaderboard)


feature_importance = predictor.feature_importance(train_data)
print("\nFeature Importance:")
print(feature_importance)


print("\nModels in final ensemble:")
print(predictor.get_model_names())


predictions = predictor.predict(test_data)

submission['accident_risk'] = predictions
submission.to_csv('submission.csv', index=False)


submission.head()

