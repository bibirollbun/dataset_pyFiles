seed = 42


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

print("train :", train.shape)
print("test :", test.shape)
print("sample_submission :", sample.shape)


train.isnull().sum().sort_values(ascending=False)


test.isnull().sum().sort_values(ascending=False)


from sklearn.impute import SimpleImputer

# Handle missing values in test set
imputer = SimpleImputer(strategy='most_frequent')
test['winddirection'] = imputer.fit_transform(test[['winddirection']])


X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']
test = test.drop('id', axis=1)


import numpy as np

def feature_engineering(df):
    # 1. Temporal features
    df["month"] = ((df["day"] - 1) // 30) % 12 + 1  # Approximate month
    df["season"] = (df["month"] % 12) // 3
    
    df["day_of_week"] = df["day"] % 7  # Approximate day of the week
    df["is_weekend"] = df["day_of_week"].isin([6, 0]).astype(int)  # 0=Sunday, 6=Saturday

    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)
    
    # 2. Temperature Features
    df["temp_range"] = df["maxtemp"] - df["mintemp"]
    df["dewpoint_depression"] = df["temparature"] - df["dewpoint"]
    df["temp_diff"] = df["maxtemp"] - df["temparature"]

    # 3. Humidity & Pressure Features
    df["humidity_pressure_ratio"] = df["humidity"] / df["pressure"]
    df["dewpoint_humidity_ratio"] = df["dewpoint"] / df["humidity"]
    df["pressure_change"] = df["pressure"].diff().fillna(0)
    
    # 4. Cloud & Sunshine Features
    df["cloud_sunshine_ratio"] = df["cloud"] / (df["sunshine"] + 1e-6)  # Avoid division by zero
    df["sunshine_category"] = df["sunshine"] // 4
    
    # 5. Wind Features
    df["wind_speed_squared"] = df["windspeed"] ** 2
    df["wind_chill"] = df["temparature"] - (df["windspeed"] * 0.1)  # Simple approximation
    
    df["wind_x"] = np.sin(np.radians(df["winddirection"]))
    df["wind_y"] = np.cos(np.radians(df["winddirection"]))

    # 6. Get the value change rule
    for col in df.columns:
        if col in ['day', 'month', 'day_of_week', 'is_weekend', 'sin_day', 'cos_day', 'season']:
            continue

        # 计算各列的差值
        diff_1 = (df[col] - df[col].shift(1)).fillna(0)
        diff_mean_3 = (df[col] - df[col].rolling(window=3, min_periods=1).mean().shift(1)).fillna(0)
        diff_mean_7 = (df[col] - df[col].rolling(window=7, min_periods=1).mean().shift(1)).fillna(0)
        diff_max_3 = (df[col] - df[col].rolling(window=3, min_periods=1).max().shift(1)).fillna(0)
        diff_max_7 = (df[col] - df[col].rolling(window=7, min_periods=1).max().shift(1)).fillna(0)
        diff_min_3 = (df[col] - df[col].rolling(window=3, min_periods=1).min().shift(1)).fillna(0)
        diff_min_7 = (df[col] - df[col].rolling(window=7, min_periods=1).min().shift(1)).fillna(0)

        # 把新列添加到列表中
        new_columns = [
            diff_1.rename(f'{col}_diff_1'),
            diff_mean_3.rename(f'{col}_diff_mean_3'),
            diff_mean_7.rename(f'{col}_diff_mean_7'),
            diff_max_3.rename(f'{col}_diff_max_3'),
            diff_max_7.rename(f'{col}_diff_max_7'),
            diff_min_3.rename(f'{col}_diff_min_3'),
            diff_min_7.rename(f'{col}_diff_min_7')
        ]

        df = pd.concat([df] + new_columns, axis=1)

    return df

    
X = feature_engineering(X)
test = feature_engineering(test)
X.head()


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

rf_params = {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': seed,
    'n_jobs': -1
}
extra_params= {
    'n_estimators': 200,
    'max_depth': 5,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': seed,
    'n_jobs': -1
}
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 200,
    'learning_rate': 0.02,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'random_state': seed,
}
hgb_params= {
    'max_iter': 200,
    'max_depth': 6,
    'learning_rate': 0.02,
    'random_state': seed,
    'verbose': 0
}
ctb_params = {
    'n_estimators': 200,
    'learning_rate': 0.02,
    'depth': 6,
    'random_strength': 0.1,
    'bagging_temperature': 0.8,
    'random_state': seed,
    'verbose': 0
}
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 200,
    'learning_rate': 0.02,
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'random_state': seed,
    'verbose': -1
}

weights = [0.3, 0.1, 0.1, 0.1, 0.1, 0.3]

models = [
    ['RandomForestClassifier', RandomForestClassifier(**rf_params)],
    ['ExtraTreesClassifier', ExtraTreesClassifier(**extra_params)],
    ['XGBClassifier', XGBClassifier(**xgb_params)],
    ['HistGradientBoostingClassifier', HistGradientBoostingClassifier(**hgb_params)],
    ['CatBoostClassifier', CatBoostClassifier(**ctb_params)],
    ['LGBMClassifier', LGBMClassifier(**lgb_params)],
]


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
best_score = 0
final_predict = None

for index, [train_index, test_index] in enumerate(skf.split(X, y)):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    y_preds = []
    preds = []
    
    print(f"----------Fold {index}----------")
    # Evaluate each model
    for index, [model_name, model] in enumerate(models):
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        
        score = roc_auc_score(y_test, y_pred)
        print(f"{model_name}'s score: {score}")
        
        pred = model.predict(test)
        preds.append(pred)

    y_pred = [0] * len(y_test)
    for index in range(len(weights)):
        y_pred = y_pred + y_preds[index] * weights[index]

    score = roc_auc_score(y_test, y_pred)
    print(f'ensemble score: {score}')

    if (score > best_score):
        best_score = score
        pred = [0] * len(test)
        for index in range(len(weights)):
            pred = pred + preds[index] * weights[index]
        final_predict = pred

print(f"----------finally----------")
print(f'best_score: {best_score}')


submission = pd.DataFrame({'id': sample['id'], 'rainfall': final_predict})
print(submission.head())
submission.to_csv('submission.csv', index=False)

