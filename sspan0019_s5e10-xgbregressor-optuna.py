import pandas as pd

train_data = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')


train_data.info()


test_data.info()


CAT_COLS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

# binary = road_signs_present, public_road, holiday, school_season
# nominal = road_type, lighting, weather
# ordinal = time_of_day (morning < afternoon < evening < night)


NUM_COLS = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
target = 'accident_risk'


train_data.duplicated().sum()


train_data.drop_duplicates(inplace=True)


train_data.isna().sum()


test_data.isna().sum()


import matplotlib.pyplot as plt
import seaborn as sns

for col in CAT_COLS:
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    train_data[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], title=col)
    train_data.groupby(col)['accident_risk'].mean().plot.bar(ax=ax[1], title='Average Accident Risk')
    plt.show()


import scipy.stats as stats

for col in NUM_COLS:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    train_data[col].plot.hist(bins=20)
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 3, 2)
    stats.probplot(train_data[col], dist="norm", plot=plt)
    plt.title(f"QQ plot of {col}")

    plt.subplot(1, 3, 3)
    sns.boxenplot(x=train_data[col])
    plt.title(f"Boxen plot of {col}")

    plt.tight_layout()
    plt.show()


sns.heatmap(train_data[NUM_COLS + ['accident_risk']].corr(), annot=True, cmap='coolwarm')


BINARY_CAT_COL = ['road_signs_present', 'public_road', 'holiday', 'school_season']

for col in BINARY_CAT_COL:
    train_data[col] = train_data[col].map({True: 1.0, False: 0.0})
    test_data[col] = test_data[col].map({True: 1.0, False: 0.0})


from sklearn.preprocessing import OneHotEncoder

NOMINAL_CAT_COLS = ['road_type', 'lighting', 'weather']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_cols = encoder.fit_transform(train_data[NOMINAL_CAT_COLS])
encoded_col_names = encoder.get_feature_names_out(NOMINAL_CAT_COLS)
encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=train_data.index)
train_data = pd.concat([train_data.drop(columns=NOMINAL_CAT_COLS), encoded_df], axis=1)

encoded_cols_test = encoder.transform(test_data[NOMINAL_CAT_COLS])
encoded_df_test = pd.DataFrame(encoded_cols_test, columns=encoded_col_names, index=test_data.index)
test_data = pd.concat([test_data.drop(columns=NOMINAL_CAT_COLS), encoded_df_test], axis=1)


from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['morning', 'afternoon', 'evening']], handle_unknown='use_encoded_value', unknown_value=-1)
train_data['time_of_day'] = encoder.fit_transform(train_data[['time_of_day']])
test_data['time_of_day'] = encoder.transform(test_data[['time_of_day']])


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_data[NUM_COLS] = scaler.fit_transform(train_data[NUM_COLS])
test_data[NUM_COLS] = scaler.transform(test_data[NUM_COLS])


train_data.info()


train_data.isna().sum()


train_data.head()


from sklearn.model_selection import train_test_split

X = train_data.drop(columns=[target])
y = train_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def objective(trial):
    
    params = {
        "objective": "reg:squarederror",
        "device": "cuda", 
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }

    n_estimators = params.pop("n_estimators")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, valid_idx in cv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dvalid, "validation")],
            verbose_eval=False,
        )

        preds = bst.predict(dvalid)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    return float(np.mean(scores))

study = optuna.create_study(
    direction="minimize",
    study_name="xgb_accident_risk",
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
best_score = study.best_value
print(f"Best CV RMSE: {best_score:.4f}")
best_params
'''


import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

best_params = {
    'n_estimators': 750,
     'max_depth': 9,
     'learning_rate': 0.12743124127771185,
     'subsample': 0.8699548900809934,
     'colsample_bytree': 0.9143089910904906,
     'gamma': 0.003936936423219337,
     'min_child_weight': 7.401081206504708,
     'reg_lambda': 0.09855984704814297,
     'reg_alpha': 0.005124760961036267
}

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    **best_params,
)

model.fit(X_train, y_train)
val_preds = model.predict(X_test)
val_rmse = np.sqrt(mean_squared_error(y_test, val_preds))
print(f"Hold-out RMSE: {val_rmse:.4f}")


final_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    **best_params,
)

final_model.fit(X, y)


test_preds = final_model.predict(test_data)
submission = pd.DataFrame({"accident_risk": test_preds.clip(0.0, 1.0)}, index=test_data.index)
submission.to_csv("/kaggle/working/submission.csv", index=True)
submission.head()

