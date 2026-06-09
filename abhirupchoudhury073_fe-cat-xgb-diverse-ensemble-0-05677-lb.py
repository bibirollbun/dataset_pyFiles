import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


SEED = 42


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train = train.drop_duplicates(subset=train.columns).reset_index(drop=True)
df = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})


def add_feature_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_custom_features(df):
    df = df.copy()
    
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df["Heart_Rate"] / (df["Duration"] + 1e-5)
    # df['Calories_Burned'] = np.where(
    #     df['Sex'] == 'male',
    #     (-55.0969 + (0.6309 * df['Heart_Rate']) + (0.1988 * df['Weight']) + (0.2017 * df['Age'])) / 4.184 * df['Duration'],
    #     (-20.4022 + (0.4472 * df['Heart_Rate']) - (0.1263 * df['Weight']) + (0.074 * df['Age'])) / 4.184 * df['Duration']
    # )

    # 2. Calculate BMR
    def calculate_bmr(row):
        # Assumes 'Sex' is 'MALE' or 'FEMALE' at this point (after cleansing)
        if row['Sex'] == 1: # Match cleansed name
            return (((9.65 * row['Weight']) + (573 * (row['Height']/100)) - (5.08 * row['Age']) + 260) / 24) / 60
        elif row['Sex'] == 0: # Match cleansed name
            return (((7.38 * row['Weight']) + (607 * (row['Height']/100)) - (2.31 * row['Age']) + 43) / 24) / 60
        else:
            # If this 'else' block is reached, it means sex cleansing was incomplete
            print(f"DEBUG: Unexpected Sex value '{row['Sex']}' detected during BMR calculation.")
            return np.nan
    df['BMR'] = df.apply(calculate_bmr, axis=1) # Estimated Basal Metabolic Rate (BMR)

    return df

def add_unique_duration_features(df, features):
    df=df.copy()
    unique_durations = sorted(df["Duration"].unique())

    for duration in unique_durations:
        for feature in features:
            col_name = f'{feature}_Duration_{int(duration)}'
            df[col_name] = np.where(df['Duration'] == duration, df[feature], 0)
    return df

def add_unique_age_features(df, features):
    df = df.copy()
    unique_ages = sorted(df["Age"].unique())

    for age in unique_ages:
        for feature in features:
            col_name = f'{feature}_Age_{int(age)}'
            df[col_name] = np.where(df['Age'] == age, df[feature], 0)
    return df

def add_gender_masking(df, features, gender_col='Sex'):
    df = df.copy()
    df['Male'] = df[gender_col]  # 1 if male, 0 otherwise
    df['Female'] = 1 - df[gender_col]  # Inverse

    # Create interactions
    for feat in features:
        df[f'{feat}_x_Male'] = df[feat] * df['Male']
        df[f'{feat}_x_Female'] = df[feat] * df['Female']

    # Drop temporary one-hot columns (optional)
    df.drop(['Male', 'Female'], axis=1, inplace=True)

    return df


def rmsle(y_true,y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


numerical = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
gender_mask_features = ["Duration", "Heart_Rate", "Body_Temp"]
train = add_feature_cross_terms(train, numerical)
test = add_feature_cross_terms(test, numerical)


def predict1(train, test, cat_params, xgb_params):
    train = train.copy()
    test = test.copy()
    FOLDS = 5
    train = add_custom_features(train)
    test = add_custom_features(test)
    train["Sex"] = train["Sex"].astype('category')
    test["Sex"] = test["Sex"].astype('category')


    X = train.drop(columns=['id', 'Calories'])
    y_log = np.log1p(train['Calories'])
    y = train['Calories']
    X_test = test.drop(columns=['id'])

    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(train[['Duration']]).astype(int).flatten()
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    scores = []
    oof_predictions = np.zeros(len(X))
    test_predictions_cat = np.zeros(len(test))
    test_predictions_xgb = np.zeros(len(test))


    for i, (train_idx, valid_idx) in enumerate(skf.split(X, duration_bins)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y_log[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y_log[valid_idx]

        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=['Sex'])

        test_predictions_cat += cat_model.predict(X_test)
        oof_predictions[valid_idx] += cat_model.predict(X.iloc[valid_idx]) * 0.3

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

        test_predictions_xgb += xgb_model.predict(X_test)
        oof_predictions[valid_idx] += xgb_model.predict(X.iloc[valid_idx]) * 0.7
        

        

        # Calculate RMSLE b/w actual y and inverse-log oof_pred
        y_actual_valid = y[valid_idx]
        oof_rmsle = rmsle(y_actual_valid, np.clip(np.expm1(oof_predictions[valid_idx]), 1, 314))
        scores.append(oof_rmsle)
        
        print(f"Fold {i+1} RMSLE: {oof_rmsle:.4f}")
    
    mean_rmsle = np.mean(scores)
    std_rmsle = np.std(scores)
    print(f"Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")

    np.save('predict_1_oof_log',oof_predictions)
    test_predictions_cat /= FOLDS
    test_predictions_xgb /= FOLDS
    weighted_avg_preds = 0.3 * test_predictions_cat + 0.7 * test_predictions_xgb
    
    np.save('predict_1_wt_avg',weighted_avg_preds)
    result = np.clip(np.expm1(weighted_avg_preds), 1, 314)
    return result


def predict2(train, test, cat_params, xgb_params):
    FOLDS = 5
    train = train.copy()
    test = test.copy()
    train = add_unique_age_features(train, ['Heart_Rate', 'Duration'])
    test = add_unique_age_features(test, ['Heart_Rate', 'Duration'])
    train = add_unique_duration_features(train, ['Heart_Rate', 'Duration'])
    test = add_unique_duration_features(test, ['Heart_Rate', 'Duration'])
    train = add_gender_masking(train, gender_mask_features)
    test = add_gender_masking(test, gender_mask_features)
    train["Sex"] = train["Sex"].astype('category')
    test["Sex"] = test["Sex"].astype('category')


    X = train.drop(columns=['id', 'Calories'])
    y_log = np.log1p(train['Calories'])
    y = train['Calories']
    X_test = test.drop(columns=['id'])

    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(train[['Duration']]).astype(int).flatten()
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    scores = []
    oof_predictions = np.zeros(len(X))
    test_predictions_cat = np.zeros(len(test))
    test_predictions_xgb = np.zeros(len(test))


    for i, (train_idx, valid_idx) in enumerate(skf.split(X, duration_bins)):
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y_log[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y_log[valid_idx]

        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=['Sex'])

        test_predictions_cat += cat_model.predict(X_test)
        oof_predictions[valid_idx] += cat_model.predict(X.iloc[valid_idx]) * 0.5

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

        test_predictions_xgb += xgb_model.predict(X_test)
        oof_predictions[valid_idx] += xgb_model.predict(X.iloc[valid_idx]) * 0.5
        

        

        # Calculate RMSLE b/w actual y and inverse-log oof_pred
        y_actual_valid = y[valid_idx]
        oof_rmsle = rmsle(y_actual_valid, np.clip(np.expm1(oof_predictions[valid_idx]), 1, 314))
        scores.append(oof_rmsle)
        
        print(f"Fold {i+1} RMSLE: {oof_rmsle:.4f}")
    
    mean_rmsle = np.mean(scores)
    std_rmsle = np.std(scores)
    print(f"Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")

    np.save('predict_2_oof_log',oof_predictions)
    test_predictions_cat /= FOLDS
    test_predictions_xgb /= FOLDS
    weighted_avg_preds = 0.5 * test_predictions_cat + 0.5 * test_predictions_xgb

    np.save('predict_2_wt_avg',weighted_avg_preds)
    result = np.clip(np.expm1(weighted_avg_preds), 1, 314)
    return result


def predict3(train, test, cat_params, xgb_params):
    FOLDS = 50
    train = train.copy()
    test = test.copy()
    train["Sex"] = train["Sex"].astype('category')
    test["Sex"] = test["Sex"].astype('category')


    X = train.drop(columns=['id', 'Calories'])
    y_log = np.log1p(train['Calories'])
    y = train['Calories']
    X_test = test.drop(columns=['id'])

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    scores = []
    oof_predictions = np.zeros(len(X))
    test_predictions_cat = np.zeros(len(test))
    test_predictions_xgb = np.zeros(len(test))


    for i, (train_idx, valid_idx) in enumerate(kf.split(X)): # type: ignore
        print(f"\nFold {i+1}")
        x_train, y_train = X.iloc[train_idx], y_log[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y_log[valid_idx]

        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=['Sex'])

        test_predictions_cat += cat_model.predict(X_test)
        oof_predictions[valid_idx] += cat_model.predict(X.iloc[valid_idx]) * 0.3

        xgb_model = XGBRegressor(**xgb_params)
        xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

        test_predictions_xgb += xgb_model.predict(X_test)
        oof_predictions[valid_idx] += xgb_model.predict(X.iloc[valid_idx]) * 0.7
        

        

        # Calculate RMSLE b/w actual y and inverse-log oof_pred
        y_actual_valid = y[valid_idx]
        oof_rmsle = rmsle(y_actual_valid, np.clip(np.expm1(oof_predictions[valid_idx]), 1, 314))
        scores.append(oof_rmsle)
        
        print(f"Fold {i+1} RMSLE: {oof_rmsle:.4f}")
    
    mean_rmsle = np.mean(scores)
    std_rmsle = np.std(scores)
    print(f"Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")

    np.save('predict_3_oof_log',oof_predictions)
    test_predictions_cat /= FOLDS
    test_predictions_xgb /= FOLDS
    weighted_avg_preds = 0.3 * test_predictions_cat + 0.7 * test_predictions_xgb

    np.save('predict_3_wt_avg',weighted_avg_preds)
    result = np.clip(np.expm1(weighted_avg_preds), 1, 314)
    return result


cat_params1 = params = {
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 10,
    "verbose": 200,
    "random_state": SEED,
    "eval_metric": "RMSE",
    "loss_function": "RMSE",
    "task_type": "GPU",
    "devices": "0:1",
    "cat_features": ["Sex"]
}

xgb_params1 = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 10,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "gamma": 0.01,
    "max_delta_step": 2,
    "tree_method": "gpu_hist",
    "device": "cuda",
    "predictor": "gpu_predictor",
    "enable_categorical": True,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "verbosity": 0,
    "random_state": SEED
}

cat_params2 = {
    "iterations": 3200,
    "learning_rate": 0.02,
    "depth": 12,
    "loss_function": "RMSE",
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "eval_metric": "RMSE",
    "early_stopping_rounds": 200,
    "cat_features": ["Sex"],
    "verbose": 200,
    "task_type": "GPU",
    "devices": "0:1"
}

xgb_params2 = {
    "max_depth": 9,
    "colsample_bytree": 0.7,
    "subsample": 0.9,
    "n_estimators": 3000,
    "learning_rate": 0.01,
    "gamma": 0.01,
    "max_delta_step": 2,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "random_state": SEED,
    "early_stopping_rounds": 100,
    "tree_method": "gpu_hist",
    "device": "cuda",
    "predictor": "gpu_predictor"
}

cat_params3 = {
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 10,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 100,
    "verbose": 200,
    "random_state": SEED,
    "task_type": "GPU",
    "devices": "0:1",
    "cat_features": ["Sex"]
}

xgb_params3 = {
    "max_depth": 10,
    "colsample_bytree": 0.75,
    "subsample": 0.9,
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "gamma": 0.01,
    "max_delta_step": 2,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "tree_method": "gpu_hist",
    "device": "cuda",
    "predictor": "gpu_predictor",
    "random_state": SEED
}


pred1 = predict1(train,test,cat_params=cat_params1,xgb_params=xgb_params1)
pred2 = predict2(train,test,cat_params=cat_params2,xgb_params=xgb_params2)
pred3 = predict3(train,test,cat_params=cat_params3,xgb_params=xgb_params3)

submission['Calories'] = 0.25 * pred1 + 0.25 * pred2 + 0.5 * pred3
submission.to_csv("submission.csv", index=False)

