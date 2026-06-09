import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import itertools


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


sorted(test["Duration"].unique()) == sorted(train["Duration"].unique())


sorted(test["Age"].unique()) == sorted(train["Age"].unique())


train['Sex'] = train['Sex'].map({'male':1,'female':0})
test['Sex'] = test['Sex'].map({'male':1,'female':0})


def add_duration_age_interactions(df, features):
    """Turn each unique Duration/Age into its ndicator feature."""
    df = df.copy()
    for col in ['Duration','Age']:
        for val in sorted(df[col].unique()):
            for feature in features:
                col_name = f'{feature}_{col}_{int(val)}'
                df[col_name] = np.where(df[col] == val, df[feature], 0)
    return df


def add_feature_cross_terms(df, features):
    df = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    return df

def add_statistical_features(df, features):
    df_new = df.copy()
    df_new["row_mean"] = df[features].mean(axis=1)
    df_new["row_std"] = df[features].std(axis=1)
    df_new["row_max"] = df[features].max(axis=1)
    df_new["row_min"] = df[features].min(axis=1)
    df_new["row_median"] = df[features].median(axis=1)
    return df_new

def add_interaction_features(df, features):
    df_new = df.copy()
    for f1, f2 in itertools.combinations(features, 2):
        df_new[f"{f1}_plus_{f2}"] = df_new[f1] + df_new[f2]
        df_new[f"{f1}_minus_{f2}"] = df_new[f1] - df_new[f2]
        df_new[f"{f2}_minus_{f1}"] = df_new[f2] - df_new[f1]
        df_new[f"{f1}_div_{f2}"] = df_new[f1] / (df_new[f2] + 1e-6)
        df_new[f"{f2}_div_{f1}"] = df_new[f2] / (df_new[f1] + 1e-6)
    return df_new


def add_gender_masking(df, features, gender_col='Sex'):

    df['Male'] = df[gender_col]  # 1 if male, 0 otherwise
    df['Female'] = 1 - df[gender_col]  # Inverse

    # Create interactions
    for feat in features:
        df[f'{feat}_x_Male'] = df[feat] * df['Male']
        df[f'{feat}_x_Female'] = df[feat] * df['Female']

    # Drop temporary one-hot columns (optional)
    df.drop(['Male', 'Female'], axis=1, inplace=True)

    return df

def add_categorical_aggregations(df,categorical_cols,numerical_cols):

    # Single categorical column case (simplified from original loop)
    for cat_col in categorical_cols:
        # Calculate min/max aggregations for all numerical columns
        aggs = df.groupby(cat_col)[numerical_cols].agg(['min', 'max'])

        # Flatten multi-index columns
        aggs.columns = [f"{cat_col}_{num_col}_{stat}" 
                       for num_col, stat in aggs.columns]

        # Merge with original data
        df = df.merge(aggs, on=cat_col, how='left')

    return df


def add_custom_features(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df["Heart_Rate"] / (df["Duration"] + 1e-5)
    df['Calories_Burned'] = np.where(
        df['Sex'] == 'male',
        (-55.0969 + (0.6309 * df['Heart_Rate']) + (0.1988 * df['Weight']) + (0.2017 * df['Age'])) / 4.184 * df['Duration'],
        (-20.4022 + (0.4472 * df['Heart_Rate']) - (0.1263 * df['Weight']) + (0.074 * df['Age'])) / 4.184 * df['Duration']
    )

    return df


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
unique_binning_features = ['Heart_Rate','Body_Temp']
numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']


train = add_feature_cross_terms(train, numerical_features)
test = add_feature_cross_terms(test, numerical_features)

train = add_interaction_features(train, numerical_features)
test = add_interaction_features(test, numerical_features)

train = add_statistical_features(train, numerical_features)
test = add_statistical_features(test, numerical_features)

train = add_duration_age_interactions(train,unique_binning_features)
test = add_duration_age_interactions(test,unique_binning_features)

train = add_gender_masking(train, features=['Duration', 'Heart_Rate', 'Body_Temp'])
test = add_gender_masking(test, features=['Duration', 'Heart_Rate', 'Body_Temp'])

train = add_categorical_aggregations(train,['Sex'],numerical_cols)
test = add_categorical_aggregations(test,['Sex'],numerical_cols)

train = add_custom_features(train)
test = add_custom_features(test)


train["Sex"] = train["Sex"].astype('category')
test["Sex"] = test["Sex"].astype('category')


import pickle
with open("/kaggle/input/cols-to-remove/cols_to_remove.bin", "rb") as f:
    features_to_drop = pickle.load(f)


train_reduced = train.drop(columns=features_to_drop)
test_reduced = test.drop(columns=features_to_drop)

print(f"\nOriginal number of features: {train.shape[1], test.shape[1]}")
print(f"Number of features after selection: {train_reduced.shape[1], test_reduced.shape[1]}")


X = train_reduced.drop(columns=['id', 'Calories'])
y_log = np.log1p(train_reduced['Calories'])
y = train_reduced['Calories']
X_test = test_reduced.drop(columns=['id'])


categorical_columns = ['Sex']
cat_features = [list(X.columns).index(col) for col in categorical_columns]
cat_features


FOLDS = 7
SEED = 40


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from sklearn.preprocessing import KBinsDiscretizer
import optuna


def rmsle(y_true,y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# cat_params = {
#      'iterations': 2000,
#      'learning_rate': 0.04702799520841546,
#      'depth': 10,
#      'l2_leaf_reg': 6.543117388416962,
#      'random_strength': 18.987839255325074,
#      'bagging_temperature': 0.004332917984131254,
#      'border_count': 255,
#      'task_type': 'GPU',
#      'devices': '0:1',
#      'cat_features': cat_features, #,<----------CHECK THIS
#      'early_stopping_rounds': 100,
#      'eval_metric': 'RMSE',
#      'random_seed': SEED,
#      'verbose': False,
#     }


n_bins = 10
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
duration_binned = discretizer.fit_transform(train[['Duration']]).astype(int).flatten()


# # OPTUNA HYPERPARAMETER SEARCH
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# def objective(trial):
#     params = {
#         'iterations': trial.suggest_int('iterations', 200, 2000, step=200),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-1, log=True),
#         'depth': trial.suggest_int('depth', 6, 10),
#         'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
#         'random_strength': trial.suggest_float('random_strength', 0.0, 20.0),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'border_count': trial.suggest_categorical('border_count', [128,254]),
#      'task_type': 'GPU',
#      'devices': '0:1',
#      'cat_features': cat_features,
#      'early_stopping_rounds': 100,
#      'eval_metric': 'RMSE',
#      'random_seed': SEED,
#      'verbose': False,
#     }
    
#     scores = []
#     for i, (train_idx, valid_idx) in enumerate(skf.split(X, duration_binned)):
#         print(f"\nFold {i+1}")
#         x_train, y_train = X.iloc[train_idx], y_log[train_idx]
#         x_valid, y_valid = X.iloc[valid_idx], y_log[valid_idx]

#         model = CatBoostRegressor(**params)
#         model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=cat_features)
    
#         # Get OOF logged predictions
#         oof_pred_logged = model.predict(x_valid)
#         # Inverse Log OOF
#         oof_pred = np.expm1(oof_pred_logged)
    
#         # Calculate RMSLE b/w actual y and inverse-log oof_pred
#         y_actual_valid = y[valid_idx]
#         oof_rmsle = rmsle(y_actual_valid, oof_pred)
#         scores.append(oof_rmsle)
        
#     return np.mean(scores)

# def run_optimization(n_trials=50):
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=n_trials)
#     print('Best trial:')
#     print(study.best_trial.params)
#     return study.best_trial.params

# best_params = run_optimization(n_trials=200)


# best_params = {'iterations': 2000, 'learning_rate': 0.014749770944969522, 'depth': 10, 'l2_leaf_reg': 0.10196069157090158, 'random_strength': 2.2316258996261022, 'bagging_temperature': 0.24728153374541328, 'border_count': 254}


# final_params = {
#     **best_params,
#      'loss_function': 'RMSE',
#      'task_type': 'GPU',
#      'devices': '0:1',
#      'cat_features': cat_features,
#      'early_stopping_rounds': 100,
#      'eval_metric': 'RMSE',
#      'random_seed': SEED,
#      # 'verbose': False,
# }
# final_params


final_params = {    
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 12,
    'l2_leaf_reg': 3,
    'border_count': 254,
    'loss_function': 'RMSE',
    'task_type': 'GPU',
    'devices': '0:1',
    'cat_features': cat_features,
    'early_stopping_rounds': 200,
    'eval_metric': 'RMSE',
    'random_seed': SEED,
    'verbose':100
}


# FINAL MODEL TRAINING
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

scores = []
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))


for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\nFold {i+1}")
    x_train, y_train = X.iloc[train_idx], y_log[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y_log[valid_idx]

    cat_model = CatBoostRegressor(**final_params)
    cat_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], cat_features=cat_features)

    # Get OOF logged predictions
    oof_pred_logged = cat_model.predict(x_valid)
    # Inverse Log OOF
    oof_pred = np.expm1(oof_pred_logged)
    # Save the OOF predictions in array
    oof_predictions[valid_idx] = oof_pred
    
    # GET test logged prediction
    test_pred_logged = cat_model.predict(X_test)
    # Inverse log test
    test_pred = np.expm1(test_pred_logged)
    # Save test predictions in array
    test_predictions += test_pred / FOLDS

    # Calculate RMSLE b/w actual y and inverse-log oof_pred
    y_actual_valid = y[valid_idx]
    oof_rmsle = rmsle(y_actual_valid, oof_pred)
    scores.append(oof_rmsle)
    
    print(f"Fold {i+1} RMSLE: {oof_rmsle:.4f}")


mean_rmsle = np.mean(scores)
std_rmsle = np.std(scores)
print(f"Mean RMSLE: {mean_rmsle:.4f} ± {std_rmsle:.4f}")


import joblib
joblib.dump(cat_model,"cat_model_6.joblib")
np.save("cat_oof.npy",oof_predictions)


# Save the results as a CSV
submission['Calories'] = np.clip(test_predictions,1,314)
submission['Calories'] = round(submission['Calories'])

submission.to_csv('submission.csv',index=False)

