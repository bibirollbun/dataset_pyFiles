!pip install xgboost==3.1.1


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

import optuna, json
import xgboost as xgb

import warnings 
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv',)
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


# downcast numerical columns

def downcasting(data: pd.DataFrame, verbose: bool=True) -> pd.DataFrame:

    mem_before = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
            
    for col in data.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="integer")
        
        elif pd.api.types.is_float_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], downcast="float")

    mem_after = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%\n")

    
    return data

train = downcasting(train)
test = downcasting(test)


def add_new_features(df):
    df['road_weather'] = df['road_type'].astype(str) + '_' + df['weather'].astype(str)
    df['road_light'] = df['road_type'].astype(str) + '_' + df['lighting'].astype(str)
    df['weather_light'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)
    # df['road_time'] = df['road_type'].astype(str) + '_' + df['time_of_day'].astype(str)
    # df['weather_time'] = df['weather'].astype(str) + '_' + df['time_of_day'].astype(str)
    df['speed_curve'] = df['speed_limit'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    df['accidents_curve'] = df['num_reported_accidents'] * df['curvature']

    return df

train = add_new_features(train)
test = add_new_features(test)


num = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents',
       'speed_curve',
       'accidents_speed',
       'accidents_curve',
]

boolean = ['road_signs_present','public_road','holiday','school_season']

cat = ['road_type','lighting','weather','time_of_day', 'road_weather', 'road_light', 'weather_light', 
       # 'road_time', 
       # 'weather_time'
]

# convert categorical columns to category type
for col in cat:
    for data in [train, test]:
        data[col] = data[col].astype("category")



cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
target_col = 'accident_risk'

fig, axs = plt.subplots(2, 4, figsize=(20, 10)) # Increased figsize for readability

fig.suptitle('Feature and Target Distribution Analysis', fontsize=18, y=1.02)

for i in range(len(cols)):
    sns.histplot(
        data=train,
        x=cols[i],
        bins=20,
        ax=axs[0, i],
        color='lightgreen',
        edgecolor='black',
        kde=False
    )
    axs[0, i].set_title(cols[i])


# histogram of original target
sns.histplot( data=train, x=target_col, bins=30, ax=axs[1, 0], color='skyblue', edgecolor='black', kde=True)
axs[1, 0].set_title(f'Histplot of {target_col}')

# boxplot of original target
sns.boxplot( data=train, y=target_col, ax=axs[1, 1], color='lightcoral')
axs[1, 1].set_title(f'Boxplot of {target_col}')
axs[1, 1].set_ylabel('')

# log-transformed version of the target for plotting
train_log_target = np.log1p(train[target_col])

# histogram of log1p transformed target
sns.histplot( x=train_log_target, bins=30, ax=axs[1, 2], color='mediumseagreen', edgecolor='black', kde=True )
axs[1, 2].set_title(f'Histplot of log1p({target_col})')

# boxplot of log1p transformed target
sns.boxplot( y=train_log_target, ax=axs[1, 3], color='plum')
axs[1, 3].set_title(f'Boxplot of log1p({target_col})')
axs[1, 3].set_ylabel('')
plt.tight_layout(rect=[0, 0, 1, 0.98]) 
plt.show()


fig, ax = plt.subplots(figsize=(10,6))
cols = boolean + num + ['accident_risk']
corr = train[cols].corr()

sns.heatmap(corr, cmap = 'crest', annot = True)
plt.title('Non categorical Feature correlation Heatmap', fontsize = 15, pad=10)
plt.tight_layout()
plt.show()


X = train.drop(['id','accident_risk'], axis=1)
y = train['accident_risk']

test = test.copy().drop(columns=['id'], axis=1)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    
    "n_estimators": 10_000,
    "max_depth": 9,
    "min_samples_split": 5,
    "random_state": 42,
    'learning_rate': 0.005, 
    
    # 'min_child_weight': 9, 
    'enable_categorical': True,
    'enable_categorical': True,
    'n_jobs': -1,
    
    'subsample': 0.7, 
    'colsample_bytree': 0.8, 
    'lambda': 2.0, 
    'alpha': 1.0,
    "tree_method": "hist",
    "device": "cuda",
    
    'seed': 42,
    'max_bin': 128
    # 'callbacks':[early_stop]
    }



seeds = [42, 128, 256, 510, 1024]

cols = [f"seed_{seed}" for seed in seeds]
oof_xgb_full = []
xgb_preds_full = []

seed_results = {seed:[] for seed in seeds}

for seed in seeds:
    SEED = seed
    n_splits = 7
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Out-of-fold (OOF) predictions for blending on the training data
    oof_xgb = np.zeros(X.shape[0])
    
    # Predictions on the test set (on the ORIGINAL target scale)
    xgb_preds = np.zeros(test.shape[0])
    
    print(f"\nTraining xgboost with {seed}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        early_stop = xgb.callback.EarlyStopping(rounds=300, metric_name='rmse', data_name='validation_0')
        
        # Ensure params uses the newly created callback object
        local_params = params.copy()
        local_params['callbacks'] = [early_stop]
        local_params["seed"] = seed
        local_params["max_bin"] = seed
        
        model = xgb.XGBRegressor(**local_params)
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)], 
            verbose= 500,
        )
        
        oof_xgb[val_idx] = model.predict(X_val)
        
        test_preds = model.predict(test)

        xgb_preds += test_preds / n_splits
        
        fold_mse = mean_squared_error(y_val, oof_xgb[val_idx])
        seed_results[seed].append(np.sqrt(fold_mse))
        print(f"----> XGBoost Fold {fold + 1} validation MSE (Original Scale): {fold_mse:.6f}")

    oof_xgb_full.append(oof_xgb)
    xgb_preds_full.append(xgb_preds)


# seeds = [42, 256, 510, 3000]

# cols = [f"seed_{seed}" for seed in seeds]
# oof_xgb_full = []
# xgb_preds_full = []

# for seed in seeds:
#     SEED = seed
#     n_splits = 7
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
#     # Out-of-fold (OOF) predictions for blending on the training data
#     oof_xgb = np.zeros(X.shape[0])
    
#     # Predictions on the test set (on the ORIGINAL target scale)
#     xgb_preds = np.zeros(test.shape[0])
    
#     print(f"\nTraining xgboost with {seed}")
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
#         print(f"\nTraining fold {fold + 1}/{n_splits}")
        
#         X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
#         X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
#         y_train_log = np.log1p(y_train)
#         y_val_log = np.log1p(y_val)
        
#         early_stop = xgb.callback.EarlyStopping(rounds=300, metric_name='rmse', data_name='validation_0')
        
#         # Ensure params uses the newly created callback object
#         local_params = params.copy()
#         local_params['callbacks'] = [early_stop]
#         local_params["seed"] = seed
        
#         model = xgb.XGBRegressor(**local_params)
        
#         model.fit(
#             X_train,
#             y_train_log,
#             eval_set=[(X_val, y_val_log)], 
#             verbose= 500,
#         )
        
#         val_preds_log = model.predict(X_val)
#         # Inverse transform to get predictions on original scale
#         val_preds_orig = np.expm1(val_preds_log)
        
#         # Store OOF predictions (on original scale)
#         oof_xgb[val_idx] = val_preds_orig
        
#         # Predict on test data (output is on log scale)
#         test_preds_log = model.predict(test)
#         # Inverse transform and accumulate (on original scale)
#         xgb_preds += np.expm1(test_preds_log) / n_splits
        
#         fold_mse = mean_squared_error(y_val, oof_xgb[val_idx])
#         print(f"----> XGBoost Fold {fold + 1} validation MSE (Original Scale): {fold_mse:.6f}")

#     oof_xgb_full.append(oof_xgb)
#     xgb_preds_full.append(xgb_preds)


# let's check it out cross-validation results for each seed
for seed, results in seed_results.items():
    print(f"Seed: {seed}, avg RMSE: {np.mean(results):.5f}, std: {np.std(results):.5f}")


oof_xgb_full_array = np.array(oof_xgb_full)
xgb_preds_full_array = np.array(xgb_preds_full)

# transpose the array to swap rows (seeds) and columns (samples)
oof_xgb_full_df = pd.DataFrame(data=oof_xgb_full_array.T, columns=cols, index=X.index) 
xgb_preds_full_df = pd.DataFrame(data=xgb_preds_full_array.T, columns=cols, index=test.index)

# check the performance of ensembling results

for col in oof_xgb_full_df.columns.tolist():
    rmse = np.sqrt(mean_squared_error(y, oof_xgb_full_df[col]))
    mse = mean_squared_error(y, oof_xgb_full_df[col])
    mae = mean_absolute_error(y, oof_xgb_full_df[col])
    
    print(f"Column: {col}")
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}\n")


oof_mean = oof_xgb_full_df.mean(axis=1)
preds_mean = xgb_preds_full_df.mean(axis=1)

print(f"Mean value of all predictions\nRMSE: {np.sqrt(mean_squared_error(y, oof_mean))}\nMSE: {mean_squared_error(y, oof_mean)}\nMAE: {mean_absolute_error(y, oof_mean)}")


alphas = [1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0]
meta_model = RidgeCV(alphas=alphas, scoring="neg_root_mean_squared_error", cv=5)
meta_model.fit(oof_xgb_full_df, y)


# prediction
meta_oof = meta_model.predict(oof_xgb_full_df)
meta_preds = meta_model.predict(xgb_preds_full_df)

meta_rmse = mean_squared_error(y, meta_oof, squared=False)
print(f"Ridge RMSE : {meta_rmse:.5f}")
print(f"Best alpha: {meta_model.alpha_}")


# save results
submission['accident_risk'] = preds_mean if rmse < meta_rmse else meta_preds
submission.to_csv('submission_1.csv', index=False) 
submission.head()


feature_names = X.columns.tolist() 
importances = model.feature_importances_
df_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

df_importances = df_importances.sort_values(by='importance', ascending=False).head(10).reset_index(drop=True)

plt.figure(figsize=(8, 5))
df_plot = df_importances.iloc[::-1] 

plt.barh(df_plot['feature'], df_plot['importance'], color='green')
plt.title('Top 10 Feature Importances in XGBoost', fontsize=14)
plt.xlabel('Feature Importance (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
target = df.columns.tolist()[-1]


def create_frequency_features(train_df, test_df, cols, num, cat):
    """
    Add frequency and binning features to the dataset.
    
    - For each column, create <col>_freq = how often each value appears in train data.
    - For numeric columns, split values into 5 and 10 quantile bins (groups) to show rank or range.
    """
    train, test = train_df.copy(), test_df.copy()

    for col in cols:
        # Frequency encoding: how common each value is
        freq = train[col].value_counts(normalize=True)
        train[f"{col}_freq"] = train[col].map(freq)
        test[f"{col}_freq"] = test[col].map(freq).fillna(train[f"{col}_freq"].mean())

        # Binning: group numeric values into quantiles
        if col in num:
            for q in [5, 10, 15, 20]:
                try:
                    train[f"{col}_bin{q}"], bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    test[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True)
                except Exception:
                    train[f"{col}_bin{q}"] = test[f"{col}_bin{q}"] = 0

    new_num = train.drop(columns=cat+[target]).columns.tolist()
    return train, test, new_num

def mixed_features(df):
    df['road_weather'] = df['road_type'].astype(str) + '_' + df['weather'].astype(str)
    df['road_light'] = df['road_type'].astype(str) + '_' + df['lighting'].astype(str)
    df['weather_light'] = df['weather'].astype(str) + '_' + df['lighting'].astype(str)

    df['speed_curve'] = df['speed_limit'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    df['accidents_curve'] = df['num_reported_accidents'] * df['curvature']

    return df

df = mixed_features(df)
df_test = mixed_features(df_test)

# Identify feature
cols = df.drop(columns=target).columns.tolist()

# Categorical features
cat = [col for col in cols if df[col].dtype in ["object","category"] and col != target]

# Numerical features
num = [col for col in cols if df[col].dtype not in ["object","category","bool"] and col not in ["id", target]]

# Creating new features based on the frequency of numerical features
df, df_test, new_num = create_frequency_features(df, df_test.copy(), cols, num, cat)

# Preparing categorical features
df[cat], df_test[cat] = df[cat].astype("category"), df_test[cat].astype("category")

# Mapping a column
map_col = "num_reported_accidents"
map_num_reported = {0:0, 1:0, 2:0, 3:2, 4:4, 5:3, 6:1, 7:0}
df[map_col] = df[map_col].map(map_num_reported)
df_test[map_col] = df_test[map_col].map(map_num_reported)

# Dropping unnecessary columns
remove = ["time_of_day", "num_lanes", "road_type", "road_signs_present", "id_freq"]
df = df.drop(columns=remove)
df_test = df_test.drop(columns=remove)

# Dropping ID and duplicates
df.drop(columns="id", inplace=True)
df.drop_duplicates(inplace=True)

df = downcasting(df)
df_test = downcasting(df_test)


fig, ax = plt.subplots(figsize=(20,10))
corr = df.select_dtypes(exclude=['category']).corr()

sns.heatmap(corr, cmap = 'crest', annot = True)
plt.title('Non categorical Feature correlation Heatmap', fontsize = 15, pad=10)
plt.tight_layout()
plt.show()


X = df.drop(['accident_risk'], axis=1)
y = df['accident_risk']

test = df_test.copy().drop(columns=['id'], axis=1)


xgb_params  = {
    'tree_method': 'hist', 
    'device': 'cuda', 
    'eval_metric': 'rmse', 
    'enable_categorical': True, 
    "n_estimators": 10_000,
    'random_state': 42,
    'max_bin': 512, 
    'min_child_weight': 3,
    'max_delta_step': 1, 
    'max_depth': 11, 
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.01,
    'reg_alpha': 0.1,
    'reg_lambda': 0.4,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    'scale_pos_weight': 0.3,
}


seeds = [42, 128, 256, 510, 1024]

cols = [f"seed_{seed}" for seed in seeds]
oof_xgb_full = []
xgb_preds_full = []

seed_results = {seed:[] for seed in seeds}

for seed in seeds:
    SEED = seed
    n_splits = 7
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Out-of-fold (OOF) predictions for blending on the training data
    oof_xgb = np.zeros(X.shape[0])
    
    # Predictions on the test set (on the ORIGINAL target scale)
    xgb_preds = np.zeros(test.shape[0])
    
    print(f"\nTraining xgboost with {seed}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        early_stop = xgb.callback.EarlyStopping(rounds=300, metric_name='rmse', data_name='validation_0')
        
        # Ensure params uses the newly created callback object
        local_params = xgb_params.copy()
        local_params['callbacks'] = [early_stop]
        local_params["seed"] = seed
        
        model = xgb.XGBRegressor(**local_params)
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)], 
            verbose= 500,
        )
        
        oof_xgb[val_idx] = model.predict(X_val)
        
        test_preds = model.predict(test)

        xgb_preds += test_preds / n_splits
        
        fold_mse = mean_squared_error(y_val, oof_xgb[val_idx])
        seed_results[seed].append(np.sqrt(fold_mse))
        print(f"----> XGBoost Fold {fold + 1} validation MSE (Original Scale): {fold_mse:.6f}")

    oof_xgb_full.append(oof_xgb)
    xgb_preds_full.append(xgb_preds)


# let's check it out cross-validation results for each seed
for seed, results in seed_results.items():
    print(f"Seed: {seed}, avg RMSE: {np.mean(results):.5f}, std: {np.std(results):.5f}")

print()

oof_xgb_full_array = np.array(oof_xgb_full)
xgb_preds_full_array = np.array(xgb_preds_full)

# transpose the array to swap rows (seeds) and columns (samples)
oof_xgb_full_df = pd.DataFrame(data=oof_xgb_full_array.T, columns=cols, index=X.index) 
xgb_preds_full_df = pd.DataFrame(data=xgb_preds_full_array.T, columns=cols, index=test.index)

# check the performance of ensembling results

for col in oof_xgb_full_df.columns.tolist():
    rmse = np.sqrt(mean_squared_error(y, oof_xgb_full_df[col]))
    mse = mean_squared_error(y, oof_xgb_full_df[col])
    mae = mean_absolute_error(y, oof_xgb_full_df[col])
    
    print(f"Column: {col}")
    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}\n")


oof_mean = oof_xgb_full_df.mean(axis=1)
preds_mean = xgb_preds_full_df.mean(axis=1)

print(f"<------ Mean value of all predictions  -------------> \nRMSE: {np.sqrt(mean_squared_error(y, oof_mean))}\nMSE: {mean_squared_error(y, oof_mean)}\nMAE: {mean_absolute_error(y, oof_mean)}")


# save results
submission['accident_risk'] = preds_mean.to_numpy()
submission.to_csv('submission.csv', index=False) 
submission.head()


oof_pred_probs = oof_xgb_full_df
test_pred_probs = xgb_preds_full_df

def objective(trial):
    weights = np.array([trial.suggest_float(f'w{m}', -1, 1) for m in oof_pred_probs.keys()])
    weights /= np.sum(weights)

    preds = np.zeros(len(y))
    for m, weight in zip(oof_pred_probs.keys(), weights):
        preds += oof_pred_probs[m] * weight

    return np.sqrt(mean_squared_error(y.to_numpy(), preds.to_numpy()))

sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=5)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, n_jobs=-1)



best_params = study.best_params

best_weights = np.array([study.best_params[f'w{m}'] for m in oof_pred_probs.keys()])
best_weights /= np.sum(best_weights)

best_weights = {
    model: weight for model, weight in sorted(
        zip(oof_pred_probs.keys(), best_weights),
        key=lambda x: x[1],
        reverse=True
    )
}
print(json.dumps(best_weights, indent=2))


weighted_test_preds = np.zeros(len(test_pred_probs["seed_42"]))
weighted_train_preds = np.zeros(len(oof_pred_probs["seed_42"]))

for m, weight in best_weights.items():
    weighted_test_preds += test_pred_probs[m] * weight
    weighted_train_preds += oof_pred_probs[m] * weight

print(f"Weighted oof train RMSE: {np.sqrt(mean_squared_error(y, weighted_train_preds)):.5f}")



# save results
submission['accident_risk'] = weighted_test_preds.to_numpy()
submission.to_csv('submission_2.csv', index=False) 
submission.head()




