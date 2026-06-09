import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import gc
import optuna
import warnings
warnings.simplefilter('ignore')


pd.set_option('display.max_columns', 100)

TARGET = 'Listening_Time_minutes'
CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads']

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
print(f"Orig  shape: {original.shape}")

# Combine with original data
original_clean = original.dropna(subset=[TARGET]).drop_duplicates()
train = pd.concat([train, original_clean], axis=0, ignore_index=True)


def feature_eng(df):
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Dictionary mappings
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    # Extract episode number
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    
    # Replace categorical values with numerical mappings
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    # Convert to categorical
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')

    # Additional features
    df['is_weekend'] = df['Publication_Day'].isin([5, 6]).astype(int).astype('category')
    df['is_weekday'] = (~df['Publication_Day'].isin([5, 6])).astype(int).astype('category')
    
    # Ratio features
    df['host_guest_ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)
    df['guest_host_ratio'] = df['Guest_Popularity_percentage'] / (df['Host_Popularity_percentage'] + 1)
    df['ads_per_minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['popularity_sum'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['popularity_product'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage'] / 100
    df['popularity_diff'] = np.abs(df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage'])
    
    # Time-based features
    df['morning_episode'] = (df['Publication_Time'] == 0).astype(int).astype('category')
    df['afternoon_episode'] = (df['Publication_Time'] == 1).astype(int).astype('category')
    df['evening_episode'] = (df['Publication_Time'] == 2).astype(int).astype('category')
    df['night_episode'] = (df['Publication_Time'] == 3).astype(int).astype('category')
    df['evening_night_episode'] = df['Publication_Time'].isin([2, 3]).astype(int).astype('category')
    
    # Day-based features
    df['monday_episode'] = (df['Publication_Day'] == 0).astype(int).astype('category')
    df['friday_episode'] = (df['Publication_Day'] == 4).astype(int).astype('category')
    
    # Sentiment features
    df['positive_episode'] = (df['Episode_Sentiment'] == 2).astype(int).astype('category')
    df['negative_episode'] = (df['Episode_Sentiment'] == 0).astype(int).astype('category')
    
    # Polynomial features for numerical columns
    for col in NUMS:
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_cubed'] = df[col] ** 3
        df[f'{col}_sqrt'] = np.sqrt(df[col] + 1)
        df[f'{col}_log'] = np.log1p(df[col])
    
    # Interaction between numerical features
    for i, col1 in enumerate(NUMS):
        for col2 in NUMS[i+1:]:
            df[f'{col1}_mult_{col2}'] = df[col1] * df[col2]
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1)
    
    # Drop unnecessary columns
    df = df.drop(columns=['Episode_Title'])
    return df

# Apply feature engineering
train = feature_eng(train)
test = feature_eng(test)

# Create interaction features
encoded_columns = []

selected_comb = [
    # 2-interaction
    ['Episode_Length_minutes', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Number_of_Ads'],    
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Host_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Genre', 'Episode_Sentiment'],
    ['Genre', 'Publication_Time'],
    ['Publication_Day', 'Publication_Time'],
    ['Podcast_Name', 'Genre'],
    ['Podcast_Name', 'Episode_Sentiment'],
    
    # 3-interaction (selected most important ones)
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Genre', 'Publication_Day', 'Publication_Time'],
    ['Podcast_Name', 'Genre', 'Episode_Sentiment'],
]

for comb in selected_comb:
    name = '_'.join(comb)
        
    if len(comb) == 2:
        train[name] = train[comb[0]].astype(str) + '_' + train[comb[1]].astype(str)
        test[name] = test[comb[0]].astype(str) + '_' + test[comb[1]].astype(str)
        
    elif len(comb) == 3:
        train[name] = (train[comb[0]].astype(str) + '_' +
                       train[comb[1]].astype(str) + '_' +
                       train[comb[2]].astype(str))
        test[name] = (test[comb[0]].astype(str) + '_' +
                      test[comb[1]].astype(str) + '_' +
                      test[comb[2]].astype(str))
    
    encoded_columns.append(name)

train[encoded_columns] = train[encoded_columns].astype('category')
test[encoded_columns] = test[encoded_columns].astype('category')

# Get all features
FEATURES = NUMS + CATS + encoded_columns + [
    'is_weekend', 'is_weekday', 'host_guest_ratio', 'guest_host_ratio', 'ads_per_minute', 
    'popularity_sum', 'popularity_product', 'popularity_diff',
    'morning_episode', 'afternoon_episode', 'evening_episode', 'night_episode', 'evening_night_episode',
    'monday_episode', 'friday_episode', 'positive_episode', 'negative_episode'
] + [f'{col}_squared' for col in NUMS] + [f'{col}_cubed' for col in NUMS] + [f'{col}_sqrt' for col in NUMS] + [f'{col}_log' for col in NUMS]

# Add numerical interactions
for i, col1 in enumerate(NUMS):
    for col2 in NUMS[i+1:]:
        FEATURES.append(f'{col1}_mult_{col2}')
        FEATURES.append(f'{col1}_div_{col2}')

print(f"Train Shape: {train.shape}")
print(f"Test  Shape: {test.shape}")
print(f"Number of features: {len(FEATURES)}")


# Target encoding function with smoothing
def target_encode(df_train, df_val, col, target, stats='mean', prefix='TE', smoothing=10):
    # Calculate the global mean
    global_mean = df_train[target].mean()
    
    # Group by the column and calculate the aggregation
    agg_data = df_train.groupby(col)[target].agg(['count', stats])
    
    # Apply smoothing
    if stats == 'mean':
        smoothed_mean = (agg_data['count'] * agg_data[stats] + smoothing * global_mean) / (agg_data['count'] + smoothing)
        agg = smoothed_mean
    else:
        agg = agg_data[stats]
    
    agg = agg.rename(f"{prefix}_{col}")
    
    df_val = df_val.copy()
    df_val[f"{prefix}_{col}"] = df_val[col].map(agg).astype(float)
    df_val[f"{prefix}_{col}"].fillna(global_mean if stats == 'mean' else agg.mean(), inplace=True)
    return df_val

# Define cross-validation strategy
FOLDS = 8
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


# Initialize arrays for predictions
oof_xgb = np.zeros(len(train))
oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
oof_ridge = np.zeros(len(train))
pred_xgb = np.zeros(len(test))
pred_lgb = np.zeros(len(test))
pred_cat = np.zeros(len(test))
pred_ridge = np.zeros(len(test))


# Training loop
for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print(f"--- Fold {i+1} / {FOLDS} ---")

    X_train = train.loc[train_idx, FEATURES + [TARGET]].reset_index(drop=True)
    y_train = X_train[TARGET]
    X_valid = train.loc[valid_idx, FEATURES].reset_index(drop=True)
    y_valid = train.loc[valid_idx, TARGET].reset_index(drop=True)
    X_test = test[FEATURES].reset_index(drop=True)

    # Nested target encoding
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for j, (train_idx2, valid_idx2) in enumerate(kf2.split(X_train)):
        inner_train = X_train.loc[train_idx2, FEATURES + [TARGET]].copy()
        inner_valid = X_train.loc[valid_idx2, FEATURES].copy()

        for col in encoded_columns:
            te_col = f'TE_{col}'
            inner_valid = target_encode(inner_train, inner_valid, col, TARGET, stats="mean", prefix="TE")
            X_train.loc[valid_idx2, te_col] = inner_valid[te_col].values

        del inner_train, inner_valid
        gc.collect()

    # Target encode validation and test sets
    for col in encoded_columns:
        X_valid = target_encode(X_train, X_valid, col, TARGET, stats="mean", prefix="TE")
        X_test = target_encode(X_train, X_test, col, TARGET, stats="mean", prefix="TE")

    # Add additional target encoding with different statistics
    for col in encoded_columns[:20]:  # Use top 20 most important interaction features
        X_train = target_encode(X_train, X_train, col, TARGET, stats="std", prefix="TE_std")
        X_valid = target_encode(X_train, X_valid, col, TARGET, stats="std", prefix="TE_std")
        X_test = target_encode(X_train, X_test, col, TARGET, stats="std", prefix="TE_std")

    # Prepare data for training
    te_cols = [f'TE_{col}' for col in encoded_columns] + [f'TE_std_{col}' for col in encoded_columns[:20]]
    X_train.drop([TARGET] + encoded_columns, axis=1, inplace=True)
    X_valid.drop(encoded_columns, axis=1, inplace=True)
    X_test.drop(encoded_columns, axis=1, inplace=True)

    # Define categorical features for LightGBM and CatBoost
    cat_features = [col for col in X_train.columns if X_train[col].dtype.name == 'category']
    
    # XGBoost model
    xgb_model = xgb.XGBRegressor(
        tree_method='hist',
        max_depth=10,
        colsample_bytree=0.6,
        subsample=0.8,
        n_estimators=10000,
        learning_rate=0.02,
        enable_categorical=True,
        early_stopping_rounds=200,
        min_child_weight=15,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    # LightGBM model
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.02,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=10000,
        early_stopping_rounds=200,
        random_state=42,
        verbose=-1
    )

    # CatBoost model
    cat_model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        cat_features=cat_features,
        early_stopping_rounds=200,
        verbose=500
    )

    # Train XGBoost
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=500)
    oof_xgb[valid_idx] = xgb_model.predict(X_valid)
    pred_xgb += xgb_model.predict(X_test) / FOLDS

    # Train LightGBM
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        categorical_feature=cat_features
    )
    oof_lgb[valid_idx] = lgb_model.predict(X_valid)
    pred_lgb += lgb_model.predict(X_test) / FOLDS

    # Train CatBoost
    cat_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=500
    )
    oof_cat[valid_idx] = cat_model.predict(X_valid)
    pred_cat += cat_model.predict(X_test) / FOLDS

    # Clean up memory
    del X_train, X_valid, X_test, y_train, y_valid
    if i != FOLDS - 1:
        del xgb_model, lgb_model, cat_model
    gc.collect()

# Evaluate individual models
rmse_xgb = mean_squared_error(train[TARGET], oof_xgb, squared=False)
rmse_lgb = mean_squared_error(train[TARGET], oof_lgb, squared=False)
rmse_cat = mean_squared_error(train[TARGET], oof_cat, squared=False)

print(f"XGBoost RMSE: {rmse_xgb:.5f}")
print(f"LightGBM RMSE: {rmse_lgb:.5f}")
print(f"CatBoost RMSE: {rmse_cat:.5f}")


# Find optimal weights for ensemble
def objective(trial):
    w1 = trial.suggest_float('w1', 0.0, 1.0)
    w2 = trial.suggest_float('w2', 0.0, 1.0)
    w3 = trial.suggest_float('w3', 0.0, 1.0)
    w4 = trial.suggest_float('w4', 0.0, 1.0)
    
    # Normalize weights
    sum_weights = w1 + w2 + w3 + w4
    w1 /= sum_weights
    w2 /= sum_weights
    w3 /= sum_weights
    w4 /= sum_weights
    
    # Weighted ensemble
    oof_ensemble = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat + w4 * oof_ridge
    
    # Calculate RMSE
    rmse = mean_squared_error(train[TARGET], oof_ensemble, squared=False)
    return rmse

# Optimize ensemble weights
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Get best weights
best_params = study.best_params
w1 = best_params['w1']
w2 = best_params['w2']
w3 = best_params['w3']
w4 = best_params['w4']

# Normalize weights
sum_weights = w1 + w2 + w3 + w4
w1 /= sum_weights
w2 /= sum_weights
w3 /= sum_weights
w4 /= sum_weights

print(f"Optimal weights: XGBoost={w1:.4f}, LightGBM={w2:.4f}, CatBoost={w3:.4f}, Ridge={w4:.4f}")

# Create ensemble prediction
oof_ensemble = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat + w4 * oof_ridge
pred_ensemble = w1 * pred_xgb + w2 * pred_lgb + w3 * pred_cat + w4 * pred_ridge

# Final RMSE
rmse_ensemble = mean_squared_error(train[TARGET], oof_ensemble, squared=False)
print(f"Ensemble RMSE: {rmse_ensemble:.5f}")

# Create submission
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub[TARGET] = pred_ensemble
sub.to_csv('submission.csv', index=False)
print("Submission file created successfully!")




