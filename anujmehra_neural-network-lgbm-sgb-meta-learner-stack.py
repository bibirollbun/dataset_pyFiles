# Kaggle Notebook for Podcast Listening Time Prediction (Playground S5E4) - KFold Target Encoding + Advanced Feature Engineering + Optimized Bagging

# 1. Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

# 2. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# 3. Feature Engineering (Upgraded)
def feature_engineering(df):
    df['Has_Guest'] = df['Guest_Popularity_percentage'].notnull().astype(int)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(0)
    df['Ads_Per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)').fillna(0).astype(int)
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Popularity_Log'] = np.log1p(df['Total_Popularity'])
    df['Ads_Length_Ratio'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# 4. Advanced KFold Target Encoding
def kfold_target_encoding(train_df, test_df, target_col, cat_col, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_df[f'{cat_col}_TE'] = np.nan
    test_target_means = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        train_part, val_part = train_df.iloc[train_idx], train_df.iloc[val_idx]
        means = train_part.groupby(cat_col)[target_col].mean()
        train_df.loc[train_df.index[val_idx], f'{cat_col}_TE'] = val_part[cat_col].map(means)
        test_target_means.append(test_df[cat_col].map(means))
    test_df[f'{cat_col}_TE'] = np.mean(test_target_means, axis=0)
    return train_df, test_df

train, test = kfold_target_encoding(train, test, target_col='Listening_Time_minutes', cat_col='Podcast_Name', n_splits=5)

# 5. Sentiment Mapping
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
train['Episode_Sentiment'] = train['Episode_Sentiment'].map(sentiment_map)
test['Episode_Sentiment'] = test['Episode_Sentiment'].map(sentiment_map)

# Label Encoding for other Categorical Columns
cat_cols = ['Genre', 'Publication_Day', 'Publication_Time']
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Final Feature List
features = ['Podcast_Name_TE', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment',
            'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',
            'Number_of_Ads', 'Ads_Per_Minute', 'Episode_Num', 'Has_Guest', 'Host_Guest_Interaction',
            'Total_Popularity', 'Popularity_Log', 'Ads_Length_Ratio']

X_full = train[features]
y_full = train['Listening_Time_minutes']
X_test = test[features]

# 6. Bagging Seeds
bagging_seeds = [42, 52, 62, 72, 82]
final_bagged_preds = np.zeros(len(X_test))

for seed in bagging_seeds:
    print(f"Training models with seed {seed}")
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=seed)

    scaler = QuantileTransformer(output_distribution='normal', random_state=seed)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lgb_model = LGBMRegressor(n_estimators=5000, learning_rate=0.01, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=seed)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[early_stopping(100), log_evaluation(0)])
    lgb_pred_val = lgb_model.predict(X_val)
    lgb_pred_test = lgb_model.predict(X_test)

    xgb_model = XGBRegressor(n_estimators=5000, learning_rate=0.01, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method='hist')
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', early_stopping_rounds=100, verbose=0)
    xgb_pred_val = xgb_model.predict(X_val)
    xgb_pred_test = xgb_model.predict(X_test)

    cat_model = CatBoostRegressor(iterations=5000, learning_rate=0.01, depth=8, random_seed=seed, loss_function='RMSE', verbose=0)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    cat_pred_val = cat_model.predict(X_val)
    cat_pred_test = cat_model.predict(X_test)

    nn = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=1000, random_state=seed)
    nn.fit(X_train_scaled, y_train)
    nn_pred_val = nn.predict(X_val_scaled)
    nn_pred_test = nn.predict(X_test_scaled)

    # Blending optimization
    def rmse_func(weights):
        blended = weights[0]*xgb_pred_val + weights[1]*cat_pred_val + weights[2]*lgb_pred_val + weights[3]*nn_pred_val
        return mean_squared_error(y_val, blended, squared=False)

    starting_values = [0.4, 0.3, 0.2, 0.1]
    bounds = [(0,1)]*4
    constraints = ({'type':'eq','fun':lambda w: 1-sum(w)})

    res = minimize(rmse_func, starting_values, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = res.x
    print(f"Optimized Weights for seed {seed}: {optimized_weights}")

    blended_preds = optimized_weights[0]*xgb_pred_test + optimized_weights[1]*cat_pred_test + optimized_weights[2]*lgb_pred_test + optimized_weights[3]*nn_pred_test
    blended_preds = np.clip(blended_preds, 0, 120)

    final_bagged_preds += blended_preds / len(bagging_seeds)

# 7. Final Submission
submission = sample_submission.copy()
submission['Listening_Time_minutes'] = np.round(final_bagged_preds, 2)
submission.to_csv('my_submission.csv', index=False)

print("✅ Final my_submission.csv generated successfully with KFold Target Encoding, Advanced Features, Blending, and Bagging!")


