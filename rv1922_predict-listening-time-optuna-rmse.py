import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cb
import optuna
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


train.info()


train.describe().T


train.isnull().sum()


numerical_column_names = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


object_column_names = train.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


print("Duplicated Rows:",train.duplicated().sum())

print("Number of Rows:",train.shape[0])

print("Number of Columns:",train.shape[1])


train.nunique()


cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']


for col in cat_cols:
    print(f"Unique categories in '{col}' column: {train[col].unique()}")
    print("<--- --- --- --- --- --- --- --- --- --->\n")


genre_mapping = {
    'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
    'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9
}

publication_day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

publication_time_mapping = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}

episode_sentiment_mapping = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2
}


podcast_name_mapping = {
    'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3,
    'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7,
    'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12,
    'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17,
    'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21,
    'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25,
    "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30,
    'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34,
    'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38,
    'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42,
    'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46,
    'Tune Time': 47
}


train["Episode_Title"] = train["Episode_Title"].str.replace("Episode ", "", regex=False).astype(int)
test["Episode_Title"] = test["Episode_Title"].str.replace("Episode ", "", regex=False).astype(int)


train['Genre'] = train['Genre'].map(genre_mapping)
test['Genre'] = test['Genre'].map(genre_mapping)

train['Publication_Day'] = train['Publication_Day'].map(publication_day_mapping)
test['Publication_Day'] = test['Publication_Day'].map(publication_day_mapping)

train['Publication_Time'] = train['Publication_Time'].map(publication_time_mapping)
test['Publication_Time'] = test['Publication_Time'].map(publication_time_mapping)

train['Episode_Sentiment'] = train['Episode_Sentiment'].map(episode_sentiment_mapping)
test['Episode_Sentiment'] = test['Episode_Sentiment'].map(episode_sentiment_mapping)


train['Podcast_Name'] = train['Podcast_Name'].map(podcast_name_mapping)
test['Podcast_Name'] = test['Podcast_Name'].map(podcast_name_mapping)


def add_custom_features(df):
    df['ad_density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['length_x_popularity'] = df['Episode_Length_minutes'] * df['Guest_Popularity_percentage']
    df['log_length'] = np.log1p(df['Episode_Length_minutes'])
    df['popularity_per_min'] = df['Guest_Popularity_percentage'] / (df['Episode_Length_minutes'] + 1)
    return df

train = add_custom_features(train)
test = add_custom_features(test)


train.head()


X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'tree_method': 'hist',     # Faster and supports GPU if device='cuda'
        'device': 'cuda',
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

# Run Optuna study
##study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)


#print("Best RMSE:", study.best_value)
#print("Best Parameters:", study.best_params)


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Out-of-fold predictions and test predictions
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))
rmse_scores = []

# XGBoost parameters
xgb_params = {
    'n_estimators': 1913,
    'max_depth': 12,
    'learning_rate': 0.02703638048415814,
    'subsample': 0.963715401933959,
    'colsample_bytree': 0.5239570527983759,
    'reg_alpha': 3.0126686792585446,
    'reg_lambda': 1.8088037056321993,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 1
}

# K-Fold loop
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"ğŸ”� Fold {fold + 1}/{n_splits}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              verbose=False)

    # Predict validation and test
    oof_pred = model.predict(X_valid)
    oof_predictions[valid_idx] = oof_pred
    test_predictions += model.predict(test) / n_splits

    # Fold RMSE
    rmse = mean_squared_error(y_valid, oof_pred, squared=False)
    rmse_scores.append(rmse)
    print(f"âœ… Fold {fold + 1} RMSE: {rmse:.4f}")

# Final RMSE
mean_rmse = np.mean(rmse_scores)
print(f"\nğŸ“Š Average RMSE across {n_splits} folds: {mean_rmse:.4f}")


submission = pd.DataFrame({
    'id': test['id'].values,
    'Listening_Time_minutes': test_predictions
})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� File Saved!!")


submission.head()

