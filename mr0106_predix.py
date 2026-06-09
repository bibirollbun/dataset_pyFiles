import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Set global seed
SEED = 42
np.random.seed(SEED)


def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
    sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
    target = 'Listening_Time_minutes'
    return train, test, sample_sub, target

train, test, sample_sub, TARGET = load_data()


def preprocess(df, is_train=True):
    df = df.copy()
    
    # Handle missing values
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(df['Episode_Length_minutes'].median())
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(0)
    
    # Temporal features
    time_map = {'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4}
    day_map = {day:i for i, day in enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], 1)}
    
    df['Time_sin'] = np.sin(2*np.pi*df['Publication_Time'].map(time_map)/4)
    df['Time_cos'] = np.cos(2*np.pi*df['Publication_Time'].map(time_map)/4)
    df['Day_sin'] = np.sin(2*np.pi*df['Publication_Day'].map(day_map)/7)
    df['Day_cos'] = np.cos(2*np.pi*df['Publication_Day'].map(day_map)/7)
    
    # Interaction features
    df['Host_Guest_Ratio'] = df['Host_Popularity_percentage']/(df['Guest_Popularity_percentage']+1)
    df['Ads_per_Minute'] = df['Number_of_Ads']/(df['Episode_Length_minutes']+0.1)
    
    if is_train:
        df[TARGET] = df[TARGET].clip(0)
    
    return df

train = preprocess(train)
test = preprocess(test, is_train=False)


def prepare_features(df):
    cat_cols = ['Podcast_Name','Episode_Title','Genre','Publication_Day','Publication_Time','Episode_Sentiment']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df.drop(columns=['id', TARGET], errors='ignore')

X, y = prepare_features(train), train[TARGET]


def train_model(X, y):
    params = {
        'objective':'reg:squarederror',
        'eval_metric':'rmse',
        'max_depth':6,
        'learning_rate':0.05,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'n_estimators':1000,
        'early_stopping_rounds':50,
        'random_state':SEED
    }
    
    models = []
    for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X), 1):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                 eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                 verbose=False)
        models.append(model)
    
    return models

models = train_model(X, y)
joblib.dump(models, 'model.pkl')


def create_submission(models, test_df):
    X_test = prepare_features(test_df)
    preds = np.mean([m.predict(X_test) for m in models], axis=0)
    submission = sample_sub.copy()
    submission[TARGET] = np.clip(preds, 0, None)
    submission.to_csv('submission.csv', index=False)
    return submission

submission = create_submission(models, test)

