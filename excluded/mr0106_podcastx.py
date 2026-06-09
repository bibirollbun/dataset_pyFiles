import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import joblib
import gc


SEED = 42
np.random.seed(SEED)


class Config:
    N_FOLDS = 5
    EARLY_STOP = 100
    
    @staticmethod
    def get_device_params():
        params = {
            'tree_method': 'hist',
            'random_state': 42
        }
        try:
            import torch
            if torch.cuda.is_available():
                params.update({
                    'device': 'cuda',
                    'predictor': 'gpu_predictor'
                })
                print("ğŸš€ GPU Acceleration Enabled")
            else:
                params['device'] = 'cpu'
        except:
            params['device'] = 'cpu'
        return params


def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
    sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
    
    for df in [train, test]:
        for col in df.select_dtypes(['float64', 'int64']):
            df[col] = pd.to_numeric(df[col], downcast='float' if 'float' in str(df[col].dtype) else 'integer')
    
    return train, test, sample_sub


def create_features(df, is_train=True, y=None):
    df = df.copy()
    
    # Handle all numeric columns systematically
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    
    # Safer log transforms with epsilon
    eps = 1e-6
    df['Host_log'] = np.log1p(df['Host_Popularity_percentage'].clip(0) + eps)
    df['Guest_log'] = np.log1p(df['Guest_Popularity_percentage'].clip(0) + eps)
    df['Host_Guest_Impact'] = df['Host_log'] * df['Guest_log']
    
    # More robust division with clipping
    df['Ads_Intensity'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'].clip(0.1) + eps)
    
    # Additional interaction feature
    df['Host_Ads_Interaction'] = df['Host_Popularity_percentage'] * df['Number_of_Ads']
    
    # Target encoding with error handling
    if is_train and y is not None:
        try:
            te = TargetEncoder()
            df['Genre_Encoded'] = te.fit_transform(df['Genre'], y)
            joblib.dump(te, 'target_encoder.pkl')
            df[TARGET] = df[TARGET].clip(0, df[TARGET].quantile(0.995))
        except Exception as e:
            print(f"Encoding error: {e}")
            df['Genre_Encoded'] = 0
    elif not is_train:
        try:
            te = joblib.load('target_encoder.pkl')
            df['Genre_Encoded'] = te.transform(df['Genre'])
        except:
            df['Genre_Encoded'] = 0
    
    return df


def prepare_features(df):
    cat_cols = ['Podcast_Name','Episode_Title','Genre',
               'Publication_Day','Publication_Time','Episode_Sentiment']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df.drop(columns=['id', TARGET], errors='ignore')


def train_model(X, y):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 8,
        'learning_rate': 0.02,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'n_estimators': 1500,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'early_stopping_rounds': Config.EARLY_STOP,
        **Config.get_device_params()
    }
    
    models = []
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=100 if fold == 1 else False)
        
        models.append(model)
        gc.collect()
    
    return models


if __name__ == "__main__":
    # Initialize
    TARGET = 'Listening_Time_minutes'
    
    # Load and prepare data
    train, test, sample_sub = load_data()
    y = train[TARGET]
    
    # Feature engineering
    train = create_features(train, is_train=True, y=y)
    test = create_features(test, is_train=False)
    
    # Model training
    X = prepare_features(train)
    models = train_model(X, y)
    
    # Generate submission
    X_test = prepare_features(test)
    test_preds = np.mean([m.predict(X_test) for m in models], axis=0)
    
    submission = sample_sub.copy()
    submission[TARGET] = test_preds.clip(0)
    submission.to_csv('submission.csv', index=False)
    joblib.dump(models, 'models.pkl')
    
    print("\nğŸ�‰ Submission created successfully!")
    print(f"ğŸ”¥ Final RMSE: {np.sqrt(mean_squared_error(y, models[0].predict(X))):.4f}")

