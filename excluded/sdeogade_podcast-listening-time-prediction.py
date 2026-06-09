# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
import shap
from sklearn.isotonic import IsotonicRegression


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_ids = test['id']


train.info()


train.head()


# Feature engineering with time-based features
def create_time_features(df):
    # Convert episode number to temporal sequence
    df['Episode_Seq'] = df['Episode_Title'].str.extract('(\d+)').astype(int)
    df.sort_values(['Podcast_Name', 'Episode_Seq'], inplace=True)
    
    # Rolling popularity features
    df['Rolling_Host_Pop_7'] = df.groupby('Podcast_Name')['Host_Popularity_percentage'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['Rolling_Guest_Pop_5'] = df.groupby('Podcast_Name')['Guest_Popularity_percentage'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # Time since last episode
    df['Days_Since_Last'] = df.groupby('Podcast_Name')['Episode_Seq'].diff().fillna(0)
    
    return df.drop(columns=['id', 'Episode_Title', 'Podcast_Name'])

train = create_time_features(train)
test = create_time_features(test)


# Handle missing values
for df in [train, test]:
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']:
        df[col].fillna(df[col].median(), inplace=True)


# Advanced feature engineering
def create_features(df):
    df['Host_Guest_Ratio'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)
    df['Ad_Intensity'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['Prime_Time'] = df['Publication_Time'].isin(['Evening', 'Night']).astype(int)
    return pd.get_dummies(df, columns=['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])

train = create_features(train)
test = create_features(test)


# Verify final data types
print("Train dtypes:")
print(train.dtypes)
print("\nTest dtypes:")
print(test.dtypes)


# Align columns
train, test = train.align(test, join='left', axis=1, fill_value=0)


# Bayesian optimization with Optuna
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist', 
        'device': "cuda",
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        "enable_categorical": False 
    }
    
    scores = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[valid_idx]
        y_train, y_val = X_train.pop('Listening_Time_minutes'), X_val.pop('Listening_Time_minutes')
        
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)
        
        model = xgb.train(params, dtrain, 1000, evals=[(dval, 'val')], 
                         early_stopping_rounds=50, verbose_eval=0)
        scores.append(model.best_score)
    
    return np.mean(scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params


# Feature selection with SHAP
X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']


model = xgb.XGBRegressor(**best_params).fit(X, y)
explainer = shap.Explainer(model)
shap_values = explainer(X)


# Select top features
shap_sum = np.abs(shap_values.values).mean(axis=0)
selected_features = X.columns[np.argsort(shap_sum)[-50:]]  # Keep top 50 features


# Final training data
X = X[selected_features]
X_test = test[X.columns]


# Ensemble models
def train_ensemble(X, y, X_test):
    xgb_preds = []
    cat_preds = []
    lgb_preds = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**best_params, n_estimators=1500)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=0)
        xgb_preds.append(xgb_model.predict(X_test))
        
        # CatBoost
        cat_model = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=7, 
                                     task_type='GPU', verbose=0)
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        cat_preds.append(cat_model.predict(X_test))
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=1500, 
                                      device='gpu', verbose=-1)
        lgb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        lgb_preds.append(lgb_model.predict(X_test))
    
    return np.mean(xgb_preds, 0), np.mean(cat_preds, 0), np.mean(lgb_preds, 0)


xgb_pred, cat_pred, lgb_pred = train_ensemble(X, y, X_test)


# Optimize ensemble weights
def optimize_weights(y_true, preds):
    from scipy.optimize import minimize
    def loss(weights):
        blended = weights[0]*preds[0] + weights[1]*preds[1] + weights[2]*preds[2]
        return mean_squared_error(y_true, blended)
    
    initial_weights = [0.33, 0.33, 0.34]
    bounds = [(0,1)]*3
    cons = ({'type': 'eq', 'fun': lambda w: sum(w)-1})
    result = minimize(loss, initial_weights, bounds=bounds, constraints=cons)
    return result.x


# Use validation set for weight optimization
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_preds = train_ensemble(X_train, y_train, X_val)
optimal_weights = optimize_weights(y_val, val_preds)


# Apply optimal weights
ensemble_pred = (optimal_weights[0] * xgb_pred +
                 optimal_weights[1] * cat_pred +
                 optimal_weights[2] * lgb_pred)


# Updated post-processing calibration
def calibrate_predictions(train_preds, test_preds, y_train):
    """Calibrate predictions using Isotonic Regression"""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(train_preds, y_train)
    return calibrator.transform(test_preds)


# Create calibration set
X_train, X_cal, y_train, y_cal = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Get uncalibrated predictions on calibration set
cal_preds = train_ensemble(X_train, y_train, X_cal)[0]  # Use XGB predictions

# Calibrate final predictions
calibrated_preds = calibrate_predictions(cal_preds, ensemble_pred, y_cal)

# Clip predictions
final_pred = np.clip(calibrated_preds, y.min(), y.quantile(0.995))


# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': final_pred
})
submission.to_csv('optimized_ensemble_submission.csv', index=False)




