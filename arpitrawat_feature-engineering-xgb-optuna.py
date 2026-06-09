import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_org=pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')


train_org.columns = train_org.columns.str.strip()
train_org['rainfall'] = train_org['rainfall'].str.lower().map({'yes': 1, 'no': 0})
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


column_order = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']

train = train[column_order]  
train_org = train_org[column_order]


train_org.sample(5)


# day column = [1,2,3,----,364,365,366]
for i in range(366):
    train_org.loc[i, 'day'] = i + 1  


from scipy.stats import entropy

def advanced_feature_engineering(df):
    
    # ------------------- Day & Seasonal Features -------------------
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['week_sin'] = np.sin(2 * np.pi * df['day'] / 7)
    df['week_cos'] = np.cos(2 * np.pi * df['day'] / 7)

    def get_season(day):
        """ Assign seasons based on standard astronomical dates """
        if 79 <= day <= 171:
            return 'spring'  # March 20 - June 20
        elif 172 <= day <= 263:
            return 'summer'  # June 21 - Sept 22
        elif 264 <= day <= 354:
            return 'fall'    # Sept 23 - Dec 20
        else:
            return 'winter'  # Dec 21 - March 19
    
    df['season'] = df['day'].apply(get_season)
    df = pd.get_dummies(df, columns=['season'], drop_first=False)

    # ------------------- Pressure Features -------------------
    df['pressure_rolling_mean'] = df['pressure'].rolling(window=7, min_periods=1).mean()
    df['pressure_rolling_std'] = df['pressure'].rolling(window=7, min_periods=1).std()
    df['pressure_diff'] = df['pressure'].diff()
    df['pressure_change_rate'] = df['pressure_diff'] / (df['pressure'].shift(1) + 1e-6)  # Avoid division by zero

    # ------------------- Temperature Features -------------------
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_ewm'] = df['temparature'].ewm(span=10, adjust=False).mean()
    df['temp_change'] = df['temparature'].diff()
    df['temp_variance'] = df['temparature'].rolling(window=5).var()

    # Heat Index (better than temp * humidity)
    df['temp_humidity_interaction'] = df['temparature'] + (0.2 * df['humidity'])

    # Wind Chill (Perceived coldness based on wind speed)
    df['wind_chill'] = 35.74 + 0.6215 * df['temparature'] - 35.75 * (df['windspeed']**0.16) + 0.4275 * df['temparature'] * (df['windspeed']**0.16)

    # ------------------- Dewpoint & Humidity Features -------------------
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    df['rh_approx'] = 100 - (5 * df['dewpoint_depression'])

    # Saturation Vapor Pressure (SVP) - Tetens' Equation
    df['svp'] = 6.1078 * np.exp((17.27 * df['temparature']) / (df['temparature'] + 237.3))

    # Absolute Humidity (AH) in g/mÂ³
    df['abs_humidity'] = (6.112 * np.exp((17.67 * df['temparature']) / (df['temparature'] + 243.5)) * df['humidity'] * 2.1674) / (273.15 + df['temparature'])

    # ------------------- Cloud & Sunshine Features -------------------
    df['cloud_category'] = pd.cut(df['cloud'], bins=[0, 20, 50, 80, 100], labels=['clear', 'partly_cloudy', 'mostly_cloudy', 'overcast'])
    df['sky_opacity'] = df['cloud'] / 100

    # Sunshine fraction (actual sunshine hours divided by daylight hours)
    df['sunshine_pct'] = df['sunshine'] / 24  # Kept as per your original logic
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1e-6)  # Avoid division by zero
    df['interaction'] = df['cloud'] + df['sunshine'] + df['humidity']

    # ------------------- Wind Features -------------------
    df['winddir_sin'] = np.sin(np.radians(df['winddirection']))
    df['winddir_cos'] = np.cos(np.radians(df['winddirection']))

    def wind_category(speed):
        """Categorize wind speed based on Beaufort scale approximation"""
        if speed < 2:
            return 'calm'
        elif speed < 6:
            return 'light_breeze'
        elif speed < 12:
            return 'moderate_breeze'
        elif speed < 20:
            return 'strong_breeze'
        elif speed < 30:
            return 'gale'
        else:
            return 'storm'
    
    df['wind_category'] = df['windspeed'].apply(wind_category)
    df = pd.get_dummies(df, columns=['wind_category'], drop_first=True)

    # Wind Power Calculation (using air density at sea level Ï� = 1.225 kg/mÂ³)
    df['wind_power'] = 0.5 * 1.225 * (df['windspeed']**3)

    # ------------------- Advanced Statistical Features -------------------
    # Rolling mean and standard deviation for trend analysis
    df['humidity_rolling_mean'] = df['humidity'].rolling(window=5, min_periods=1).mean()
    df['humidity_rolling_std'] = df['humidity'].rolling(window=5, min_periods=1).std()

    # Temperature entropy (measuring unpredictability in weather)
    df['temp_entropy'] = df['temparature'].rolling(window=10, min_periods=1).apply(lambda x: entropy(np.histogram(x, bins=5)[0] + 1e-6), raw=True)

    # Temperature and pressure lag features
    for lag in range(1, 4):
        df[f'temp_lag_{lag}'] = df['temparature'].shift(lag)
        df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag)

    # Interaction between temperature, humidity, and wind
    df['temp_humidity_wind_interaction'] = df['temparature'] * df['humidity'] / (df['windspeed'] + 1e-6)

    # Fourier features for capturing seasonality
    df['fourier_1'] = np.sin(2 * np.pi * df['day'] / 180)
    df['fourier_2'] = np.cos(2 * np.pi * df['day'] / 180)

    return df



train=advanced_feature_engineering(train)
train_org=advanced_feature_engineering(train_org)
test=advanced_feature_engineering(test)


# verifying feature engineering result
train.shape,train_org.shape,test.shape


train = pd.concat([train, train_org], ignore_index=True)



X=train.drop(columns='rainfall')
y=train['rainfall']



import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Set Optuna verbosity to reduce logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Number of folds
FOLDS = 50
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(test), FOLDS))

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function to optimize XGBoost hyperparameters"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 5, 300),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
    }

    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", enable_categorical=True)
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, val_preds)

fold = 1
for train_idx, val_idx in skf.split(X, y):
    print(f"\nOptimizing hyperparameters for fold {fold}...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Run Optuna to find the best hyperparameters for this fold
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30)

    best_params = study.best_params
    print(f"Best parameters for fold {fold}: {best_params}")

    # Train model with best parameters
    model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", enable_categorical=True)
    model.fit(X_train, y_train)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

    test_preds[:, fold - 1] = model.predict_proba(test)[:, 1]

    fold += 1

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

test_pred = test_preds.mean(axis=1)



submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.rainfall=test_pred
submission.to_csv("submission.csv", index=False)




