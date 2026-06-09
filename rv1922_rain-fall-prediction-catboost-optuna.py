import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import optuna
pd.set_option('future.no_silent_downcasting', True)
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.metrics import roc_auc_score, make_scorer


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
original = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


train.head()


train.info()


train.isnull().sum()


train = train.drop(columns=['id'])

test_ids = test["id"]
test = test.drop(columns=['id'])


def generate_features(df):

    ## lag feature
    for lag in [1, 3, 7]:
        df[f'Pressure_lag{lag}'] = df['pressure'].shift(lag)
        df[f'Humidity_lag{lag}'] = df['humidity'].shift(lag)

    ## amount of change
    df['Pressure_change_1d'] = df['pressure'] - df['pressure'].shift(1)
    df['Humidity_change_1d'] = df['humidity'] - df['humidity'].shift(1)

    ## temperature related
    df['Temp_range'] = df['maxtemp'] - df['mintemp']
    df["avg_temp"] = (df["maxtemp"] + df["mintemp"]) / 2
    df['Dewpoint_diff'] = df['temparature'] - df['dewpoint']

    ## sunshine, cloud amount
    df['Sunshine_per_hour'] = df['sunshine'] / 24
    df['Cloud_per_hour'] = df['cloud'] / 24
    df['Cloud_Humidity_ratio'] = df['cloud'] / (df['humidity'] + 1e-5)
    df['Cloud_Sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)

    ## wind related
    df['Wind_x'] = df['windspeed'] * np.cos(np.radians(df['winddirection']))
    df['Wind_y'] = df['windspeed'] * np.sin(np.radians(df['winddirection']))

    ## others
    df['humidity_cloud_interaction'] = df['humidity'] * df['cloud']
    df['humidity_sunshine_interaction'] = df['humidity'] * df['sunshine']
    df['Pressure_Humidity_Interaction'] = df['pressure'] * df['humidity']
    df["cloud_wind_interaction"] = df["cloud"] * df["windspeed"]
    df['relative_dryness'] = 100 - df['humidity']
    df['sunshine_percentage'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['cloud_percentage'] = df['cloud'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['weather_index'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    df['Temp_Ratio'] = df['temparature'] / df['maxtemp'].max()

    df['Try'] = (df['humidity'] + df['cloud'] + df['dewpoint'])
    df['Try_2'] = (df['cloud'] - df['sunshine']) + df['temparature']

    # wet-bulb temperature
    def calc_wet_bulb(T, RH):
        return T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + \
               np.arctan(T + RH) - np.arctan(RH - 1.676331) + \
               0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) - 4.686035

    df['wet_bulb_temp'] = calc_wet_bulb(df['temparature'], df['humidity'])

    # saturated vapor pressure
    def calc_saturation_vapor_pressure(temp):
        return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))

    df['e_s_temp'] = calc_saturation_vapor_pressure(df['temparature'])
    df['e_s_dewpoint'] = calc_saturation_vapor_pressure(df['dewpoint'])

    # vapor pressure deficit
    df['vapor_pressure_deficit'] = df['e_s_temp'] - df['e_s_dewpoint']

    df.fillna(method='bfill', inplace=True)

    return df


train = generate_features(train)
test = generate_features(test)


train.isnull().sum()


train.head()


X = train.drop(columns=['rainfall'])  
y = train['rainfall']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),  # Increased range for better exploration
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'depth': trial.suggest_int('depth', 4, 16),  # Increased depth range for more complex trees
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 15),  # Wider range for regularization
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.5, 20),  # Wider range for randomness
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Adding subsample for randomization
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),  # Feature sampling
        'verbose': 0,
        'random_state': 42
    }

    model = CatBoostClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=30,
        verbose=0
    )
    
    y_pred = model.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_pred)
    
    return roc_auc


#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=10)

#best_params = study.best_params
#best_score = study.best_value

#print("Best Hyperparameters:", best_params)
#print("Best ROC AUC Score:", best_score)


best_params = {
    'iterations': 2818,
    'learning_rate': 0.2454478491807572,
    'depth': 9,
    'l2_leaf_reg': 12.78609353544912,
    'border_count': 109,
    'bagging_temperature': 0.7123032964576839,
    'random_strength': 9.823676863255114,
    'colsample_bylevel': 0.6059192602627542,
    'task_type': 'GPU',  # Enable GPU acceleration
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_true, y_pred_proba, preds = [], [], []

for train_index, test_index in skf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    catboost_model = CatBoostClassifier(**best_params)
    catboost_model.fit(X_train, y_train)

    y_val_pred_proba = catboost_model.predict_proba(X_val)[:, 1]
    y_true.extend(y_val)
    y_pred_proba.extend(y_val_pred_proba)

    test_preds = catboost_model.predict_proba(test[X.columns])[:, 1]  
    preds.append(test_preds)

auc_roc = roc_auc_score(y_true, y_pred_proba)
print(f"ğŸ�† AUC-ROC Score: {auc_roc:.4f}")


test.head()


final_preds = np.mean(preds, axis=0)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'rainfall': final_preds})
submission.to_csv('submission.csv', index=False)

