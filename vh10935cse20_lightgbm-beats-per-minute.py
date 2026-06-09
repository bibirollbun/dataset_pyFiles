import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head(3)


train.info()


train.dtypes


train.isna().sum()


test.head(2)


test.info()


test.isna().sum()


test.describe().T


columns = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
           'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
           'TrackDurationMs', 'Energy', 'BeatsPerMinute']
plt.figure(figsize=(20, 15))
for i, col in enumerate(columns):
    plt.subplot(4, 3, i+1)
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(columns):
    plt.subplot(4, 3, i+1)
    sns.boxplot(y=train[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


print("\nCorrelation Matrix:\n")
corr_matrix = train.corr()
print(corr_matrix)


train['Loudness_x_Duration'] = train['AudioLoudness'] * train['TrackDurationMs']
train['Energy_x_Rhythm'] = train['Energy'] * train['RhythmScore']
train['Mood_x_VocalContent'] = train['MoodScore'] * train['VocalContent']
train['Acoustic_x_Live'] = train['AcousticQuality'] * train['LivePerformanceLikelihood']


train.dtypes


test['Loudness_x_Duration'] = test['AudioLoudness'] * test['TrackDurationMs']
test['Energy_x_Rhythm'] = test['Energy'] * test['RhythmScore']
test['Mood_x_VocalContent'] = test['MoodScore'] * test['VocalContent']
test['Acoustic_x_Live'] = test['AcousticQuality'] * test['LivePerformanceLikelihood']


X=train.drop(columns=['id','BeatsPerMinute'])
y=train['BeatsPerMinute']
test_id=test['id']
test=test.drop(columns='id',axis=1)


from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import optuna


def lgb_objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = lgb.LGBMRegressor(**params) # Corrected to LGBMRegressor
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        val_preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)


print("Running Optuna study...")
study = optuna.create_study(direction='minimize')
study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)

print("Best hyperparameters found by Optuna:")
best_params = study.best_params
print(best_params)

# Get the best trial RMSE
print(f"Best RMSE from the study: {study.best_value:.4f}")


final_model = lgb.LGBMRegressor(**best_params, random_state=42)

# Fit the model on all training data
final_model.fit(X, y)


test_predictions = final_model.predict(test)


submission = pd.DataFrame({
    "id": test_id,
    "BeatsPerMinute": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

