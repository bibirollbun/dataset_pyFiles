import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

print(f'train_df shape: {train_df.shape}')
print(f'test_df shape: {test_df.shape}')


train_df.head()


print(train_df.isnull().sum())
print('\n')
print(test_df.isnull().sum())


print(train_df.nunique())
print('\n')
print(test_df.nunique())


train_df.dtypes


features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']

fig = plt.figure(figsize=(20, 5 * len(features)))

for i, var_name in enumerate(features):
    ax = fig.add_subplot(len(features), 1, i + 1)
    sns.histplot(data=train_df, x=var_name, ax=ax, kde=True, bins='fd')
    ax.set_title(var_name)

plt.tight_layout()
plt.show()


features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']

fig = plt.figure(figsize=(20, 5 * len(features)))

for i, var_name in enumerate(features):
    ax = fig.add_subplot(len(features), 1, i + 1)
    sns.boxplot(data=train_df, x=var_name, ax=ax)
    ax.set_title(var_name)

plt.tight_layout()
plt.show()


corr = train_df.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='RdBu')

plt.title('Correlation Heatmap', fontsize=16)
plt.show()


train_id = train_df['id'] 
test_id = test_df['id'] 

train_df.drop('id', axis=1, inplace=True)
test_df.drop('id', axis=1, inplace=True)

ntrain = train_df.shape[0]
ntest = test_df.shape[0]

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop('BeatsPerMinute', axis=1, inplace=True)

y_train = train_df['BeatsPerMinute']

print(f"all_data shape: {all_data.shape}")


all_data['square_AudioLoudness'] = (all_data['AudioLoudness'] ** 2)
all_data['log_square_AudioLoudness'] = np.log1p(all_data['square_AudioLoudness'] + 1)

sns.histplot(data=all_data, x='log_square_AudioLoudness', kde=True, bins='fd')


pt_VocalContent = PowerTransformer(method='box-cox')
all_data[['box_cox_VocalContent']] = pt_VocalContent.fit_transform(all_data[['VocalContent']])

sns.histplot(data=all_data, x='box_cox_VocalContent', kde=True, bins='fd')


pt_AcousticQuality = PowerTransformer(method='yeo-johnson')  # works with 0 and negative values
all_data[['yj_AcousticQuality']] = pt_AcousticQuality.fit_transform(all_data[['AcousticQuality']])

sns.histplot(data=all_data, x='yj_AcousticQuality', kde=True, bins='fd')


pt_InstrumentalScore = PowerTransformer(method='yeo-johnson') 
all_data[['yj_InstrumentalScore']] = pt_InstrumentalScore.fit_transform(all_data[['InstrumentalScore']])

sns.histplot(data=all_data, x='yj_InstrumentalScore', kde=True, bins='fd')


pt_LivePerformanceLikelihood = PowerTransformer(method='box-cox')
all_data[['LivePerformanceLikelihood']] = pt_LivePerformanceLikelihood.fit_transform(all_data[['LivePerformanceLikelihood']])

sns.histplot(data=all_data, x='LivePerformanceLikelihood', kde=True, bins='fd')


all_data.head()


X = all_data[:ntrain]
X_test_final = all_data[ntrain:]
y = y_train


def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 500, 2000)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    num_leaves = trial.suggest_int('num_leaves', 20, 255)
    feature_fraction = trial.suggest_float('feature_fraction', 0.4, 1.0)
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0)
    bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
    lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
    lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 100)
    
    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=12)
    rmses = []

    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            min_data_in_leaf=min_data_in_leaf,
            n_jobs=-1,
            random_state=12
        )

        # Fit model (no early stopping)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        rmses.append(rmse)

    return np.mean(rmses)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best hyperparameters:", best_params)


final_model = LGBMRegressor(
    **best_params,
    n_jobs=-1,
    random_state=12
)

final_model.fit(X, y)


y_pred_final = final_model.predict(X_test_final)


sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

# Replace target column with predictions
sub["BeatsPerMinute"] = y_pred_final  

# Save submission
sub.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

