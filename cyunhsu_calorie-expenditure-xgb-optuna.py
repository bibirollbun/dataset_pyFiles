!pip install optuna
!pip install optuna[visualization]


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer
import warnings
import optuna.visualization as vis

warnings.filterwarnings("ignore")


import pandas as pd

train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train.head()


train.drop(columns=['id'], inplace=True)


test_id = test['id'].copy()
test = test.drop(columns=['id'])


print('train shape:', train.shape,'test shape:',test.shape)
train.info()
train.describe()


# Target Variable Analysis (Calories)

sns.histplot(train['Calories'], bins=50, kde=True)
plt.title("Calories Distribution")
plt.show()
print("Skewness:", train['Calories'].skew())


if train['Sex'].dtype == 'object':
    map_dict = {'male': 0, 'female': 1}
    train['Sex'] = train['Sex'].map(map_dict)
    test['Sex']  = test['Sex'].map(map_dict)


# Print counts
print('train_sex_counts:\n', train['Sex'].value_counts())
print('\ntest_sex_counts:\n', test['Sex'].value_counts())

# Plot with custom colors
train['Sex'].value_counts().plot(kind='bar', color=['#FF6EB4', '#4A90E2'])
plt.title('Gender Distribution')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


numeric_cols = train.select_dtypes('number').columns.drop('Calories')
n = len(numeric_cols)

fig, axs = plt.subplots(n, 2, figsize=(12, 4 * n))

for i, col in enumerate(numeric_cols):
    sns.histplot(train[col], ax=axs[i, 0], kde=True, bins=30)
    axs[i, 0].set_title(f'Hist – {col}')

    sns.boxplot(x=train[col], ax=axs[i, 1])
    axs[i, 1].set_title(f'Box – {col}')

plt.tight_layout()


corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


train.corr()['Calories'].sort_values(ascending=False)


def add_features(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['HeartTemp'] = df['Heart_Rate'] * df['Body_Temp']
    df['Effort'] = df['Heart_Rate'] * df['Duration']
    df['TempEffort'] = df['Body_Temp'] * df['Duration']
    return df

train = add_features(train)
test = add_features(test)


engineered_cols = ['BMI', 'Effort', 'HeartTemp', 'TempEffort']
sns.heatmap(train[engineered_cols + ['Calories']].corr(), annot=True)


train['BMI'] = train['Weight'] / ((train['Height'] / 100) ** 2)
sns.scatterplot(x='BMI', y='Calories', data=train)
plt.title("Calories Burned vs BMI")


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
for fold, (tr, va) in enumerate(kfold.split(train)):
    model = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=fold,
    )
    model.fit(train.iloc[tr, :-1], train.iloc[tr, -1])
    oof[va] = model.predict(train.iloc[va, :-1])
print("Baseline 5-fold RMSLE:", rmsle(train["Calories"], oof))


# Define features and targets
X = train.drop(columns=['Calories'])
y = train['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }

    model = XGBRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(mean_squared_log_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)
    return np.sqrt(-np.mean(scores))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)


best_params = study.best_params
best_params.update({
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
})

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
final_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dvalid, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=False
)


print("Best Params:", best_params)


print(f'Best CV RMSLE = {study.best_value:.5f}')

final_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dvalid, 'valid')],
    early_stopping_rounds=10)

print(f'Final model trees = {final_model.best_iteration}')


# Optimization history
fig1 = vis.plot_optimization_history(study)
fig1.show()

# Parameter importances
fig2 = vis.plot_param_importances(study)
fig2.show()

# Parameter interaction
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()

# Slice plot
fig4 = vis.plot_slice(study)
fig4.show()


importance = final_model.get_score(importance_type='gain')


importance_df = pd.Series(importance).sort_values(ascending=False).to_frame('Gain')
importance_df.index.name = 'Feature'
importance_df.reset_index(inplace=True)

print(importance_df)


xgb.plot_importance(final_model, importance_type='gain')
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.show()



y_pred = final_model.predict(dvalid)
y_pred = np.maximum(0, y_pred)

from sklearn.metrics import mean_squared_log_error
rmsle_final = np.sqrt(mean_squared_log_error(y_test, y_pred))
print("Final RMSLE:", rmsle_final)



dsubmit = xgb.DMatrix(test)

y_submit = final_model.predict(dsubmit)
y_submit = np.maximum(0, y_submit)

submission = pd.DataFrame({
    'id': test_id,
    'Calories': y_submit
})

submission.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")

# submission preview
submission.head()

