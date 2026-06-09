import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import optuna


train       = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', index_col='id')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', index_col='id')
test        = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', index_col='id')

train = pd.concat([train, train_extra], axis=0, ignore_index=True)


train.info()


CAT_COLS = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
NUM_COLS = ['Compartments', 'Weight Capacity (kg)']


print(train.isnull().mean())


for col in train.columns:
    print(col, train[col].nunique())


fig, ax = plt.subplots(2, 4, figsize=(20, 10))

for i, col in enumerate(CAT_COLS):
    train[col].value_counts().plot.pie(ax=ax[i//4, i%4], autopct='%.2f%%', title=col)
    ax[i//4, i%4].set_ylabel('')

fig.delaxes(ax[1, 3])


for col in CAT_COLS:
    print(train.groupby(col)['Price'].mean())
    print()


print(train[NUM_COLS + ['Price']].describe())


fig, axes = plt.subplots(1, len(NUM_COLS) + 1, figsize=(20, 5))

for i, col in enumerate(NUM_COLS + ['Price']):
    train[col].plot(kind='hist', ax=axes[i], title=col)
    axes[i].set_ylabel('')

plt.tight_layout()
plt.show()


for col in NUM_COLS + ['Price']:
    q1  = train[col].quantile(0.25)
    q3  = train[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f'{col}: {train[(train[col] < lower_bound) | (train[col] > upper_bound)].shape[0]} outliers')


correlation_matrix = train[NUM_COLS + ['Price']].corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)

plt.title('Correlation Heatmap for Numerical Features')
plt.show()


train[CAT_COLS] = train[CAT_COLS].fillna('unknown')
test[CAT_COLS]  = test[CAT_COLS].fillna('unknown')


train = pd.get_dummies(train, columns=CAT_COLS)
test  = pd.get_dummies(test, columns=CAT_COLS)


train_sample = train.sample(frac=0.1, random_state=42)
imputer = KNNImputer(n_neighbors=5)
imputer.fit(train_sample[NUM_COLS])

train[NUM_COLS] = imputer.transform(train[NUM_COLS])
test[NUM_COLS]  = imputer.transform(test[NUM_COLS])


bins   = [0, 10, 20, 30, float('inf')]
labels = [1, 2, 3, 4]

train['Weight Category'] = pd.cut(train['Weight Capacity (kg)'], bins=bins, labels=labels, right=False)
test['Weight Category']  = pd.cut(test['Weight Capacity (kg)'], bins=bins, labels=labels, right=False)

train = train.drop(columns=['Weight Capacity (kg)'])
test  = test.drop(columns=['Weight Capacity (kg)'])


train = pd.get_dummies(train, columns=['Compartments', 'Weight Category'], drop_first=True)
test = pd.get_dummies(test, columns=['Compartments', 'Weight Category'], drop_first=True)


train.head()


train_sample = train.sample(frac=0.1, random_state=42)
target       = train_sample['Price']
predictors   = train_sample.drop(columns=['Price'])

train_predictors, eval_predictors, train_target, eval_target = train_test_split(predictors, target, train_size=0.7, random_state=42)


models = {
    'LR'  : LinearRegression(),
    'KNN' : KNeighborsRegressor(),
    'HGB' : HistGradientBoostingRegressor(),
    'ADA' : AdaBoostRegressor(),
    'RF'  : RandomForestRegressor(),
    'XGB' : XGBRegressor(),
    'CB'  : CatBoostRegressor(verbose=0)
}


for model in models:
    print('Training', model)
    models[model].fit(train_predictors, train_target)


eval_predictions_base = np.mean(eval_target)
loss_base = mean_squared_error(eval_target, [eval_predictions_base] * len(eval_target))

print(f'Base mean: {np.sqrt(loss_base):.2f}')


eval_predictions_base = np.median(eval_target)
loss_base = mean_squared_error(eval_target, [eval_predictions_base] * len(eval_target))

print(f'Base median: {np.sqrt(loss_base):.2f}')


model_losses = {}

for model in models:
    eval_predictions = models[model].predict(eval_predictors)
    loss = mean_squared_error(eval_target, eval_predictions)
    model_losses[model] = np.sqrt(loss)

sorted_losses = sorted(model_losses.items(), key=lambda x: x[1])

print('Evaluation Losses:')

for model, loss in sorted_losses:
    print(f'{model}: {loss:.2f}')


target       = train['Price']
predictors   = train.drop(columns=['Price'])
train_predictors, eval_predictors, train_target, eval_target = train_test_split(predictors, target, train_size=0.7, random_state=42)


def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 300),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 127),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-10, 1e-3, log=True),
        'max_bins': trial.suggest_int('max_bins', 128, 255)
    }

    model = HistGradientBoostingRegressor(**params, loss='squared_error', random_state=42)
    model.fit(train_predictors, train_target)
    eval_predictions = model.predict(eval_predictors)
    loss = mean_squared_error(eval_target, eval_predictions)
    return np.sqrt(loss)

study = optuna.create_study(direction='minimize', study_name='HGB Regression')
optuna.logging.set_verbosity(optuna.logging.DEBUG)
study.optimize(objective, n_trials=20)

best_params = study.best_params
best_score  = study.best_value

print("Best parameters:", best_params)
print("Best score:", best_score)


final_predictors = train.drop(columns=['Price'])
final_targets    = train['Price']

final_model = HistGradientBoostingRegressor(**best_params)
final_model.fit(final_predictors, final_targets)


price_predictions = final_model.predict(test)

test['Price'] = price_predictions
test[['Price']].to_csv('s5e2-submission.csv', index=True)

