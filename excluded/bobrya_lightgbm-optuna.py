import warnings


warnings.filterwarnings('ignore')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
import numpy as np
import lightgbm
from phik import phik_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


RANDOM_STATE = 0
TEST_SIZE = 0.25


train_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


train_data.date = pd.to_datetime(train_data.date)
train_data['year'] = train_data.date.dt.year
train_data['month'] = train_data.date.dt.month
train_data['day'] = train_data.date.dt.day
train_data = train_data.dropna(subset='num_sold')
train_data.num_sold = train_data.num_sold.astype('int32')
train_data = train_data[train_data['num_sold'] < 2200]
train_data.country = train_data.country.astype('category')
train_data.store = train_data.store.astype('category')
train_data['product'] = train_data['product'].astype('category')


test_data.date = pd.to_datetime(test_data.date)
test_data['year'] = test_data.date.dt.year
test_data['month'] = test_data.date.dt.month
test_data['day'] = test_data.date.dt.day
test_data.country = test_data.country.astype('category')
test_data.store = test_data.store.astype('category')
test_data['product'] = test_data['product'].astype('category')


def pie_chart(data, column, **kwargs):
    data[column].value_counts().plot(
        kind='pie',
        figsize=(10, 10),
        autopct='%1.1f%%',
        wedgeprops={'width': 0.5},
        ylabel='',
        **kwargs
    )
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.3, 1), loc='upper right', fontsize=11)
    plt.show()


def hist_chart(data, column, **kwargs):
    data[column].plot(
        kind='hist',
        figsize=(10, 10),
        color='#BA55D3',
        **kwargs    
    )
    plt.axvline(x=data[column].mean(), color='red', linestyle='--', label='Mean value')
    plt.axvline(x=data[column].median(), color='green', linestyle='--', label='Median value')
    plt.legend()
    plt.show()


def boxplot_chart(data, column, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(
        x=column,
        data=data,
        ax=ax,
        palette=['#BA55D3'],
        flierprops={
            'marker': 'o',
            'markersize': 10,
            'markerfacecolor': 'blue',
            'markeredgecolor': 'black'
        }
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def line_chart(data, index, values, aggfunc, title, xlabel, ylabel):
    data.pivot_table(
        index=index,
        values=values,
        aggfunc=aggfunc).plot(     
            style='o-',
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=(10, 6),
            markerfacecolor='blue',
            markeredgecolor='black',
            markersize=8,
            linewidth=2,
            fontsize=14
        )
    
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


train_data.info()


train_data.head()


print(train_data.count()/len(train_data))


pie_chart(train_data, 'country')


pie_chart(train_data, 'store')


pie_chart(train_data, 'product')


hist_chart(train_data, 'num_sold')
boxplot_chart(train_data, 'num_sold', '', '', '')


line_chart(train_data, 'year', 'num_sold', 'count', '', '', '')


line_chart(train_data, 'month', 'num_sold', 'count', '', '', '')


line_chart(train_data, 'day', 'num_sold', 'count', '', '', '')


train_data.info()


plt.figure(figsize=(10, 8))
sns.heatmap(phik_matrix(train_data[[
    'country',
    'store',
    'product',
    'year',
    'month',
    'num_sold'
]]), annot=True, cmap='coolwarm')
plt.show()


X = train_data.drop(['id', 'num_sold', 'date'], axis=1)
y = train_data.num_sold


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=RANDOM_STATE,
    test_size=TEST_SIZE
)


cat_cols = [
    'country',
    'store',
    'product']
num_cols = [
    'year',
    'month',
    'day']
feature_cols = cat_cols + num_cols
target_col = 'num_sold'


def objective(trial):
    dtrain = lightgbm.Dataset(X_train, label=y_train)
    param = {
        'objective': 'regression',
        'metric': 'MAPE',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 0,
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1.),
        'num_leaves': trial.suggest_int('num_leaves', 2, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 180),
        'n_estimators': trial.suggest_int('n_estimators', 20, 300),
        'importance_type': trial.suggest_categorical('importance_type', ['split', 'gain']),
    }
    # you can add something more in param

    gbm = lightgbm.train(param, dtrain)
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)
    mape = mean_absolute_percentage_error(y_test, pred_labels)
    return mape


%%time

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200) # change n_trials for number of iterations

print("Number of finished trials: {}".format(len(study.trials)))

best_mape = study.best_trial.value
print(f"The best MAPE value is: {best_mape}")

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {} = {},".format(key, value))


model = lightgbm.LGBMRegressor(
    learning_rate = 0.060073740016653845,
    num_leaves = 293,
    max_depth = 171,
    n_estimators = 127,
    importance_type = 'split',
    random_state = RANDOM_STATE)


model.fit(X_train, y_train, categorical_feature=cat_cols)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, feature_names=feature_cols, plot_type="bar")


prediction = model.predict(test_data[feature_cols])


test_data['num_sold'] = prediction
test_data.num_sold = test_data.num_sold.astype('int32')


test_data.info()

