!pip install lime -q
!pip install shap -q
!pip install eli5 -q


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
plt.rcParams["figure.figsize"] = (16, 6)


import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import category_encoders as ce
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from catboost.utils import get_gpu_device_count
import warnings
warnings.filterwarnings("ignore")

import eli5
import shap
import lime


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


train.head()


train.info()


train_extra = pd.concat([train, train_extra])


train.set_index(['id'], inplace=True)
test.set_index(['id'], inplace=True)
train_extra.set_index(['id'], inplace=True)


train.describe()


def info(col):
    print('-' * 30, f'{col}', '-' * 30, sep='')
    
    print(f'Train dtype: {train[col].dtype}, Test dtype: {test[col].dtype}')
    
    print(f'NaN values: Train: {train[col].isna().sum()}, Test: {test[col].isna().sum()}')
    
    unique_values_train = train[col].unique()
    unique_values_test = test[col].unique()
    
    print(f'Nunique: Train: {len(unique_values_train)}, Test: {len(unique_values_test)}')
    if len(unique_values_train) < 20:
        print(f'Unique values: {", ".join(map(str, unique_values_train))}')
    
    print('-' * (60 + len(col)), end='\n'*2)


for col in train.columns[:-1]:
    info(col)


train.head()


cat_param = [
    'Brand',
    'Material',
    'Size',
    'Laptop Compartment',
    'Waterproof',
    'Style',
    'Color'
]

num_param = [
    'Weight Capacity (kg)',
    'Compartments',
    'Price'
]


def fillna_cat(df, coefficient=0.7, train=True):
    for col in cat_param:
        top_values = df[col].value_counts().index.tolist()
        
        probabilities = [coefficient] + [(1 - coefficient) / (len(top_values) - 1)] * (len(top_values) - 1)
        
        fill_value = np.random.choice(top_values, p=probabilities)
        
        df.loc[df[col].isna(), col] = fill_value
        if train:
            print(f"Column: {col}, Most freq: {fill_value}")
    return df


train = fillna_cat(train)
train_extra = fillna_cat(train_extra, train=False)
test = fillna_cat(test, train=False)


def fillna_num(df, train=True):
    for col in num_param[:-1]:
        median = df[col].median()
        df.loc[df[col].isna(), col] = median
        if train:
            print(f"Column: {col}, Most freq: {median}")
    return df


train = fillna_num(train)
train_extra = fillna_num(train_extra, train=False)
test = fillna_num(test, train=False)


def duplicates(df):
    old_shape = df.shape
    df.drop_duplicates(keep='last', inplace=True)
    if (old_shape == df.shape):
        print("# No duplicates")
    else:
        print(f"# Duplicates found, {old_shape[0] - df.shape[0]} num.")
    return df


train = duplicates(train)


train_extra = duplicates(train_extra)


train_extra['dataset'] = 'train'
test['dataset'] = 'test'

visualisation_df = pd.concat([train_extra, test], axis=0)

train_extra.drop(columns='dataset', inplace=True)
test.drop(columns='dataset', inplace=True)


for feature in cat_param:
    sns.countplot(data = visualisation_df, x=feature, hue='dataset', palette='summer')
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {feature}')
    plt.show()


for features in num_param:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(data=visualisation_df, y='dataset', x=features, ax=ax[0], orient='h', palette='hot')
    sns.violinplot(data=visualisation_df, y='dataset', x=features, ax=ax[1], palette='summer')
    plt.show()


for features in num_param[:-1]:
    sns.histplot(data=visualisation_df, hue='dataset', x=features, kde=True, bins=50)
    plt.show()


correlation_matrix = train.corr(numeric_only=True) 
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', mask=mask)
plt.title('Corr matrix')
plt.show()


X = train_extra.loc[:, cat_param+num_param[:-1]]
target = train_extra[num_param[-1]]
test = test.loc[:, cat_param+num_param[:-1]]


continuous = [
    'Weight Capacity (kg)'
]
discrete = [
    'Compartments'
]
cat_param = list(set(cat_param) - set(continuous + discrete))


TE = ce.TargetEncoder(smoothing=20, cols=cat_param)

train_encoded = TE.fit_transform(X[cat_param], target)

test_encoded = TE.transform(test[cat_param])

for col in cat_param:
    X[f'TE_{col}'] = train_encoded[col]
    test[f'TE_{col}'] = test_encoded[col]


for cat_col in cat_param:
    for num_col in continuous + discrete:
        intera_col = f'{cat_col}_x_{num_col}'
        X[intera_col] = X[cat_col].astype(str) + '_' + X[num_col].astype(str)
        test[intera_col] = test[cat_col].astype(str) + '_' + test[num_col].astype(str)
        X[intera_col] = X[intera_col].astype('category').cat.codes
        test[intera_col] = test[intera_col].astype('category').cat.codes
    X[cat_col] = X[cat_col].astype('category').cat.codes
    test[cat_col] = test[cat_col].astype('category').cat.codes


scaler = RobustScaler().set_output(transform="pandas")

scaler.fit(X)

X = scaler.transform(X)
test = scaler.transform(test)


# scaler = StandardScaler()

# X['Weight Capacity (kg)'] = scaler.fit_transform(X[['Weight Capacity (kg)']])  
# test['Weight Capacity (kg)'] = scaler.transform(test[['Weight Capacity (kg)']])


X.head()


print(f'GPU: {get_gpu_device_count()}')


X_sample = X.sample(frac=0.7, random_state=42)
target_sample = target[X_sample.index]


def objective_catboost(trial):
    max_depth = trial.suggest_int("max_depth", 3, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    n_estimators = trial.suggest_int("n_estimators", 1000, 2000)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    random_strength = trial.suggest_float("random_strength", 0, 10)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1)
    border_count = trial.suggest_int("border_count", 32, 255)

    model = CatBoostRegressor(
        task_type='GPU',
        devices='0:1', 
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        border_count=border_count,
        silent=True
    )

    score = abs(cross_val_score(model, X_sample, target_sample, cv=3, scoring="neg_root_mean_squared_error")).mean()
    return score

study = optuna.create_study(direction="minimize")
study.optimize(objective_catboost, n_trials=30)


!pip install plotly -q


plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_contour(study, params=["max_depth", "learning_rate"]).show()
plot_contour(study, params=["n_estimators", "learning_rate"]).show()
plot_slice(study, params=["max_depth", "learning_rate", "n_estimators"]).show()


params = study.best_params
catboost_model = CatBoostRegressor(
        task_type='GPU',
        devices='0:1', 
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_strength=params['random_strength'],
        bagging_temperature=params['bagging_temperature'],
        border_count=params['border_count'],
        silent=True
    )
catboost_model.fit(X, target)


def objective_xgb(trial):
    max_depth=trial.suggest_int("max_depth", 3, 10)
    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    min_child_weight=trial.suggest_int("min_child_weight", 1, 100)
    subsample=trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0)
    n_estimators=trial.suggest_int("n_estimators", 500, 1500)
    reg_alpha=trial.suggest_float("reg_alpha", 0, 1)
    reg_lambda=trial.suggest_float("reg_lambda", 0, 1)

    model = XGBRegressor(
        tree_method="gpu_hist",
        random_state=42,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )

    scores = cross_val_score(
        model, 
        X_sample, 
        target_sample, 
        cv=3,  
        scoring="neg_mean_squared_error"  
    )

    rmse_scores = np.sqrt(-scores)  
    
    return rmse_scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=30)



model_xgb = XGBRegressor(
    tree_method="gpu_hist",
    enable_categorical=True,
    random_state=42,
    **study.best_trial.params
)


model_xgb.fit(
    X, 
    target,
    eval_metric="rmse",
    verbose=False
)


explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X.iloc[:200000, ])

shap.summary_plot(shap_values, X.iloc[:200000, ])


shap.initjs()
print('Correct answer:', target.iloc[2])
shap.force_plot(explainer.expected_value, shap_values[2,:], X.iloc[2,:])


pred = model_xgb.predict(test)
# pred = catboost_model.predict(test)


submission = pd.DataFrame({
    'id': test.index,
    'Price': pred
})
submission.to_csv('baseline.csv', index=False)


submission

