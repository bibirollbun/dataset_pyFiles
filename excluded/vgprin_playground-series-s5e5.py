import sys
sys.path.append('../src')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def quick_eda(df):
    """
    Perform a quick exploratory data analysis (EDA) on a pandas DataFrame.
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze.
    Outputs:
    --------
    - Prints the shape of the DataFrame (number of rows and columns).
    - Prints the number of duplicate rows in the DataFrame.
    - Displays the first few rows of the DataFrame as a sample.
    - Prints a summary of data types, non-missing counts, missing counts, 
      and the percentage of missing values for each column.
    Notes:
    ------
    This function uses `display()` to show the DataFrame and summary table, 
    which is particularly useful in Jupyter Notebook environments.
    """
    
    # dataframe shape
    print(f'Shape: {df.shape[0]} rows and {df.shape[1]} columns')

    # duplicates check
    dupes = df.duplicated().sum()
    print(f'Duplicates check: {dupes} duplicate rows found\n')

    # sample data
    print('Sample data:')
    display(df.head())

    # data types and (non-)missing count
    print('Data types and missing count:')    
    info_df = pd.DataFrame({
        'dtype': df.dtypes,
        'non_missing': df.count(),
        'missing': df.isnull().sum(),
        'missing_pct': round(df.isnull().mean() * 100, 2)
    })
    display(info_df)

    # summarise the dataset
    print('Summary:')
    display(df.describe().round(2))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
quick_eda(df_train)


plt.figure(figsize=(6, 6))

sns.heatmap(
    df_train.drop(columns='id').corr(numeric_only=True),
    cmap='coolwarm_r',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.01,
    annot=True,
    annot_kws={'fontsize':7, 'color': 'black'},
    fmt='.2f'
)

plt.title('Correlation heatmap', fontsize=16)
plt.tight_layout()
plt.show()


df_train.drop(columns='id').hist(figsize=(8, 6), bins=30)
plt.tight_layout()
plt.show()


df_train['Age_sqr'] = df_train['Age'] ** 2
df_train['Sex_male'] = (df_train['Sex'] == 'male').astype(int)
df_train['Duration_Male'] = df_train['Duration'] * df_train['Sex_male']
df_train['log_Duration'] = np.log1p(df_train['Duration'])
df_train['log_Heart_Rate'] = np.log1p(df_train['Heart_Rate'])
df_train['Duration_HeartR'] = df_train['Duration'] * df_train['Heart_Rate']
df_train['Duration_BodyT'] = df_train['Duration'] * df_train['Body_Temp']
df_train['HeartR_BodyT'] = df_train['Heart_Rate'] * df_train['Body_Temp']
df_train['BMI'] = df_train['Weight'] / (df_train['Height'] / 100) ** 2
df_train['BMI_Duration'] = df_train['BMI'] * df_train['Duration']
df_train['BMI_Heart_Rate'] = df_train['BMI'] * df_train['Heart_Rate']
df_train['Temp_dev'] = df_train['Body_Temp'] - 37.0
df_train['Heart_Age_ratio'] = df_train['Heart_Rate'] / df_train['Age']


from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import xgboost as xgb
import lightgbm as lgb


target = 'Calories'
features = [col for col in df_train.columns if col not in ['id', target, 'Sex']]

x = df_train[features]
y = df_train[target]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.35, random_state=42)

y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)

print(f'Treino: {x_train.shape}')
print(f'Validação: {x_valid.shape}')


scaler = StandardScaler()

k_features = 15
feature_selector = SelectKBest(score_func=f_regression, k=k_features)

lgbm = lgb.LGBMRegressor(random_state=42)

pipeline = Pipeline([
    ('scaler', scaler),
    ('selector', feature_selector),
    ('model', lgbm)
])


param_distributions = {
    'selector__k': [10, 12, 15, 18, 'all'],
    'model__n_estimators': [100, 200, 300, 500, 700],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__num_leaves': [15, 31, 50, 70],
    'model__max_depth': [-1, 5, 10, 15],
    'model__reg_alpha': [0, 0.01, 0.1, 0.5],
    'model__reg_lambda': [0, 0.01, 0.1, 0.5],
    'model__colsample_bytree': [0.7, 0.8, 0.9, 1.0], 
    'model__subsample': [0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)


random_search.fit(x_train, y_train_log)

print('-' * 30)
print(f'Melhores parâmetros encontrados:')
print(random_search.best_params_)
print('-' * 30)
print(f'Melhor score (neg MSE) na validação cruzada: {random_search.best_score_:.4f}')
print('-' * 30)

best_pipeline = random_search.best_estimator_


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


y_pred_log = best_pipeline.predict(x_valid)
y_pred = np.expm1(y_pred_log)
score = rmsle(y_valid, y_pred)

print('-' * 30)
print(f'RMSLE (validação): {score:.5f}')
print('-' * 30)


best_pipeline.fit(x, np.log1p(y))


selector_estimator = xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=100)
feature_selector = SelectFromModel(
    selector_estimator,
    max_features=18,
    threshold=-np.inf
)

xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')

new_pipeline = Pipeline([
    ('scaler', scaler),
    ('selector', feature_selector),
    ('model', xgb_model)
])


param_grid = {
    'model__n_estimators': [250, 300, 350],
    'model__learning_rate': [0.07, 0.1, 0.13],
    'model__max_depth': [5, 6, 7],
    'model__subsample': [1.0],
    'model__colsample_bytree': [0.9],
    'model__reg_alpha': [0.5],
    'model__reg_lambda': [0.1]
}

grid_search = GridSearchCV(
    estimator=new_pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=-1
)


grid_search.fit(x_train, y_train_log)

print('-' * 30)
print(f'Melhores parâmetros encontrados:')
print(grid_search.best_params_)
print('-' * 30)
print(f'Melhor score (neg MSE) na validação cruzada: {grid_search.best_score_:.4f}')
print('-' * 30)

best_pipeline_xgb = grid_search.best_estimator_


y_pred_log_xgb = best_pipeline_xgb.predict(x_valid)
y_pred_xgb = np.expm1(y_pred_log_xgb)
score_xgb = rmsle(y_valid, y_pred_xgb)

print('-' * 30)
print(f'RMSLE (validação): {score_xgb:.5f}')
print('-' * 30)


best_pipeline_xgb.fit(x, np.log1p(y))


df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
quick_eda(df_test)


df_test['Age_sqr'] = df_test['Age'] ** 2
df_test['Sex_male'] = (df_test['Sex'] == 'male').astype(int)
df_test['Duration_Male'] = df_test['Duration'] * df_test['Sex_male']
df_test['log_Duration'] = np.log1p(df_test['Duration'])
df_test['log_Heart_Rate'] = np.log1p(df_test['Heart_Rate'])
df_test['Duration_HeartR'] = df_test['Duration'] * df_test['Heart_Rate']
df_test['Duration_BodyT'] = df_test['Duration'] * df_test['Body_Temp']
df_test['HeartR_BodyT'] = df_test['Heart_Rate'] * df_test['Body_Temp']
df_test['BMI'] = df_test['Weight'] / (df_test['Height'] / 100) ** 2
df_test['BMI_Duration'] = df_test['BMI'] * df_test['Duration']
df_test['BMI_Heart_Rate'] = df_test['BMI'] * df_test['Heart_Rate']
df_test['Temp_dev'] = df_test['Body_Temp'] - 37.0
df_test['Heart_Age_ratio'] = df_test['Heart_Rate'] / df_test['Age']


y_test_log = best_pipeline_xgb.predict(df_test[features])
y_test = np.expm1(y_test_log)

submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission['Calories'] = y_test
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()

