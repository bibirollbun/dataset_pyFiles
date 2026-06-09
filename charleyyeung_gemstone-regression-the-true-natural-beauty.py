import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


test = pd.read_csv('/kaggle/input/playground-series-s3e8/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s3e8/train.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s3e8/sample_submission.csv')


print(test.info())
print(train.info())
print(sample.info())


pd.DataFrame(train.describe()).T


test.describe()


sample.describe()


print(train.isna().sum().sum())


print(train.duplicated().sum())


train.drop('id', axis = 1, inplace = True)


cat_col = train.columns[train.dtypes == 'object']
num_col = [col for col in train.columns if col not in cat_col]


print(cat_col)
print(num_col)


n_rows = (len(train.columns)+2)//3
fig, axes = plt.subplots(n_rows+1, 3, figsize = (22,n_rows*7))
axes = axes.flatten()

fig.suptitle(f'Train Data Distributions\n\n',
         ha='center',  fontsize=28)
for i, col in enumerate(train.columns):
    if col in cat_col:
        sns.countplot(data = train, x = col, ax = axes[i])
        axes[i].set_title(f"{col} Count Graph", fontsize = 18)
        axes[i].set_xlabel(col, fontsize = 14)
        axes[i].bar_label(axes[i].containers[0])
    else:
        sns.histplot(data = train, x = col, shrink = 0.8, kde = True, ax = axes[i])
        axes[i].set_title(f"{col} Histogram", fontsize = 18)
        axes[i].set_xlabel(col, fontsize = 14)

for j in range(len(axes) - len(train.columns)):
    axes[len(train.columns) + j].set_visible(False)

plt.tight_layout()
plt.show();



fig, axes = plt.subplots(2,2, figsize = (22,14))
axes = axes.flatten()
for i, col in enumerate(['carat','x','y','z']):
    sns.scatterplot(data = train, x = col,y = 'price', ax = axes[i])
    axes[i].set_title(f"{col} Boxplot", fontsize = 18)
    axes[i].set_xlabel(col, fontsize = 14)
plt.tight_layout()
plt.show();
    
    


matrix = train[num_col].corr()
sns.heatmap(matrix, cmap="Greens", annot=True)
plt.show();


X = train.drop(columns = 'price', axis = 1)
y = train['price']
num_col.remove('price')


print(cat_col)
print(num_col)


class OutlierClipper:
    def __init__(self, columns, lower_quantile=0.00, upper_quantile=0.99):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.bounds[col] = {
                'lower': X[col].quantile(self.lower_quantile),
                'upper': X[col].quantile(self.upper_quantile)
            }
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.columns:
            bounds = self.bounds[col]
            X_[col] = X_[col].clip(bounds['lower'], bounds['upper'])
            X_.loc[X_[col] == 0, col] = bounds['lower']
        return X_


    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('outlier_clipper', OutlierClipper(columns=num_col)),
        ('scaler', RobustScaler())
    ]), num_col),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_col)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', None)  
])


param_grids = {
    'linear': {'model': [LinearRegression()]},
    
    'lasso': {
        'model': [Lasso(max_iter=1000)], 
        'model__alpha': [0.1, 1, 10]
    },
    
    'ridge': {
        'model': [Ridge()], 
        'model__alpha': [0.1, 1, 10]
    },
    
    'rf': {
        'model': [RandomForestRegressor(random_state=4)],
        'model__n_estimators': [50], 
        'model__max_depth': [5, None],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    },
    
    'xgb': {
        'model': [XGBRegressor(random_state=4)],
        'model__n_estimators': [50],
        'model__max_depth': [3, 5],
        'model__learning_rate': [0.1],
        'model__subsample': [0.8],
        'model__colsample_bytree': [0.8]
    }
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


%%time
best_models = {}

all_results = []
for model_name, params in param_grids.items():
    print(f"\nRunning GridSearchCV for {model_name}...")
    
    grid = GridSearchCV(model_pipeline, params, cv=5, 
                        scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    results_df = pd.DataFrame(grid.cv_results_)
    results_df['model_name'] = model_name
    all_results.append(results_df)
    
    val_pred = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, val_pred))
    
    best_models[model_name] = {
        'model': grid.best_estimator_,
        'rmse': rmse,
        'params': grid.best_params_
    }


best_model_name = min(best_models, key=lambda x: best_models[x]['rmse'])
final_model = best_models[best_model_name]['model']


combined_results = pd.concat(all_results, ignore_index=True)
output_df = combined_results[['model_name', 'params', 'mean_test_score', 'std_test_score']]
output_df = output_df.sort_values('mean_test_score', ascending=False)
output_df = output_df.rename(columns={
    'mean_test_score': 'RMSE',
    'std_test_score': 'RMSE_std'
})
output_df['RMSE'] = -output_df['RMSE']  
output_df['RMSE_std'] = output_df['RMSE_std']

rows = []
for model_name, model_info in best_models.items():
    model = model_info['model']
    
    test_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, test_pred)
    mse = mean_squared_error(y_test, test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_pred)
    
    rows.append({
        'Model name': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    })

output_table = pd.DataFrame(rows)

output_table = output_table.sort_values(by='RMSE').reset_index(drop=True)

def highlight_rmse(s):
    return ['font-weight: bold;  \
    background-color: lightgrey' if s.name == 'RMSE' else '' for _ in s]

output_table.style.apply(highlight_rmse)


best_model_index = output_df['RMSE'].idxmin()
best_model_row = output_df.loc[best_model_index]
best_model_name = best_model_row['model_name']
best_model_params = best_model_row['params']

print(f"\nBest model: {best_model_name}")
print(f"\nBest parameters: {best_model_params}")
print(f"\nBest RMSE: {best_model_row['RMSE']}")

final_model = best_models[best_model_name]['model']
test_pred = final_model.predict(X_test)


def evaluate_model(true, predicted, model_name):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    table = pd.DataFrame({'Model name': [model_name],
                         'Mean Absolute Error': mae,
                         'Mean Squared Error':mse,
                         'Root Mean Squared Error': rmse,
                         'R²': r2_square})
                         
    return table
    
result_table = evaluate_model(y_test, test_pred, best_model_name) 
result_table


print(f"RMSE is {round((rmse/test_pred.mean())*100,4)}% of the prediction mean {test_pred.mean()}")


pred_table = pd.DataFrame({'First 10 predictions':test_pred[:10]})
pred_table




