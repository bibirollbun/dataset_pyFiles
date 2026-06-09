# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install catboost  --upgrade scikit-learn


train_df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')
print(train_df.shape)
train_df.head()


test_df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')
print(test_df.shape)
test_df.head()


sub_df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')
sub_df.head()


train_df = train_df.drop(columns=['ID'])
sub_id = test_df['ID']
test_df = test_df.drop(columns=['ID'])


train_df.info(verbose=True)


test_df.info(verbose=True)


for col in train_df.columns:
  print(train_df[col].value_counts())


train_nan = train_df.isna().sum()
train_nan[train_nan>0]


test_nan = test_df.isna().sum()
test_nan[test_nan>0]


import matplotlib.pyplot as plt
import seaborn as sns

  
plt.figure(figsize=(8, 5))
sns.displot(data=train_df['y'], kind="hist", bins=30, kde=True, color='skyblue', height=6, aspect=1.5)
plt.show()


train_df = train_df[train_df['y']<175]


cat_cols = train_df.select_dtypes(include=['object']).columns.to_list()
cat_cols


num_cols = test_df.select_dtypes(exclude=['object']).columns.to_list()
num_cols



# Plot every category feature versus the output (time)
for col in cat_cols:
    plt.figure(figsize=(30, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=col, y='y', data=train_df)
    plt.title(f'Time vs {col}')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


# Plot the distribution of each category feature
for col in cat_cols:
    plt.figure(figsize=(30, 5))

    plt.subplot(1, 2, 2)
    sns.countplot(x=col, data=train_df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()


from scipy.stats import f_oneway

for col in cat_cols:
    groups = [train_df[train_df[col] == cat]['y'] for cat in train_df[col].unique()]

    if all(len(g) > 0 for g in groups):
        f_stat, p_val = f_oneway(*groups)
        print(f"{col}: p-value = {p_val:.4f}")
    else:
        print(f"{col}: Insufficient data for ANOVA")


cat_cols.remove('X4')
train_df = train_df.drop(columns=['X4'])
test_df = test_df.drop(columns=['X4'])


from sklearn.preprocessing import TargetEncoder

target_encoder = TargetEncoder()

train_df[cat_cols] = target_encoder.fit_transform(train_df[cat_cols], train_df['y'])
test_df[cat_cols] = target_encoder.transform(test_df[cat_cols])


train_df[cat_cols]


test_df[cat_cols]


for col in cat_cols:
    plt.figure(figsize=(8, 5))
    sns.displot(data=train_df[col], kind="hist", bins=30, kde=True, color='skyblue', height=6, aspect=1.5)
    plt.show()


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df2 = df.copy()
    df2 = df2[(df2[col] >= lower_bound) & (df2[col] <= upper_bound)]
    return df2


for col in cat_cols :
  train_df = remove_outliers(train_df,col)


corr_matrix = train_df[cat_cols+['y']].corr()

plt.figure(figsize=(30, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


for col in num_cols:
    plt.figure(figsize=(15, 10))
    sns.boxplot(x=col, y='y', data=train_df)
    plt.title(f'Time vs {col}')
plt.tight_layout()
plt.show()


from scipy.stats import ttest_ind

num_cols_to_remove=[]
for col in num_cols:

    time_0 = train_df[train_df[col] == 0]['y']
    time_1 = train_df[train_df[col] == 1]['y']

    if len(time_0)>1 and len(time_1)>1:

      t_stat, p_val = ttest_ind(time_0, time_1, equal_var=False)

      if p_val >= 0.05:
          num_cols_to_remove.append(col)
          print(f"{col}: p-value = {p_val:.4f}")

    else :
      num_cols_to_remove.append(col)


print(num_cols_to_remove)


train_df = train_df.drop(columns=num_cols_to_remove)
test_df = test_df.drop(columns=num_cols_to_remove)


for c in num_cols_to_remove:
  num_cols.remove(c)


train_df.head()


test_df.head()


from sklearn.preprocessing import MinMaxScaler, StandardScaler

train_scaler = MinMaxScaler()

train_df[cat_cols] = train_scaler.fit_transform(train_df[cat_cols])
test_df[cat_cols] = train_scaler.transform(test_df[cat_cols])



target_scaler = StandardScaler()

train_df['y'] = target_scaler.fit_transform(train_df[['y']])


X = train_df.drop(columns=['y'])
y = train_df['y']


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV


# models = {
#     'Linear Regression': LinearRegression(),
#     'Ridge': Ridge(alpha=1.0),
#     'Lasso': Lasso(alpha=1.0),
#     'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
#     'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
#     'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
#     'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
#     'SVR': SVR(kernel='rbf', C=1.0),
#     'KNN': KNeighborsRegressor(n_neighbors=5),
#     'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
#     'CatBoost': cb.CatBoostRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, verbose=0),
#     'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
#     'MLP' : MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1, max_iter=500)
# }


# cv_folds = 5
# scoring = 'r2'

# results = {}
# for name, model in models.items():
  
#     # R² scores
#     r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)

#     results[name] = {
#         'R2 Mean': np.mean(r2_scores),
#         'R2 Std': np.std(r2_scores)
#     }


# results_df = pd.DataFrame(results).T
# results_df.sort_values(by='R2 Mean',ascending=False)


# catboost_model = cb.CatBoostRegressor(random_state=42, verbose=0)

# param_grid = {
#     'iterations': [100, 500, 1000],
#     'learning_rate': [0.01, 0.1],
#     'depth': [4, 6],
#     'l2_leaf_reg': [1, 3, 5],
#     'bagging_temperature': [0.5, 1.0]
# }


# nn_model = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)

# param_grid = {
#     'hidden_layer_sizes': [
#         (100, 50),
#         (100, 100),         
#         (200, 100),          
#         (100, 50, 25),       
#         (200, 100, 50),      
#         (150, 150, 100),    
#         (300, 200, 100)      
#     ],  
#     'activation': ['relu', 'tanh'],  
#     'learning_rate_init': [0.001, 0.01],  
#     'alpha': [0.001, 0.01],  
#     'max_iter': [300, 500],  
#     'solver': ['adam'], 
#     'batch_size': ['auto', 32]  
# }


# rf_model = RandomForestRegressor(random_state=42)

# param_grid = {
#     'n_estimators': [100, 200, 300],  
#     'max_depth': [10, 20, 30, None],  
#     'min_samples_split': [2, 5, 10],  
#     'min_samples_leaf': [1, 2, 4],  
#     'max_features': ['auto', 'sqrt', 0.3],  
#     'bootstrap': [True, False]  
# }


# grid_search = GridSearchCV(
#     estimator=rf_model,
#     param_grid=param_grid,
#     cv=3,  
#     scoring='r2',  
#     n_jobs=-1,  
#     verbose=1
# )

# grid_search.fit(X, y)


# print("Best parameters:", grid_search.best_params_)
# print("Best cross-validated R² score:", grid_search.best_score_)


catboost_best_params = {'bagging_temperature': 0.5, 'depth': 4, 'iterations': 1000, 'l2_leaf_reg': 5, 'learning_rate': 0.01}

best_catboost_model = cb.CatBoostRegressor(random_state=42, verbose=0, **catboost_best_params)
# best_catboost_model.fit(X, y)


# nn_best_params = {'activation': 'tanh', 'alpha': 0.01, 'batch_size': 32, 'hidden_layer_sizes': (100, 50, 25), 'learning_rate_init': 0.01, 'max_iter': 300, 'solver': 'adam'}

# best_nn_model = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1, **nn_best_params)
# best_nn_model.fit(X, y)


rf_best_params = {'bootstrap': True, 'max_depth': 10, 'max_features': 0.3, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}

best_rf_model = RandomForestRegressor(random_state=42, **rf_best_params)
# best_rf_model.fit(X, y)


from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


voting_regressor = VotingRegressor(
    estimators=[
        ('rf', best_rf_model),
        ('catboost', best_catboost_model)
    ],
    weights=None
)

param_grid = {
    'weights': [
        [1, 1],         
        [1, 2],          
        [2, 1],          
        [1.5, 1],       
        [1, 1.5]       
    ]
}

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

def r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return r2_score(y, y_pred)


grid_search = GridSearchCV(
    estimator=voting_regressor,
    param_grid=param_grid,
    cv=kf,
    scoring=r2_scorer,  
    n_jobs=-1, 
    verbose=1,
    error_score='raise'
)


grid_search.fit(X, y)


print("Best weights:", grid_search.best_params_['weights'])
print("Best cross-validated R² score:", grid_search.best_score_)


best_voting_model = grid_search.best_estimator_


y_pred_scaled = best_voting_model.predict(test_df)
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_pred


submission_df = pd.DataFrame({'ID': sub_id, 'y': y_pred.flatten()})
submission_df.to_csv('mercedes_sub.csv', index=False)




