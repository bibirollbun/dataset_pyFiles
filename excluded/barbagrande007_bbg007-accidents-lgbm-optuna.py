import warnings
warnings.filterwarnings('ignore')

import math
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv', index_col='id')
submit = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


def create_summary(df):
    describe = df.describe().transpose()
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary["MissingValues"] = df.isna().sum()
    summary["UniqueValues"] = df.nunique()
    summary["Value_1"] = df.iloc[0]
    summary["Value_2"] = df.iloc[1]
    summary["Value_3"] = df.iloc[2]
    summary = pd.concat([summary, describe], axis=1)
    
    return summary

create_summary(train)


# Map labels to angles
def cyclic_conversion(df):
    mapping = {
        "morning": 0,
        "afternoon": 2*math.pi/3,   # 120 degrees
        "evening": 4*math.pi/3      # 240 degrees
    }
    
    # Create cos/sin features
    df["theta"] = df["time_of_day"].map(mapping)
    df["time_cos"] = df["theta"].apply(math.cos)
    df["time_sin"] = df["theta"].apply(math.sin)

    return df

train = cyclic_conversion(train)
test = cyclic_conversion(test)


train.head()


cols = 4
rows = int(np.ceil(len(train.columns) / cols))

fig,ax = plt.subplots(nrows=rows,ncols=cols,figsize=(20,20))
ax = ax.flatten()

plt.suptitle("Visualize all features",size=24, y=1.01)

for i,col in enumerate(train.columns):
    if train[col].dtype == float or train[col].dtype == int:
        sns.boxplot(data=train,y=col,ax=ax[i],orient="vertical")
        ax[i].set_title(f"{col}")
    else:
        sns.countplot(data=train,x=col,ax=ax[i])
        ax[i].set_title(f"{col}")
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)

# Remove empty subplots
for i in range(len(train.columns), len(ax)):
    fig.delaxes(ax[i])

plt.tight_layout()
plt.show()


X = train.drop('accident_risk', axis=1)
y = train['accident_risk']

X_test = test.copy()


num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


preprocessor = ColumnTransformer([
    ('scaler', StandardScaler(), num_cols),
    ('encoder', OneHotEncoder(), cat_cols),
],remainder='drop')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=51)


models = {
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
    'Lasso': Lasso(),
    'DecisionTree': DecisionTreeRegressor(),
    'KNeighbors': KNeighborsRegressor(),
    'LightGBM': lgb.LGBMRegressor(verbose=-1),
    'XGBoost': xgb.XGBRegressor(),
}

results = {}
# Loop through the models and fit them
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

    pipeline.fit(X_train, y_train)
    
    # Cross-validation
    scores = cross_val_score(pipeline, X_val, y_val, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    
    # print(f"{name} RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    results[name] = rmse_scores.mean()
    
    # Predict on validation set
    y_pred = pipeline.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # print(f"{name} Validation RMSE: {rmse:.2f}\n")


# Visualize the results in a bar plot
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE'])
results_df = results_df.sort_values(by='RMSE', ascending=True)
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='RMSE', data=results_df)
plt.xticks(rotation=45)
plt.title('Model RMSE Comparison')
plt.xlabel('Model')
plt.ylabel('RMSE')

# Add result values on top of bars
for index, value in enumerate(results_df['RMSE']):
    plt.text(index, value + 0.001, f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Set optuna logging level
# optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):

    # Define hyperparameters to tune
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 51,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    model = lgb.LGBMRegressor(**params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    # Cross-validation
    scores = -1 * cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(scores).mean()

    return rmse_scores

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, n_jobs=-1)

# Get the best hyperparameters
best_params = study.best_params
best_value = study.best_value
print("Best Hyperparameters: ", best_params)
print("Best RMSE: ", best_value)


best_model = lgb.LGBMRegressor(**best_params, verbose=-1, objective='regression', metric='rmse', random_state=51)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print(f"Best Model RMSE: {rmse:.2f}\n")


final_model = lgb.LGBMRegressor(**best_params, verbose=-1, objective='regression', metric='rmse', random_state=51)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', final_model)])
pipeline.fit(X, y)


y_test = pipeline.predict(X_test)


submit['accident_risk'] = y_test


sns.histplot(train['accident_risk'], label='Train Data', kde=True, multiple='dodge')
sns.histplot(submit['accident_risk'], label='Test Data', kde=True, multiple='dodge')
plt.title('Distribution of Predicted Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Density')
plt.legend()
plt.show()


submit.to_csv('submission_bbg007.csv', index=False)

