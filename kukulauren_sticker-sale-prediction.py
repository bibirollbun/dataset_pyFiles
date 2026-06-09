import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


test_ids = test['id'].copy()


submission=pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
submission


train.info()


train.describe(include='all')


train['num_sold'].isna().any().any()


train = train.dropna(subset=['num_sold']).drop_duplicates(subset=['id'])


print(train['num_sold'].isnull().sum())  # should print 0


train['date'] = pd.to_datetime(train['date'])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['year_month'] = train['date'].dt.to_period('M')



test['date'] = pd.to_datetime(test['date'])
test['year'] = test['date'].dt.year
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['year_month'] = test['date'].dt.to_period('M')


monthly_avg_by_year = train.groupby(['year', 'month'])['num_sold'].mean().unstack(level=0)


monthly_avg_by_year


years = monthly_avg_by_year.columns
n_years = len(years)

fig, axes = plt.subplots(n_years, 1, figsize=(10, 4 * n_years), sharex=True)

for i, year in enumerate(years):
    ax = axes[i] if n_years > 1 else axes
    monthly_avg_by_year[year].plot(ax=ax, marker='o', linestyle='-', color='b')
    ax.set_title(f'Average Review Rating for {year}')
    ax.set_ylabel('Avg Rating')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True)

plt.xlabel('Month')
plt.tight_layout()
plt.show()


train['country'].unique()


import seaborn as sns
ax=sns.barplot(x='country',y='num_sold',data=train)


train['store'].unique()


ax=sns.barplot(x='store',y='num_sold',data=train)


store_country_sales = train.groupby(['store', 'country'])['num_sold'].sum().reset_index()
plt.figure(figsize=(12, 8))
sns.barplot(x='country', y='num_sold', hue='store', data=store_country_sales, dodge=True)

plt.xlabel('Total Sales (num_sold)')
plt.ylabel('Store Type')
plt.title('Sales per Store Type in Different Countries')


train['product'].unique()
sns.barplot(x='product',y='num_sold',data=train)
plt.xticks(rotation=45, ha="right")


train=train.drop(['date','id','year_month'], axis=1)


test = test.drop(['date','id','year_month'], axis=1)


test


cat_cols = ['country', 'store', 'product'] 
  
for col in cat_cols: 
    temp = pd.get_dummies(train[col]).astype('int') 
    train = pd.concat([train, temp], axis=1) 
    
  
train.drop(cat_cols, axis=1, inplace=True) 
print(train.head())


cat_cols = ['country', 'store', 'product'] 
  
for col in cat_cols: 
    temp_test = pd.get_dummies(test[col]).astype('int') 
    test = pd.concat([test, temp_test], axis=1) 
    
  
test.drop(cat_cols, axis=1, inplace=True) 
print(test.head())


X = train.drop('num_sold', axis=1)
y = train['num_sold']


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
kf = KFold(n_splits=5, shuffle=True, random_state=42)


models = {
    'CatBoost': CatBoostRegressor(verbose=100, random_seed=42,  early_stopping_rounds=100),
    'XGBoost': XGBRegressor(max_depth=10, colsample_bytree=0.7, subsample=0.9, n_estimators=100, learning_rate=0.02,
                            gamma=0.01, max_delta_step=2, early_stopping_rounds=100, eval_metric='rmse',
                            enable_categorical=True, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, min_samples_split=2, max_depth=10,random_state=42)
}


results = {name: {'oof': np.zeros(len(train)), 'pred': np.zeros(len(test)), 'rmsle': []} for name in models}


test


print(train.isnull().sum())  # should print 0


for name, model in models.items():
    print(f"Training Model: {name}")
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold} ---")
        
        # Splitting the data
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        x_test = test.copy()
        
        # Training depending on model type
        if name == 'XGBoost':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)
        elif name == 'CatBoost':
            model.fit(x_train, y_train, eval_set=(x_valid, y_valid))
        else:
            model.fit(x_train, y_train)
        
        # Predictions
        oof_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)
        
        # Store predictions
        results[name]['oof'][valid_idx] = oof_pred
        results[name]['pred'] += test_pred / kf.n_splits
        
        # Evaluate performance
        rmsle = np.sqrt(mean_squared_log_error(y_valid, np.maximum(0, oof_pred)))
        results[name]['rmsle'].append(rmsle)
        print(f"Fold {fold} RMSLE: {rmsle:.4f}")


print("\n=== Final Model Comparison ===")
for name in models:
    mean_rmsle = np.mean(results[name]['rmsle'])
    std_rmsle = np.std(results[name]['rmsle'])
    print(f"{name}: Mean RMSLE = {mean_rmsle:.4f} ± {std_rmsle:.4f}")


from sklearn.model_selection import GridSearchCV
rf_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 15,20, None],
    'min_samples_split': [2,4,5],
    'min_samples_leaf': [1, 2],
    'max_features': [ 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')


rf_grid_search.fit(X, y)


print(f"Best parameters for RandomForest: {rf_grid_search.best_params_}")
print(f"Best score for RandomForest: {rf_grid_search.best_score_}")


import numpy as np
print("Best RMSE score:", np.sqrt(abs(rf_grid_search.best_score_)))


catboost_param_grid = {
    'iterations': [500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [6, 10, 12],
    'l2_leaf_reg': [1, 3, 5],
}

catboost_model = CatBoostRegressor(random_seed=42, verbose=0)

catboost_grid_search = GridSearchCV(estimator=catboost_model, param_grid=catboost_param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')


catboost_grid_search.fit(X, y)


print(f"Best parameters for CatBoost: {catboost_grid_search.best_params_}")
print(f"Best score for CatBoost: {catboost_grid_search.best_score_}")


import numpy as np
print("Best RMSE score:", np.sqrt(abs(catboost_grid_search.best_score_)))


catboost_tuned_model = CatBoostRegressor(depth= 10, iterations= 500, l2_leaf_reg= 5, learning_rate= 0.01)
catboost_tuned_model.fit(X,y)


test


y_preds=catboost_tuned_model.predict(test)


import pandas as pd
prediction = pd.DataFrame({
    'id': test_ids,
    'predicted_value': y_preds
})


prediction


prediction.to_csv('submission.csv', index=False)


print(submission.head())

print(f"\nPredict Mean: {y_preds.mean():.2f}")
print(f"Predict Median: {np.median(y_preds):.2f}")

