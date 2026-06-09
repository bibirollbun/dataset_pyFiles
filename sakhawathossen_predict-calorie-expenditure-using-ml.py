import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import make_scorer

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')





train_df.head(5)


train_df.info()


# Sex  = object /categorical datatype
# remaining all are numerical datatype


train_df.describe


train_df.isnull().sum()


plt.figure(figsize=(10, 5))
sns.histplot(train_df['Calories'], bins=50, kde=True)
plt.title('Distribution of Calories Burned', fontsize=16, weight='bold')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


corr = train_df[cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


le  = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])


train_df


train_df['BMI'] = train_df['Weight'] / ((train_df['Height']/100)**2)
train_df['HR_per_min'] = train_df['Heart_Rate']/train_df['Duration']
train_df['tem_diff_from_norm'] = train_df['Body_Temp'] - 37.0


train_df


test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df['BMI'] = test_df['Weight'] / ( (test_df['Height'] / 100) ** 2 )
test_df['HR_per_min'] = test_df['Heart_Rate'] / test_df['Duration']
test_df['Temp_diff_from_norm'] = test_df['Body_Temp'] - 37.0


test_df


train_df['id']


train_df = train_df.drop(['id', 'Age', 'Height','Weight'], axis=1)



train_df


y = train_df['Calories']


y


X = train_df.drop(columns=['Calories'])


X


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'criterion':['gini','entropy'],
    'max_depth' : [3,5,10,None],
    'min_samples_split' : [2,4,5,10],
    'min_samples_leaf': [1,2,4,8,10],
    'max_features' : ['sqrt','log2',None]
 }

#applpying GridSearchCv
grid_search = GridSearchCV(estimator = dt , param_grid = param_grid , cv=5 ,scoring = 'accuracy',n_jobs = -1 ,verbose = 1)

grid_search.fit(X_train,y_train)



best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)




# 6. Feature Importance Plot
importances = best_model.feature_importances_
feature_names = X.columns

# Sort and plot
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', figsize=(12, 6), title='Feature Importances')
plt.tight_layout()
plt.show()


# 5. Predict and evaluate
y_pred = best_model.predict(X_valid)

y_pred


# 6. Calculate loss metrics
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_valid, y_pred)



print("Best Parameters:", grid_search.best_params_)
print("MSE (Loss Function):", mse)
print("RMSE:", rmse)
print("MAE:", mae)


rmse/y_valid.mean() *100

#If the target values are in the hundreds or thousands, then 12.7 might be very good.

#Check: RMSE / Mean of y_test â†’ if this ratio is <10â€“15%, thatâ€™s acceptable.


from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor(random_state = 42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_valid)

print("RD RMSE ",np.sqrt(mean_squared_error(y_valid,rf_pred)))


np.sqrt(mean_squared_error(y_valid,rf_pred))/y_valid.mean() *100



# === RANDOM FOREST ===
rf = RandomForestRegressor(random_state=42)



rf_param_grid = {
    'n_estimators': [100, 200,500],
    'max_depth': [2,4,5, 7,8,10, None],
    'min_samples_split': [2, 4,6,5,8, 10],
    'min_samples_leaf': [1, 2, 4,5,6,7,8],
    'max_features': ['sqrt', 'log2']
}



rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid,
                       cv=3, scoring='neg_mean_squared_error',
                       n_jobs=-1, verbose=1)


# find best estimators 
rf_grid.fit(X_train,y_train)
rf_best = rf_grid.best_estimator_


rf_pred = rf_best.predict(X_valid)

rf_mse = mean_squared_error(y_test,rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test,rf_pred)


print("\nðŸŽ¯ Random Forest Best Parameters:", rf_grid.best_params_)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest MSE:", rf_mse)
print("Random Forest MAE:", rf_mae)


rf_rmse/y_valid.mean() *100


import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid,
                        cv=3, scoring='neg_mean_squared_error',
                        n_jobs=-1, verbose=1)








xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_


# Predictions & Evaluation
xgb_pred = xgb_best.predict(X_valid)
xgb_mse = mean_squared_error(y_valid, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_valid, xgb_pred)


print("\nðŸš€ XGBoost Best Parameters:", xgb_grid.best_params_)
print("XGBoost RMSE:", xgb_rmse)
print("XGBoost MSE:", xgb_mse)
print("XGBoost MAE:", xgb_mae)


xgb_rmse/y_valid.mean() *100








































