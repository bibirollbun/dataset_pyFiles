import pandas as pd
import numpy as np

train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train_df.head()


train_df.info()


train_copy = train_df.copy()
test_copy = test_df.copy()


train_copy["curve_limit_interaction"] = train_copy["curvature"] * train_copy["speed_limit"]
test_copy["curve_limit_interaction"] = test_copy["curvature"] * test_copy["speed_limit"]


train_copy['speed_per_lane'] = train_copy['speed_limit'] / (train_copy['num_lanes'] + 1e-6)
test_copy['speed_per_lane'] = test_copy['speed_limit'] / (test_copy['num_lanes'] + 1e-6)


train_copy['accidents_per_lane'] = train_copy['num_reported_accidents'] / (train_copy['num_lanes'] + 1e-6)
test_copy['accidents_per_lane'] = test_copy['num_reported_accidents'] / (test_copy['num_lanes'] + 1e-6)


cat_nums = ["road_type", "lighting", "weather","time_of_day"]

train_copy = pd.get_dummies(train_copy, columns = cat_nums, dtype = int)
test_copy = pd.get_dummies(test_copy, columns = cat_nums, dtype = int)


train_copy = train_copy.replace({True: 1, False: 0})
test_copy = test_copy.replace({True: 1, False: 0})


x_train = train_copy.drop(["id", "accident_risk"], axis = 1)

y_train = train_copy["accident_risk"]

x_test = test_copy.drop("id", axis = 1)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

X_train, X_val, Y_train, Y_val = train_test_split(
    scaled_train,
    y_train,
    test_size = 0.2,
    random_state = 42
)


from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()

linear_model.fit(X_train, Y_train)

linear_predicts = linear_model.predict(X_val)

mse_score = mean_squared_error(Y_val, linear_predicts)
rmse_score = np.sqrt(mse_score)

print(f"RMSE Score: {rmse_score:.4f}")



from sklearn.ensemble import RandomForestRegressor

fr_model = RandomForestRegressor(random_state = 42, n_estimators = 100)

fr_model.fit(X_train, Y_train)

fr_predicts = fr_model.predict(X_val)

mse_score = mean_squared_error(Y_val, fr_predicts)
rmse_score = np.sqrt(mse_score)

print(f"RMSE Score: {rmse_score:.4f}")



import numpy as np
from lightgbm import LGBMRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import time 


lgb_params = {
    'n_estimators': [200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1], 
    'max_depth': [5, 10, 15, -1],      
    'num_leaves': [20, 31, 40],       
    'colsample_bytree': [0.7, 0.8, 1.0] 
}


lgb_model = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1) 


random_search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=lgb_params,
    n_iter=10,  
    cv=5,       
    n_jobs=-2,  
    scoring="neg_mean_squared_error", 
    verbose=2,  
    random_state=42
)

print("LGBM RandomizedSearch starting...")
start_time = time.time()


random_search_lgb.fit(X_train, Y_train)

end_time = time.time()
print("---")
print(f"LGBM Search completed. Time: {(end_time - start_time) / 60:.2f} minute")
print("---")


print("Best parameters:")
print(random_search_lgb.best_params_)


best_rmse = np.sqrt(-random_search_lgb.best_score_)
print(f"\n Best CV (Cross-Val) RMSE Score: {best_rmse:.4f}")


best_lgb_model = random_search_lgb.best_estimator_
val_predicts = best_lgb_model.predict(X_val)
final_rmse = np.sqrt(mean_squared_error(Y_val, val_predicts))

print(f"\nFinal RMSE score on X_val: {final_rmse:.4f}")


import pandas as pd


best_model = random_search_lgb.best_estimator_


final_test_data = scaled_test


test_ids = test_df['id']


print("Making Predicts...")
test_predictions = best_model.predict(final_test_data)


test_predictions[test_predictions < 0] = 0


submission_df = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_predictions
})


submission_df.to_csv('submission.csv', index=False)


print(submission_df.head())


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


importances = best_lgb_model.feature_importances_


feature_names = X_train.columns


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})


importance_df = importance_df.sort_values(by='Importance', ascending=False)


print("--- Important Features ---")
print(importance_df)
print("-----------------------------------------")



plt.figure(figsize=(10, 8)) 
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df.head(20) 
)
plt.title('Top 20 Most Important Features (LGBM)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout() 





