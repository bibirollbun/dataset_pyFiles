import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train


test


train.info()


train.nunique()


train.isna().sum()


print(train['road_type'].unique())
print(train['lighting'].unique())
print(train['weather'].unique())
print(train['time_of_day'].unique())


train = pd.get_dummies(train, columns=['road_type', 'lighting', 'weather', 'time_of_day'], drop_first=True)
test = pd.get_dummies(test, columns=['road_type', 'lighting', 'weather', 'time_of_day'], drop_first=True)


train.columns


train[["road_signs_present", "public_road", "holiday", "school_season", 
    "road_type_rural", "road_type_urban", "lighting_dim", "lighting_night", 
    "weather_foggy", "weather_rainy", "time_of_day_evening", "time_of_day_morning"]] = train[["road_signs_present", "public_road", "holiday", 
                                                                                           "school_season", "road_type_rural", "road_type_urban", "lighting_dim", "lighting_night", 
                                                                                           "weather_foggy", "weather_rainy", "time_of_day_evening", "time_of_day_morning", ]].astype(int)



test[["road_signs_present", "public_road", "holiday", "school_season", 
    "road_type_rural", "road_type_urban", "lighting_dim", "lighting_night", 
    "weather_foggy", "weather_rainy", "time_of_day_evening", "time_of_day_morning"]] = test[["road_signs_present", "public_road", "holiday", 
                                                                                           "school_season", "road_type_rural", "road_type_urban", "lighting_dim", "lighting_night", 
                                                                                           "weather_foggy", "weather_rainy", "time_of_day_evening", "time_of_day_morning", ]].astype(int)


train


X = train.drop(['id', 'accident_risk'], axis=1)
y = train['accident_risk']


test


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

oof_scores = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train, 
              eval_set = [(X_val, y_val)],
              callbacks = [lgb.early_stopping(20)])
    
    preds = model.predict(X_val)
    err = mean_squared_error(y_val, preds, squared=False)
    print("RMSE for this fold", err)
    oof_scores.append(err)

print(f"Average validation RMSE: {np.mean(oof_scores):.4f}")




final_model = lgb.LGBMRegressor()
final_model.fit(X, y)



X_test = test.drop(columns=['id'])
final_preds = final_model.predict(X_test)



submission = pd.DataFrame({
    'id': test['id'],
    'accident_risk': final_preds
})

submission





submission.to_csv('submission.csv', index=False)
submission.head()




