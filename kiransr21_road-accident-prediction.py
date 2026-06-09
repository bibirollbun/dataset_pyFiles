import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Reading the csv file
df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


# Checking the Data type and Count of each column
df.info()
#df_test.info()
#df_submission.info()


df.head()


# Checking the null values
df.isnull().sum()





## check all the unique values in it

def getUnique(columns):
    result = {}
    for col in columns:
        result[col] = df[col].unique().tolist()
    return result

unique_values = getUnique(['road_type', 'lighting', 'weather', 'time_of_day'])
print(unique_values)


sns.violinplot(data=df, x=df['road_type'], y =df['accident_risk'])


sns.violinplot(data=df, x=df['time_of_day'], y =df['accident_risk'])


sns.violinplot(data=df, x=df['weather'], y =df['accident_risk'])


sns.violinplot(data=df, x=df['lighting'], y =df['accident_risk'])


 sns.lineplot(y=df['curvature'], x=df['accident_risk'], data=df)


df[['curvature','accident_risk']].corr()


sns.kdeplot(df[df['road_signs_present'] == True]['accident_risk'], label='True', fill=True)
sns.kdeplot(df[df['road_signs_present'] == False]['accident_risk'], label='False', fill=True)
plt.title('Accident Risk Distribution by Road sign')
plt.legend()
plt.show()


sns.kdeplot(df[df['holiday'] == True]['accident_risk'], label='True', fill=True)
sns.kdeplot(df[df['holiday'] == False]['accident_risk'], label='False', fill=True)
plt.title('Accident Risk Distribution by holiday')
plt.legend()
plt.show()


sns.kdeplot(df[df['school_season'] == True]['accident_risk'], label='True', fill=True)
sns.kdeplot(df[df['school_season'] == False]['accident_risk'], label='False', fill=True)
plt.title('Accident Risk Distribution by holiday')
plt.legend()
plt.show()


df.info()


df.columns
             


plt.figure(figsize=(8,6))
plt.scatter(df['curvature'], df['accident_risk'], c=df['speed_limit'], cmap='coolwarm', s=50)
plt.colorbar(label='Speed Limit')
plt.xlabel('Curvature')
plt.ylabel('Accident Risk')
plt.title('Accident Risk vs Curvature (colored by Speed Limit)')
plt.show()


plt.figure(figsize=(8,6))
plt.hexbin(df['curvature'], df['accident_risk'], 
           C=df['speed_limit'], 
           gridsize=30, cmap='coolwarm')
plt.colorbar(label='Speed Limit')
plt.xlabel('Curvature')
plt.ylabel('Accident Risk')
plt.title('Accident Risk vs Curvature (Speed Limit shown by color)')
plt.show()


from sklearn.preprocessing import OneHotEncoder





categorical_cols = ['road_type', 'lighting', 'weather','time_of_day']

# Initialize encoder
encoder = OneHotEncoder(drop='first', sparse=False)

# Fit and transform
encoded = encoder.fit_transform(df[categorical_cols])

# Create DataFrame from encoded output
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns from main DataFrame
df = df.drop(columns=categorical_cols)

# Concatenate the encoded columns back
df = pd.concat([df, encoded_df], axis=1)

bool_columns = ['road_signs_present', 'public_road','holiday', 'school_season']
df[bool_columns] = df[bool_columns].astype(int)





df['Risk_Factor'] = df['curvature'] * df['speed_limit']
df['visibility_factor'] = df['weather_rainy'] + df['weather_foggy'] + df['lighting_night']
df['traffic_factor'] = df['school_season'] * (df['time_of_day_morning'] + df['time_of_day_evening'])
df['control_factor'] = df['road_signs_present'] * df['speed_limit']
df['exposure_factor'] = df['public_road'] * df['speed_limit']
df['speed_visibility_factor'] = df['speed_limit'] * (1 - df['visibility_factor'])


#df.info()
df.columns


corr = df.corr()


mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 8))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap (Lower Triangle)")
plt.show()


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


X = df.drop(columns=['id', 'accident_risk'], errors='ignore')
y = df['accident_risk']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(
    n_estimators=350,     
    learning_rate=0.02,   
    max_depth=7,  #6
    min_child_weight=3,
    subsample=0.8,        
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=1.2,
    random_state=42    #42
)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='rmse',
              verbose=False,
              early_stopping_rounds=100)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")



from catboost import CatBoostRegressor

cat_model = CatBoostRegressor(
    iterations=700,
    learning_rate=0.05,
    depth=7,
    random_seed=42,
    verbose=100
)
cat_model.fit(X_train, y_train,
             eval_set=(X_train, y_train),   # use a holdout split
    early_stopping_rounds=200)


y2_pred = cat_model.predict(X_test)

mae = mean_absolute_error(y_test, y2_pred)
rmse = np.sqrt(mean_squared_error(y_test, y2_pred))
r2 = r2_score(y_test, y2_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


import matplotlib.pyplot as plt
import xgboost as xgb

xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain', title='Top Features for XGBoost')
plt.show()



import matplotlib.pyplot as plt
import catboost as cat

feature_names =['num_lanes', 'curvature', 'speed_limit', 'road_signs_present',
       'public_road', 'holiday', 'school_season', 'num_reported_accidents',
       'road_type_rural', 'road_type_urban', 'lighting_dim',
       'lighting_night', 'weather_foggy', 'weather_rainy',
       'time_of_day_evening', 'time_of_day_morning', 'Risk_Factor',
       'visibility_factor', 'traffic_factor', 'control_factor',
       'exposure_factor','speed_visibility_factor']

feature_importance = cat_model.get_feature_importance()

# Create a DataFrame for easier plotting
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)





plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('CatBoost Feature Importance')
plt.gca().invert_yaxis() # To display the most important feature at the top
plt.show()


df_test.info()


categorical_cols_test = ['road_type', 'lighting', 'weather','time_of_day']

# Initialize encoder
encoder = OneHotEncoder(drop='first', sparse=False)

# Fit and transform
encoded = encoder.fit_transform(df_test[categorical_cols_test])

# Create DataFrame from encoded output
encoded_test = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols_test))

# Drop original categorical columns from main DataFrame
df_test = df_test.drop(columns=categorical_cols_test)

# Concatenate the encoded columns back
df_test = pd.concat([df_test, encoded_test], axis=1)

bool_columns = ['road_signs_present', 'public_road','holiday', 'school_season']
df_test[bool_columns] = df_test[bool_columns].astype(int)

# featured columns

df_test['Risk_Factor'] = df_test['curvature'] * df_test['speed_limit']
df_test['visibility_factor'] = df_test['weather_rainy'] + df_test['weather_foggy'] + df_test['lighting_night']
df_test['traffic_factor'] = df_test['school_season'] * (df_test['time_of_day_morning'] + df_test['time_of_day_evening'])
df_test['control_factor'] = df_test['road_signs_present'] * df_test['speed_limit']
df_test['exposure_factor'] = df_test['public_road'] * df_test['speed_limit']
df_test['speed_visibility_factor'] = df_test['speed_limit'] * (1 - df_test['visibility_factor'])


test = df_test
test_ids = test['id'] 

# Drop ID before prediction
X_test_final = test.drop(columns=['id'])



X_test_final.columns


y_pred_test = xgb_model.predict(X_test_final)
y2_pred_test = cat_model.predict(X_test_final)


submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y2_pred_test
})

submission['accident_risk'] = submission['accident_risk'].clip(0, 1)
submission.to_csv('submission.csv', index=False)


submission.head()




