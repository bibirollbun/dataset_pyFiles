from IPython.display import Image, display
display(Image(filename='/kaggle/input/kdkdkff/Screenshot 2025-07-24 040803.png', width=800))



import pandas as pd
df=pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_train.csv')
df


df.columns


df_test=pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_test.csv')
df_test


df_test.info()


df.info()


df.describe()


mis = df.isna().mean() * 100 > 70
df[df.columns[mis]].isna().mean() * 100


mis = df.isna().mean() * 100 < 70
df[df.columns[mis]].isna().mean() * 100


df['last_service_date'] = pd.to_datetime(df['last_service_date'], errors='coerce')


df['damage_type']


df['damage_cost']


df['year'].describe()


df['year'].value_counts().sort_index().plot(kind='bar')


df['damage_cost']=df['damage_cost'].fillna(0)
df['damage_type']=df['damage_type'].fillna('No Damage')


df['damage_type'].describe()


df.info()


df['warranty_expired'] = ((df['warranty_years'] + df['year']) < 2025).astype(int)
df['warranty_expired'].describe()


df['num_owners'].describe()


def get_state(row):
    if row['num_owners'] == 1 and row['mileage'] < 5000:
        return 'lightly_used' 
    elif row['num_owners'] >= 3 or row['mileage'] > 15000:
        return 'heavily_used'  
    else:
        return 'moderately_used'  

df['state'] = df.apply(get_state, axis=1)

df['state'].info()


df['state'].value_counts().sort_index().plot(kind='pie')


df['performance_score'] = (df['horsepower'] + df['torque']) / df['zero_to_60_s']
df['performance_score'].describe()



df['months_since_service'] = ((pd.to_datetime("2025-07-21") - pd.to_datetime(df['last_service_date'])).dt.days // 30)


df['ownership_score'] = df['num_owners'] * df['mileage']
df['ownership_score'].describe()


def usage_level(score):
    if score < 8386:
        return 'low'
    elif score <= 30627:
        return 'medium'
    else:
        return 'high'

df['usage_level'] = df['ownership_score'].apply(usage_level)
df['usage_level'].describe()


import matplotlib.pyplot as plt
import seaborn as sns


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 8))
df[numeric_cols].boxplot(rot=90)
plt.title('Outlier of numeric_cols')
plt.show()



cat = df.select_dtypes(include=['object']).columns
cat


import matplotlib.pyplot as plt
df[numeric_cols].plot(kind = "box" , subplots = True , figsize = (15,20) , layout = (3,8))
plt.show()


df['mileage'].describe()


df['price'].describe()


    Q1 = df['performance_score'].quantile(0.25)
    Q3 = df['performance_score'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['performance_score'] < lower_bound) | (df['performance_score'] > upper_bound)]


    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]


    Q1 = df['damage_cost'].quantile(0.25)
    Q3 = df['damage_cost'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['damage_cost'] < lower_bound) | (df['damage_cost'] > upper_bound)]


import matplotlib.pyplot as plt
df[numeric_cols].plot(kind = "box" , subplots = True , figsize = (15,20) , layout = (3,8))
plt.show()


df['performance_score'].describe()


df['damage_cost'].describe()


df['non_original_parts'].unique()


df['non_original_parts'].describe()


df.to_csv("car_data.csv", index=False)


from sklearn.preprocessing import StandardScaler

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('price') 

scaled_df = df.copy()


scaler = StandardScaler()
scaled_df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


df['state'].unique()


df=df.drop('id',axis=1)


from sklearn.preprocessing import LabelEncoder

nominal = ['brand', 'color', 'engine_config', 'transmission', 'drivetrain',
                'market_region', 'interior_material', 'brake_type', 'tire_brand', 'model', 'damage_type','state','usage_level','service_history']

df = pd.get_dummies(df, columns=nominal, drop_first=True)





correlation = df.corr()['price']
positive_corr = correlation[correlation > 0].sort_values(ascending=False)
print(positive_corr)



df.info()


 top_features= correlation[correlation > 0.39].index.drop('price')
X = df[top_features]

y = df['price']



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



model = LinearRegression()
model.fit(X_train, y_train)



from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



xgb_model = XGBRegressor(n_estimators=1000,max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)



y_pred = xgb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor




stack_model = StackingRegressor(
    estimators=[
    ('xgb', xgb_model),
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=300, random_state=42))
],
    final_estimator=Ridge()
)
stack_model.fit(X_train, y_train)



y_pred = stack_model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)



import numpy as np


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)



import matplotlib.pyplot as plt
from xgboost import plot_importance

plt.figure(figsize=(12, 8))
plot_importance(xgb_model, max_num_features=20, importance_type='gain', height=0.8)
plt.title("Top 20 Important Features (XGBoost)")
plt.show()




print("Train R2:", xgb_model.score(X_train, y_train))

print("Test R2:", xgb_model.score(X_test, y_test))



import pandas as pd
test=pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_test.csv')
test


test.info()


test['damage_cost']=test['damage_cost'].fillna(0)
test['damage_type']=test['damage_type'].fillna('No Damage')


test.drop(['last_service_date'], axis=1, inplace=True)




nominal = ['brand', 'color', 'engine_config', 'transmission', 'drivetrain',
           'market_region', 'interior_material', 'brake_type', 'tire_brand',
           'model', 'damage_type', 'state', 'usage_level', 'service_history','id']


available_nominal = [col for col in nominal if col in test.columns]

test = pd.get_dummies(test, columns=available_nominal, drop_first=True)





X_submission = test[top_features]





for col in top_features:
    if col not in test.columns:
        test[col] = 0


test = test[top_features]




sample_submission = pd.read_csv("/kaggle/input/predict-supercars-prices-2025/sample_submission.csv")
sample_submission


predicted_prices =xgb_model.predict(test)




original_test = pd.read_csv('/kaggle/input/predict-supercars-prices-2025/supercars_test.csv')


ids = original_test['id']





submission = pd.DataFrame({
    'id': ids,           
    'target': predicted_prices
})

submission.to_csv('submission2.csv', index=False)







