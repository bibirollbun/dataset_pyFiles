import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from xgboost import XGBRegressor




# Preprocessing: Extract year and month from 'year-month'
df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/wildfire_sizes_before_2010.csv")
def join_data(df):
    other_df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/weather_monthly_state_aggregates.csv")
    third_df = pd.read_csv("/kaggle/input/forest-fire-prediction-epoch-hackathon/merged_state_data.csv")
    df = pd.merge(df, other_df, left_on=['STATE', 'month'], right_on=['State', 'year_month'], how='left')
    df = pd.merge(df, third_df, left_on=['State'], right_on=['State'], how='left')
    df[['year', 'month']] = df['month'].str.split('-', expand=True)
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['Percentage of Federal Land'] = df['Percentage of Federal Land'].str.rstrip('%').astype(float) / 100

    # replacement_values = {
    # 'PRCP': 100,     

    # }
    # df.fillna(value=replacement_values, inplace=True)

    return df
df = join_data(df)


print(df.head())
list_parameters = ['mean_elevation', 'Land Area (sq mi)', 'PRCP', 'EVAP', 'Percentage of Federal Land']
X = df[['year', 'month']]  # Features: year and month
y = np.log(df['total_fire_size'])  # Target: total_fire_size
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_others = df[list_parameters]
X_poly = np.hstack((X_poly, X_others))

#{'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 500}
model = XGBRegressor(objective='reg:squarederror', n_estimators=400, learning_rate=0.1, random_state=42)
model.fit(X_poly, y)
print('done')
    


ans_df = pd.read_csv('/kaggle/input/forest-fire-prediction-epoch-hackathon/zero_submission.csv')
ans_df.head()
ans_df = join_data(ans_df)

X = ans_df[['year', 'month']]  # Features: year and month
X_poly = poly.fit_transform(X)
X_others = ans_df[list_parameters]
X_poly = np.hstack((X_poly, X_others))

predictions = np.exp(model.predict(X_poly))

# Step 4: Fill the 'total_fire_size' column with the predictions
ans_df['total_fire_size'] = predictions

ans_df['year-month'] = ans_df['year'].astype(str) + '-' + ans_df['month'].astype(str).str.zfill(2)

ans_df = ans_df[['ID', 'STATE', 'year-month', 'total_fire_size']]
ans_df.rename(columns={'year-month': 'month'}, inplace=True)

ans_df.to_csv('submission.csv', index=False)
ans_df.head(n = 30)







