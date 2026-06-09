# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse

from sklearn.linear_model import LinearRegression

import random
random.seed(42)




#load data

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

train.columns



# check NA and clean the data
print(f"Nulls in train_data out of {train.shape[0]} rows")
for col in train.columns:
    print(f'{col}: {train[col].isna().sum()}')
    
print(f"\n\nNulls in test_data out of {test.shape[0]} rows")
for col in test.columns:
    print(f'{col}: {test[col].isna().sum()}')

train.dropna(inplace=True)





print(f"Unique values in train_data out of {train.shape[0]} rows")
for col in train.columns:
    print(f'{col}: {train[col].nunique()}')
    
print(f"\n\n Unique values in test_data out of {test.shape[0]} rows")
for col in test.columns:
    print(f'{col}: {test[col].nunique()}')

# print(f'\n\n')
# for col in train.columns:
#     print(f'{col}: {train[col].unique()}')


plt.hist(train['num_sold'], bins=20)
plt.show()




# print(test.year.unique(), train.year.unique())
print(train.country.unique(), test.country.unique(), train.store.unique(), test.store.unique(), train['product'].unique(), test['product'].unique())


import requests

# Country to Alpha-3 code mapping
alpha3 = {'Canada': 'CAN', 'Finland': 'FIN', 'Italy': 'ITA', 'Kenya': 'KEN', 'Norway': 'NOR', 'Singapore': 'SGP'}

# Fetch GDP per capita
def fetch_gdp_data(countries, years):
    gdp_data, session = {}, requests.Session()
    for country in countries:
        code = alpha3.get(country)
        for year in years:
            url = f"https://api.worldbank.org/v2/country/{code}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
            try:
                data = session.get(url).json()
                gdp_data[(country, year)] = data[1][0]['value'] if len(data) > 1 and data[1] else None
            except:
                gdp_data[(country, year)] = None
    return gdp_data

# Fetch data once
gdp_data = fetch_gdp_data(alpha3.keys(), range(2010, 2020))

# Add GDP feature
def add_gdp_feature(df):
    df['gdp'] = df.assign(year=pd.to_datetime(df['date']).dt.year).set_index(['country', 'year']).index.map(gdp_data.get)
    return df



# create features
train_df = train.copy(deep=True)
test_df = test.copy(deep =True)
for df in [train_df , test_df]:
    df = add_gdp_feature(df)

    df['date'] = pd.to_datetime(df['date'])
    df['year']= df['date'].dt.year
    df['month']= df['date'].dt.month
    df['day_of_year']= df['date'].dt.day_of_year
    df['day_of_week']= df['date'].dt.day_of_week
    df['day_of_month']= df['date'].dt.day
    
    df.drop('date',  axis=1, inplace=True)

    import numpy as np

    # Encode Month (12 months in a year)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Encode Day of Year (365 days in a year)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
    
    # df['day_of_year_sin4'] = np.sin(4 * np.pi * df['day_of_year'] / 366)
    # df['day_of_year_cos4'] = np.cos(4 * np.pi * df['day_of_year'] / 366)
    
    # df['day_of_year_sin6'] = np.sin(6 * np.pi * df['day_of_year'] / 366)
    # df['day_of_year_cos6'] = np.cos(6 * np.pi * df['day_of_year'] / 366)
    
    # df['day_of_year_sin8'] = np.sin(8 * np.pi * df['day_of_year'] / 366)
    # df['day_of_year_cos8'] = np.cos(8 * np.pi * df['day_of_year'] / 366)
    
    # Encode Day of Week (7 days in a week)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Encode Day of Month (31 days in a month)
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

    df['year_scaled'] = df['year']%2000

    for col in df.columns:
        print(f'{col}: {df[col].nunique()}, ')
        
dummy_columns = [ 'country', 'store', 'product', 
]
train_df = pd.get_dummies(train_df, columns=dummy_columns, drop_first=True)
test_df = pd.get_dummies(test_df, columns=dummy_columns, drop_first=True)



test_df#.columns.to_list()



target = 'num_sold'

train_df.drop('id', axis=1, inplace=True)
X = train_df.drop(target, axis=1)
y = train_df[target]



#split train and val data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_val


# create regression model

model = LinearRegression()
model.fit( X_train, y_train)
    
y_pred = model.predict(X_val)
_mape = mape(y_val, y_pred)
_mse = mse(y_val, y_pred)
print(_mape, _mse)




import lightgbm as lgb

# Initialize the LightGBM model
lgbm_model = lgb.LGBMRegressor(
    n_estimators=1000,    # Number of boosting rounds
    learning_rate=0.07,   # Step size for updating weights
    max_depth=16,         # Limits tree depth
    num_leaves=128,       # Number of leaves in each tree
    subsample=1,       # Fraction of data used for each iteration
    colsample_bytree=1, # Fraction of features used per tree
    random_state=42,
    n_jobs=-1 # Use all CPU cores for training
)

# Train the model
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
               eval_metric='mape' )

# Predictions
y_pred_lgbm = lgbm_model.predict(X_val)

# Evaluate performance
mape_score = mape(y_val, y_pred_lgbm)
mse_score = mse(y_val, y_pred_lgbm)

print(f"MAPE: {mape_score:.4f}, MSE: {mse_score:.4f}")



y_val, y_pred
y_val, y_pred_lgbm


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Calculate residuals
error = y_val.values - y_pred_lgbm

# Define bins symmetrically around zero
max_error = max(abs(error))  # Find the max absolute error
bins = np.linspace(-max_error, max_error, 40)  # Create bins centered around 0

plt.figure(figsize=(10, 6))

# Plot histogram with Seaborn for better aesthetics
sns.histplot(error, bins=bins, kde=True, color='purple', edgecolor='black', alpha=0.75)

# Vertical line at zero for reference
plt.axvline(0, color='black', linestyle='dashed', linewidth=2, alpha=0.8, label="Zero Error")

# Labels and title
plt.xlabel("Prediction Error (Actual - Predicted)", fontsize=14, fontweight='bold')
plt.ylabel("Frequency", fontsize=14, fontweight='bold')
plt.title("Distribution of Prediction Errors", fontsize=16, fontweight='bold')

# Grid and legend
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()

# Customize x-axis labels
ticks = np.linspace(-max_error, max_error, 9)  # More frequent ticks
plt.xticks(ticks)

plt.show()



import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Calculate residuals
error = y_val.values - y_pred

# Define bins symmetrically around zero
max_error = max(abs(error))  # Find the max absolute error
bins = np.linspace(-max_error, max_error, 40)  # Create bins centered around 0

plt.figure(figsize=(10, 6))

# Plot histogram with Seaborn for better aesthetics
sns.histplot(error, bins=bins, kde=True, color='purple', edgecolor='black', alpha=0.75)

# Vertical line at zero for reference
plt.axvline(0, color='black', linestyle='dashed', linewidth=2, alpha=0.8, label="Zero Error")

# Labels and title
plt.xlabel("Prediction Error (Actual - Predicted)", fontsize=14, fontweight='bold')
plt.ylabel("Frequency", fontsize=14, fontweight='bold')
plt.title("Distribution of Prediction Errors", fontsize=16, fontweight='bold')

# Grid and legend
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()

# Customize x-axis labels
ticks = np.linspace(-max_error, max_error, 9)  # More frequent ticks
plt.xticks(ticks)

plt.show()



# Get the coefficients and feature names
coefficients = model.coef_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Abs_Coefficient': abs(coefficients)  # Absolute value of coefficients
})

# Sort features by absolute coefficient value to see feature importance
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

# Display the feature importance
feature_importance




import xgboost as xgb

# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=4000,       # Number of boosting rounds
    learning_rate=0.05,      # Step size for weight updates
    max_depth=10,           # Maximum tree depth
    min_child_weight = 2, 
    colsample_bytree= .9,    # Fraction of features used per tree
    subsample= .9,           # Fraction of data used per iteration
    objective='reg:squarederror',  # Loss function for regression
    eval_metric='mape',      # Root Mean Squared Error as evaluation metric
    early_stopping_rounds=100, # Stop if validation error doesn't improve
    random_state=42, 
    reg_alpha = 3, reg_lambda = 4
)

# Train the model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# Predictions
y_pred_xgb = xgb_model.predict(X_val)

# Evaluate performance
mape_score = mape(y_val, y_pred_xgb)
mse_score = mse(y_val, y_pred_xgb)

print(f"MAPE: {mape_score:.4f}, MSE: {mse_score:.4f}")






sub = pd.DataFrame()
fin_test = test_df.copy(deep=True)
sub = fin_test[['id']]

fin_test.drop('id', axis=1, inplace=True)

preds = xgb_model.predict(fin_test)


sub['num_sold'] = preds
sub


sub.to_csv('/kaggle/working/submission.csv', index=False)




