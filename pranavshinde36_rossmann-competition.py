import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import plot_tree
%matplotlib inline
warnings.filterwarnings('ignore')


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_store = pd.read_csv("store.csv")


df_train.head()


df_store.head()


# join / merge train and store df
df = pd.merge(df_train, df_store, on='Store', how='left')


df.head()


df.info()


df.shape


df.isnull().sum()


# drop columns with too many null values 
df.drop(["CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek",
             "Promo2SinceYear","PromoInterval"],axis=1, inplace=True,)


df.isnull().sum()


# Fix null values 
df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(),inplace=True)


df.isnull().sum()


# seperate date into day & month and drop date

df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.drop(columns=['Date'], inplace=True)
df.head()


# Remove Unwanted rows where sales are zero because we want to predict sales only 
df = df.drop(df[df["Sales"] == 0].index)
df.shape


# drop customers column as its unpredicted in future days
df.drop(["Customers"],axis=1, inplace=True,)
df.head()


# drop open Column as its obious if closed no sales and already removed 0 sales record from df
df.drop(["Open"],axis=1, inplace=True,)
df.head()


# Value	Meaning	Expected Impact on Sales
# 0	No Holiday	Normal business operations.
# a	Public Holiday	General holidays (e.g., Reunification Day). Shops usually closed.
# b	Easter Holiday	Easter. Significant impact on sales leading up to it.
# c	Christmas	Christmas. High sales volume before, likely closed during.
df["StateHoliday"].unique()


# Fix there two types of 0 issue
df["StateHoliday"] = df["StateHoliday"].replace(0, '0')
df["StateHoliday"].unique()


# Lambda function: if DayOfWeek is 6 or 7, return 1, else 0
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)


# Map motnth to season to check wether which season the sales boost 
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)
df.head()



# 1. Sort the data! (CRITICAL STEP)
# Ensure data is ordered by Store and Date (Ascending)
# Note: Ensure you have a 'Year' column if your data spans multiple years.
df = df.sort_values(by=['Store', 'Month', 'Day'])

# 2. Create Lag 1 (Yesterday's Sales)
# We group by 'Store' so we don't mix data between different stores.
df['Sales_Lag_1'] = df.groupby('Store')['Sales'].shift(1)

# 3. Create Lag 7 (Last Week's Sales)
df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)

# 4. (Optional) Fill NaN values
# The first row of every store will be NaN (because there is no "yesterday").
# You can drop them or fill them with 0.
df = df.dropna()


# CompetitionDistance can be noisy. Group them
df['CompetitionDistance'] = pd.cut(df['CompetitionDistance'],
                               bins=[0, 500, 2000, np.inf],
                               labels=['Near', 'Medium', 'Far'],
                               right=False,
                               include_lowest=True)



# Rearrange columns with logical grouping of features and Sales as last column
cols = [
    # Store characteristics 
    'Store', 'StoreType', 'Assortment',
    
    # Time features
    'DayOfWeek', 'Day', 'Month', 'Is_Weekend', 'Season',
    
    # Competition features
    'CompetitionDistance',
    
    # Promotion features 
    'Promo', 'Promo2',
    
    # Status indicators
    'StateHoliday', 'SchoolHoliday',
    
    # Target variable
    'Sales_Lag_1','Sales_Lag_7','Sales'
]

df = df[cols]
df.head()


import seaborn as sns
import matplotlib.pyplot as plt
# Lets see Which Season has the most sales.
# Calculate average sales by season
season_sales = df.groupby('Season')['Sales'].mean().reset_index()

# Plot
sns.barplot(data=season_sales, x='Season', y='Sales')
plt.title('Average Sales by Season')
plt.savefig('season_sales.png')


# lets see sales by day of week 
# Group by DayOfWeek (1=Mon, 7=Sun)
day_sales = df.groupby('DayOfWeek')['Sales'].mean().reset_index()

sns.barplot(data=day_sales, x='DayOfWeek', y='Sales')
plt.title('Average Sales by Day of Week')
plt.savefig('day_sales.png')


# Select only numeric columns
cols = ['Sales', 'Promo', 'SchoolHoliday', 'DayOfWeek']
corr = df[cols].corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap='RdBu')
plt.title('Correlation Matrix')
plt.savefig('correlation.png')


df.head()


# One-Hot Encoding 
# For nominal variables with few categories. 
# drop_first=True avoids multicollinearity (dummy variable trap).
nominal_cols = ['StoreType', 'Assortment', 'StateHoliday', 'Season']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)


# Target Encoding 
# For high cardinality 'Store' column.
# WARNING: In a real ML pipeline, compute this mean on X_train ONLY and map to X_test 
# to avoid data leakage.
store_means = df.groupby('Store')['Sales'].mean()
df['Store_encoded'] = df['Store']
df.drop(['Store'], axis=1, inplace=True)


# Odinal Encoding
# 1. Define the mapping
comp_dist_map = {'Near': 0, 'Medium': 1, 'Far': 2}

# 2. Create the new encoded column by mapping the values
df['CompetitionDistance_encoded'] = df['CompetitionDistance'].map(comp_dist_map)

# 3. Drop the original column
df.drop(['CompetitionDistance'], axis=1, inplace=True)


df.head()
df.shape


# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# separate dataset into train and test
X = df.drop(['Sales'], axis=1)
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
X_train.shape, X_test.shape


X.head()


y.head()


##Create a Function to Evaluate Model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# If the data is truly numerical (distances), convert to float/int
X_train['CompetitionDistance_encoded'] = X_train['CompetitionDistance_encoded'].astype(float)
X_test['CompetitionDistance_encoded'] = X_test['CompetitionDistance_encoded'].astype(float)

XGB = XGBRegressor(random_state=42)
XGB.fit(X_train, y_train)
# Perform Prediction
y_pred = XGB.predict(X_test)
# Evaluate Model
mae, rmse, r2_square = evaluate_model(y_test, y_pred)
print(f'MAE: {mae}\nRMSE: {rmse}\nR2 Square: {r2_square}')


GB = GradientBoostingRegressor(random_state=42)
GB.fit(X_train, y_train)
y_pred = GB.predict(X_test)
# Evaluate Model
mae, rmse, r2_square = evaluate_model(y_test, y_pred)
print(f'MAE: {mae}\nRMSE: {rmse}\nR2 Square: {r2_square}')


from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import time

# 1. OPTIMIZATION: Use 'hist' tree method (much faster for large data) 
# and set n_jobs=1 here to avoid fighting with RandomizedSearchCV for CPU cores.
xgb_model = XGBRegressor(
    n_jobs=1,              # Let RandomizedSearchCV handle parallelization
    tree_method='hist',    # <--- HUGE SPEEDUP: Uses histogram binning
    random_state=42
)

xgboost_params = {
    'n_estimators': [100, 300, 500],       # Reduced upper limit (1000 is often overkill for tuning)
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2]
}

# 2. OPTIMIZATION: Train on a subset (e.g., 20% of data)
# Hyperparameters found on a representative sample usually transfer well to the full dataset.
sample_size = int(len(X_train) * 0.2) # Use 20% of data
X_sample = X_train.sample(sample_size, random_state=42)
y_sample = y_train.loc[X_sample.index]

print(f"Tuning on {sample_size} rows (20% sample) for speed...")

start_time = time.time()

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgboost_params,
    n_iter=50,             # <--- 50 is usually sufficient to find good params
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1,             # <--- Parallelize the SEARCH, not the model
    random_state=42
)

random_search.fit(X_sample, y_sample)

print(f"Time taken: {round((time.time() - start_time)/60, 2)} minutes")
print(f"Best Params: {random_search.best_params_}")

# 3. FINAL STEP: Train the best model on the FULL dataset
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)


## Retraining the models with best parameters
models = {
    "Xgboost Regressor":XGBRegressor(subsample=0.8, reg_lambda= 1, reg_alpha= 0.1, n_estimators= 500,
                                      min_child_weight= 5, max_depth= 9, learning_rate= 0.1, 
                                     gamma= 0.3, colsample_bytree= 0.8,n_jobs=-1) 
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    
    print('='*35)
    print('\n')

