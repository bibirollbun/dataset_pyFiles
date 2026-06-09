import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# Data lodaing
raw_data = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
raw_data.head()


raw_data.info()


raw_data['date'] = pd.to_datetime(raw_data['date'])


raw_data.isna().sum()


raw_data.dropna(inplace=True)


raw_data.isna().sum()


raw_data.head()


sales_cn = raw_data[['country','num_sold']].groupby(['country']).sum().reset_index()
sales_cn


# Create the bar plot using df.plot()
ax = sales_cn.plot(kind='bar', x='country', y='num_sold', legend=False)

# Add values on top of each bar
for i, value in enumerate(sales_cn['num_sold']):
    ax.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=10)

# Customize and show the plot
plt.title('Bar Plot with Values')
plt.ylabel('num_sold')
plt.tight_layout()
plt.show()


salesnum=raw_data['store'].value_counts(normalize=True)
plt.figure(figsize=(5,6))
salesnum.plot(kind='pie',y=salesnum.values, autopct='%1.1f%%')
plt.ylabel('')
plt.show()


sales_pro = raw_data[['product','num_sold']].groupby(['product']).sum().reset_index()
sales_pro


ax = sales_pro.plot(kind='bar', x='product', y='num_sold', legend=False)

# Add values on top of each bar
for i, value in enumerate(sales_pro['num_sold']):
    ax.text(i, value + 1, str(value), ha='center', va='bottom', fontsize=10)

# Customize and show the plot
plt.title('Bar Plot with Values')
plt.ylabel('num_sold')
plt.tight_layout()
plt.show()


sns.kdeplot(raw_data['num_sold'])
plt.show()


raw_data.head()


raw_data.drop('id',axis=1,inplace=True)


raw_data.nunique()


raw_data.head()


encoder =LabelEncoder()
categorical_columns = ['country', 'store', 'product']


for col in categorical_columns:
    raw_data[col] = encoder.fit_transform(raw_data[col])

# 2. Handle the Date column by extracting date features (Year, Month, Day, Weekday)
raw_data['Year'] = raw_data['date'].dt.year
raw_data['Month'] = raw_data['date'].dt.month
raw_data['Day'] = raw_data['date'].dt.day
raw_data['Weekday'] = raw_data['date'].dt.weekday

# Drop the original 'Date' column after extracting features
raw_data.drop('date', axis=1, inplace=True)
raw_data.head()


from sklearn.preprocessing import StandardScaler

# Scale the features and target
scaler = StandardScaler()

X = raw_data.drop('num_sold', axis=1)
y = raw_data['num_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize and train the model
rf = RandomForestRegressor(max_depth=4, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)
# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Random Forest RMSE: {rmse}")


from sklearn.ensemble import GradientBoostingRegressor

# Initialize and train the model
gb = GradientBoostingRegressor(learning_rate= 0.2, max_depth= 4, n_estimators= 400, random_state=42)
gb.fit(X_train, y_train)

# Predict on the test set
y_pred = gb.predict(X_test)
# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Gradient Boosting Regressor RMSE: {rmse}")


from xgboost import XGBRegressor

# 5. Train XGBoost Regressor Model
xgb_regressor = XGBRegressor(learning_rate=0.2, max_depth=4, n_estimators=400,random_state=42)
xgb_regressor.fit(X_train, y_train)

# 6. Make Predictions and Evaluate
y_pred = xgb_regressor.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"XGB Regressor RMSE: {rmse}")


from sklearn.model_selection import train_test_split, GridSearchCV

# 5. Define Models
models = {
    'XGBoost': XGBRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Hyperparameter Grids
param_grids = {
    'XGBoost': {
        'n_estimators': [100, 200,300,400],
        'max_depth': [2, 4],
        'learning_rate': [0.1, 0.2]
    },
    'Random Forest': {
        'n_estimators': [100, 200,300,400],
        'max_depth': [2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200,300,400],
        'max_depth': [2, 4],
        'learning_rate': [0.1, 0.2]
    }
}

# 6. Perform GridSearchCV and Evaluate Models
best_models = {}
for name, model in models.items():
    print(f"Tuning hyperparameters for {name}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best Parameters for {name}: {grid_search.best_params_}")

# 7. Compare Model Performance
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Calculate RMSE
    print(f"{name} RMSE: {rmse:.2f}")


test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df = test.copy()
df['date'] = pd.to_datetime(df['date'])

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# 2. Handle the Date column by extracting date features (Year, Month, Day, Weekday)
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Day'] = df['date'].dt.day
df['Weekday'] = df['date'].dt.weekday

# Drop the original 'Date' column after extracting features
df.drop('date', axis=1, inplace=True)
df.head()


df.drop('id',axis=1,inplace=True)
df = scaler.transform(df)


# Gradient Boosting Regressor give good results
y_pred = gb.predict(df)


sub = pd.DataFrame({'id':test['id'],'num_sold':y_pred})
sub.to_csv('submission.csv',index=False)
sub.round(2)

