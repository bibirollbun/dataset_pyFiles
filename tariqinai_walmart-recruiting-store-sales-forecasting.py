# importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.dates as mdates
import plotly.graph_objects as go
from scipy.stats import pointbiserialr 
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')


display(stores.head())
display(features.head())
display(train.head())
display(test.head())


print(features.shape)
print(train.shape)
print(test.shape)
print(stores.shape)


display(features.info())
display(train.info())
display(test.info())
display(stores.info())


# Merging train with store data
train_merged = train.merge(stores, on=['Store'], how='left')
display(train_merged.head())
print(train.shape)
print(train_merged.shape)
print(stores.shape)


# merging train with features
features = features.drop(columns=['IsHoliday'])
train_final = train_merged.merge(features, on=['Store', 'Date'], how='left')
display(train_final.head())


test_merged = test.merge(stores, on=['Store'], how='left')
test_final = test_merged.merge(features, on=['Store', 'Date'], how='left')


display(test_merged.head())
display(test_final.head())


def markdown_null_to_zero(train_final):
    train_final['MarkDown1'] = train_final['MarkDown1'].fillna(0)
    train_final['MarkDown2'] = train_final['MarkDown2'].fillna(0)
    train_final['MarkDown3'] = train_final['MarkDown3'].fillna(0)
    train_final['MarkDown4'] = train_final['MarkDown4'].fillna(0)
    train_final['MarkDown5'] = train_final['MarkDown5'].fillna(0)
    return train_final


train_final = markdown_null_to_zero(train_final)
display(train_final[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].info())


train_final.isnull().sum()


train_final.dtypes


train_final['Date'] = pd.to_datetime(train['Date'])
test_final['Date'] = pd.to_datetime(test['Date'])


# Group by store and sum up sales
store_sales = train_final.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)

# Plot sales per store
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=store_sales.index, y=store_sales.values, palette="viridis")  # Removed 'hue' and 'legend=False'

plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.title("Total Sales by Store")
plt.xticks(rotation=90)  # Rotate store numbers for better visibility

# Format Y-axis tick labels with exact values and commas
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Manually remove the legend if it's created automatically
if ax.get_legend() is not None:
    ax.legend_.remove()

plt.show()


# Group by Date and sum sales to get weekly total sales trend
weekly_sales_trend = train_final.groupby("Date")["Weekly_Sales"].sum()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(weekly_sales_trend.index, weekly_sales_trend.values, marker="o", linestyle="-", color="b")

# Format x-axis to show only Month-Year while keeping all weeks
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.title("Weekly Sales Trend Over Time")

# Formatting X-axis to show Month-Year while keeping all weeks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))  # Format as "Feb-2010"
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Show labels at month intervals

# Rotate labels for better readability
plt.xticks(rotation=45)

# Ensure y-axis shows full values (no scientific notation)
plt.ticklabel_format(style='plain', axis='y')

plt.grid(True)  # Add grid for better readability

plt.show()


# Group by Date and sum up sales + markdowns
weekly_trend = train_final.groupby("Date").agg({
    "Weekly_Sales": "sum",
    "MarkDown1": "sum",
    "MarkDown2": "sum",
    "MarkDown3": "sum",
    "MarkDown4": "sum",
    "MarkDown5": "sum"
}).reset_index()

# Create interactive figure
fig = go.Figure()

# Add Weekly Sales line
fig.add_trace(go.Scatter(x=weekly_trend["Date"], y=weekly_trend["Weekly_Sales"], 
                         mode="lines+markers", name="Weekly Sales", line=dict(color="blue")))

# Add MarkDown lines
fig.add_trace(go.Scatter(x=weekly_trend["Date"], y=weekly_trend["MarkDown1"], 
                         mode="lines", name="MarkDown1", line=dict(dash="dash", color="red")))
fig.add_trace(go.Scatter(x=weekly_trend["Date"], y=weekly_trend["MarkDown2"], 
                         mode="lines", name="MarkDown2", line=dict(dash="dash", color="green")))
fig.add_trace(go.Scatter(x=weekly_trend["Date"], y=weekly_trend["MarkDown3"], 
                         mode="lines", name="MarkDown3", line=dict(dash="dash", color="purple")))
fig.add_trace(go.Scatter(x=weekly_trend["Date"], y=weekly_trend["MarkDown4"], 
                         mode="lines", name="MarkDown4", line=dict(dash="dash", color="orange")))
fig.add_trace(go.Scatter(x=weekly_trend["Date"], y=weekly_trend["MarkDown5"], 
                         mode="lines", name="MarkDown5", line=dict(dash="dash", color="brown")))

# Customize layout
fig.update_layout(title="Interactive Weekly Sales and Markdowns Over Time",
                  xaxis_title="Date",
                  yaxis_title="Total Sales / Markdowns",
                  xaxis=dict(tickformat="%b-%Y"),  # Show Month-Year on x-axis
                  hovermode="x unified",
                  template="plotly_white")

# Show interactive plot
fig.show()


# CPI Across Stores
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_final, x="Store", y="CPI", palette="coolwarm")
plt.xlabel("Store Number")
plt.ylabel("Consumer Price Index (CPI)")
plt.title("Distribution of CPI Across Stores")
plt.xticks(rotation=90)
plt.grid(axis="y")
plt.show()


# Unemployment Across Stores
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_final, x="Store", y="Unemployment", palette="magma")
plt.xlabel("Store Number")
plt.ylabel("Unemployment Rate")
plt.title("Distribution of Unemployment Across Stores")
plt.xticks(rotation=90)
plt.grid(axis="y")
plt.show()


# CPI Trend Over Time
plt.figure(figsize=(12, 6))
train_final.groupby("Date")["CPI"].mean().plot(color="blue", linestyle="-", marker="o")

plt.xlabel("Date")
plt.ylabel("Consumer Price Index (CPI)")
plt.title("CPI Trend Over Time")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 6))
train_final.groupby("Date")["Unemployment"].mean().plot(color="red", linestyle="--", marker="s")

plt.xlabel("Date")
plt.ylabel("Unemployment Rate")
plt.title("Unemployment Trend Over Time")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


store_sales = train_final.groupby("Store")["Weekly_Sales"].sum()
store_size = train_final.groupby("Store")["Size"].mean()

plt.figure(figsize=(12, 6))
plt.scatter(store_size, store_sales, alpha=0.7)
plt.title("Store Size vs Total Sales")
plt.xlabel("Store Size")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()


train_final["Year"] = train_final["Date"].dt.year
train_final["Month"] = train_final["Date"].dt.month


test_final["Year"] = test_final["Date"].dt.year
test_final["Month"] = test_final["Date"].dt.month


# Create a new column for total markdown 
train_final["Total_MarkDown"] = (train_final["MarkDown1"].fillna(0) +
                                 train_final["MarkDown2"].fillna(0) +
                                 train_final["MarkDown3"].fillna(0) +
                                 train_final["MarkDown4"].fillna(0) +
                                 train_final["MarkDown5"].fillna(0)) 

# Drop individual MarkDown columns if you don't need them
train_final.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"], inplace=True)


test_final["Total_MarkDown"] = (test_final["MarkDown1"].fillna(0) +
                                    test_final["MarkDown2"].fillna(0) +
                                    test_final["MarkDown3"].fillna(0) +
                                    test_final["MarkDown4"].fillna(0) +
                                    test_final["MarkDown5"].fillna(0))

test_final.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"], inplace=True)


numerical = train_final.select_dtypes(include=['float64', 'int64', 'int32']).columns
# Correlation matrix for numerical features and target
correlation_with_target = train_final[numerical].corr()[["Weekly_Sales"]].sort_values(by="Weekly_Sales", ascending=False)

# Plot correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_with_target, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation with Weekly Sales")
plt.show()


# Compute correlation matrix
collinearity_matrix = train_final[numerical].corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(collinearity_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Collinearity Heatmap for Numerical Features")
plt.show()


# Point-Biserial Correlation
correlation, p_value = pointbiserialr(train_final["IsHoliday"], train_final["Weekly_Sales"])
print(f"Point-Biserial Correlation: {correlation:.2f}, p-value: {p_value:.4f}")


# Encode Type column (e.g., one-hot encoding or label encoding)
type_encoded = pd.get_dummies(train_final["Type"], drop_first=True)

# Calculate mutual information
mi = mutual_info_regression(type_encoded, train_final["Weekly_Sales"])
print(f"Mutual Information for Type: {mi}")


# Contingency table
contingency_table = pd.crosstab(train_final["Type"], train_final["IsHoliday"])
print("Contingency Table:")
display(contingency_table)


# CramÃ©r's V Calculation Function
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]  # Chi-squared statistic
    n = confusion_matrix.to_numpy().sum()  # Convert to NumPy and sum total observations
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

# Compute CramÃ©r's V
cramers_v_value = cramers_v(contingency_table)

# Convert to float and print correctly
print(f"CramÃ©r's V: {float(cramers_v_value):.2f}")


# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2:.2f}, p-value: {p:.4f}")


# Select numerical columns for boxplots
numerical = train_final.select_dtypes(include=['float64', 'int64', 'int32']).columns

# Set up dynamic subplots based on the number of numerical features
num_features = len(numerical)
rows = (num_features // 3) + (num_features % 3 > 0)  # Adjust rows based on feature count

plt.figure(figsize=(12, 4 * rows))  # Adjust height dynamically

for i, col in enumerate(numerical, 1):
    plt.subplot(rows, 3, i)  # Dynamically set row count
    sns.boxplot(y=train_final[col])
    plt.title(col)

plt.tight_layout()
plt.show()


# Plot distributions
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical, 1):
    plt.subplot(4, 3, i)
    sns.histplot(train_final[col], kde=True)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


# Apply sine and cosine transformation for Month
train_final["Month_sin"] = np.sin(2 * np.pi * train_final["Month"] / 12)
train_final["Month_cos"] = np.cos(2 * np.pi * train_final["Month"] / 12)

# Drop the original Month column after transformation
train_final.drop(columns=["Month"], inplace=True)


test_final["Month_sin"] = np.sin(2 * np.pi * test_final["Month"] / 12)
test_final["Month_cos"] = np.cos(2 * np.pi * test_final["Month"] / 12)

# Drop the original Month column after transformation
test_final.drop(columns=["Month"], inplace=True)


# Convert Year to Days Since Start
train_final["Days_Since_Start"] = (train_final["Date"] - train_final["Date"].min()).dt.days
test_final["Days_Since_Start"] = (test_final["Date"] - test_final["Date"].min()).dt.days

# Drop the original Year column
train_final.drop(columns=["Year"], inplace=True)
test_final.drop(columns=["Year"], inplace=True)


# Initialize scalers
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# Features to scale
columns_minmax = ["Size", "Fuel_Price"]  # MinMaxScaler for small-range features
columns_standard = ["Temperature", "CPI"]  # StandardScaler for normally distributed features
columns_robust = ["Unemployment", "Total_MarkDown"]  # RobustScaler for outlier-heavy features

# Apply scalers
train_final[columns_minmax] = minmax_scaler.fit_transform(train_final[columns_minmax])
train_final[columns_standard] = standard_scaler.fit_transform(train_final[columns_standard])
train_final[columns_robust] = robust_scaler.fit_transform(train_final[columns_robust])

# Verify the transformations
train_final.head()


test_final[columns_minmax] = minmax_scaler.transform(test_final[columns_minmax])
test_final[columns_standard] = standard_scaler.transform(test_final[columns_standard])
test_final[columns_robust] = robust_scaler.transform(test_final[columns_robust])
test_final.head()


train_final.drop(columns=["Date"], inplace=True)
test_final.drop(columns=["Date"], inplace=True)


train_final = pd.get_dummies(train_final, columns=["Type"], drop_first=True)
test_final = pd.get_dummies(test_final, columns=["Type"], drop_first=True)

# Verify Encoding
display(train_final.head())


train_final["IsHoliday"] = train_final["IsHoliday"].astype(int)


# Ensure all negative sales are set to zero
train_final["Weekly_Sales"] = train_final["Weekly_Sales"].apply(lambda x: max(x, 0))


# Define Features (X) and Target (y)
X = train_final.drop(columns=["Weekly_Sales"])  # Drop target variable
y = train_final["Weekly_Sales"]  # Target variable

# Set a fixed random seed for reproducibility
RANDOM_SEED = 42  

# Split the dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# Print Shapes to Verify
print(f"Train Set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test Set: X_test={X_test.shape}, y_test={y_test.shape}")


# Define a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)  # Train Model
    y_pred = model.predict(X_test)  # Make Predictions
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {"MAE": mae, "MSE": mse, "R2": r2}

# Dictionary to store results
linear_model_results = {}

# 1ï¸�âƒ£ Simple Linear Regression (Baseline)
linear_reg = LinearRegression()
linear_model_results["Linear Regression"] = evaluate_model(linear_reg, X_train, X_test, y_train, y_test)

# 2ï¸�âƒ£ Ridge Regression (L2 Regularization)
ridge_reg = Ridge(alpha=1.0)  # Alpha controls regularization strength
linear_model_results["Ridge Regression"] = evaluate_model(ridge_reg, X_train, X_test, y_train, y_test)

# 3ï¸�âƒ£ Lasso Regression (L1 Regularization)
lasso_reg = Lasso(alpha=0.01)  # Alpha controls regularization strength
linear_model_results["Lasso Regression"] = evaluate_model(lasso_reg, X_train, X_test, y_train, y_test)

# 4ï¸�âƒ£ ElasticNet Regression (L1 + L2 Regularization)
elasticnet_reg = ElasticNet(alpha=0.01, l1_ratio=0.5)  # l1_ratio balances L1 and L2
linear_model_results["ElasticNet Regression"] = evaluate_model(elasticnet_reg, X_train, X_test, y_train, y_test)

# Display Results
linear_results_df = pd.DataFrame(linear_model_results).T
display(linear_results_df)


# Define models to evaluate
models = {
    "Linear Regression": linear_reg,
    "Ridge Regression": ridge_reg,
    "Lasso Regression": lasso_reg,
    "ElasticNet Regression": elasticnet_reg
}

# Sample random points for faster plotting
sample_size = min(5000000, len(y_test))  
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sampled = y_test.iloc[sample_indices]

# Create subplots (2 rows, 4 columns: Residual + Prediction Error for each model)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

# Iterate through models and generate plots
for i, (name, model) in enumerate(models.items()):
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_sampled = y_pred[sample_indices]

    # 1ï¸�âƒ£ Residual Plot
    sns.residplot(x=y_pred_sampled, y=y_test_sampled, line_kws={"color": "red"}, ax=axes[0, i])
    axes[0, i].set_title(f"Residual Plot: {name}")
    axes[0, i].set_xlabel("Predicted Sales")
    axes[0, i].set_ylabel("Residuals")

    # 2ï¸�âƒ£ Prediction Error Curve
    axes[1, i].scatter(y_test_sampled, y_pred_sampled, alpha=0.5, color="blue", label="Predictions")
    axes[1, i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--r", label="Perfect Fit")  
    axes[1, i].set_title(f"Prediction Error Curve: {name}")
    axes[1, i].set_xlabel("Actual Sales")
    axes[1, i].set_ylabel("Predicted Sales")
    axes[1, i].legend()

plt.tight_layout()
plt.show()


# Train KNN with best parameters
best_knn = KNeighborsRegressor(n_neighbors=10, metric='manhattan')
best_knn.fit(X_train, y_train)

# Make predictions
y_pred_best_knn = best_knn.predict(X_test)

# Evaluate Performance
knn_best_results = {
    "MAE": mean_absolute_error(y_test, y_pred_best_knn),
    "MSE": mean_squared_error(y_test, y_pred_best_knn),
    # "RMSE": root_mean_squared_error(y_test, y_pred_best_knn),
    "R2": r2_score(y_test, y_pred_best_knn)
}
knn_best_results_df = pd.DataFrame(knn_best_results, index=["KNN (K=10, Manhattan)"])
display(knn_best_results_df)


# Sample 5000 random points for faster visualization
sample_size = min(500000, len(y_test))  
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sampled = y_test.iloc[sample_indices]
y_pred_knn_sampled = y_pred_best_knn[sample_indices]

# 1ï¸�âƒ£ Residual Plot for KNN
plt.figure(figsize=(10,5))
sns.residplot(x=y_pred_knn_sampled, y=y_test_sampled, line_kws={"color": "red"})
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals (Errors)")
plt.title("Residual Plot (KNN: K=10, Manhattan)")
plt.show()

# 2ï¸�âƒ£ Prediction Error Curve for KNN
plt.figure(figsize=(10,5))
plt.scatter(y_test_sampled, y_pred_knn_sampled, alpha=0.5, color="blue", label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "--r", label="Perfect Fit")  # Diagonal line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Prediction Error Curve (KNN: K=10, Manhattan)")
plt.legend()
plt.show()


# Train a baseline Decision Tree
dt_regressor = DecisionTreeRegressor(random_state=42, max_depth=None, min_samples_leaf=5, min_samples_split=2)
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_regressor.predict(X_test)

# Evaluate performance
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print results
print(f"Decision Tree (Baseline):\nMAE: {mae_dt}\nMSE: {mse_dt}\nRMSE: {rmse_dt}\nRÂ²: {r2_dt}")


# Residual Plot
residuals_dt = y_test - y_pred_dt
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred_dt, y=residuals_dt)
plt.axhline(y=0, color='black', linestyle='dashed')
plt.title("Residual Plot: Decision Tree Regression")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.show()

# Prediction Error Curve 
plt.figure(figsize=(8,5)) 
sns.scatterplot(x=y_test, y=y_pred_dt, label="Predictions") 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit") 
plt.xlabel("Actual Sales") 
plt.ylabel("Predicted Sales") 
plt.legend() 
plt.title("Prediction Error Curve: Decision Tree Regression") 
plt.show() 


# Train a baseline Random Forest model
rf_regressor = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, max_depth=20, max_features=None)
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate performance 
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print results
print(f"Random Forest (Baseline):\nMAE: {mae_rf}\nMSE: {mse_rf}\nRMSE: {rmse_rf}\nRÂ²: {r2_rf}")


# Residual Plot
residuals_rf = y_test - y_pred_rf
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred_rf, y=residuals_rf)
plt.axhline(y=0, color='black', linestyle='dashed')
plt.title("Residual Plot: Random Forest Regression")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.show()


# Prediction Error Curve
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred_rf, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.legend()
plt.title("Prediction Error Curve: Random Forest Regression")
plt.show()


# Train a baseline XGBoost model
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_regressor.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_regressor.predict(X_test)

# Evaluate performance
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print results
print(f"XGBoost (Baseline):\nMAE: {mae_xgb}\nMSE: {mse_xgb}\nRMSE: {rmse_xgb}\nRÂ²: {r2_xgb}")


# Define hyperparameter grid
param_grid_xgb = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 6, 10],
    "subsample": [0.6, 0.8, 1.0]
}

# Perform Randomized Search
random_search_xgb = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1), 
    param_distributions=param_grid_xgb, 
    n_iter=10, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)

random_search_xgb.fit(X_train, y_train)

# Best parameters
best_params_xgb = random_search_xgb.best_params_
print("Best Parameters for XGBoost:", best_params_xgb)

# Train best model
best_xgb = xgb.XGBRegressor(**best_params_xgb, objective='reg:squarederror', random_state=42, n_jobs=-1)
best_xgb.fit(X_train, y_train)

# Predictions & Evaluation
y_pred_best_xgb = best_xgb.predict(X_test)
mae_best_xgb = mean_absolute_error(y_test, y_pred_best_xgb)
mse_best_xgb = mean_squared_error(y_test, y_pred_best_xgb)
rmse_best_xgb = np.sqrt(mse_best_xgb)
r2_best_xgb = r2_score(y_test, y_pred_best_xgb)

# Print best results
print(f"XGBoost (Tuned):\nMAE: {mae_best_xgb}\nMSE: {mse_best_xgb}\nRMSE: {rmse_best_xgb}\nRÂ²: {r2_best_xgb}")


# Residual Plot
residuals_xgb = y_test - y_pred_best_xgb
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_pred_best_xgb, y=residuals_xgb)
plt.axhline(y=0, color='black', linestyle='dashed')
plt.title("Residual Plot: XGBoost Regression")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.show()


# Prediction Error Curve 
plt.figure(figsize=(8,5)) 
sns.scatterplot(x=y_test, y=y_pred_best_xgb, label="Predictions") 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Fit") 
plt.xlabel("Actual Sales") 
plt.ylabel("Predicted Sales") 
plt.legend() 
plt.title("Prediction Error Curve: XGBoost Regression") 
plt.show() 


# Train a Bagging Regressor (with Decision Trees as base models)
bagging_regressor = BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1)
bagging_regressor.fit(X_train, y_train)

# Make predictions
y_pred_bagging = bagging_regressor.predict(X_test)

# Evaluate Performance
mae_bagging = mean_absolute_error(y_test, y_pred_bagging)
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
rmse_bagging = np.sqrt(mse_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# Print Results
print(f"Bagging Regressor:\nMAE: {mae_bagging}\nMSE: {mse_bagging}\nRMSE: {rmse_bagging}\nRÂ²: {r2_bagging}")


# Train an AdaBoost Regressor
adaboost_regressor = AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
adaboost_regressor.fit(X_train, y_train)

# Make predictions
y_pred_adaboost = adaboost_regressor.predict(X_test)

# Evaluate Performance
mae_adaboost = mean_absolute_error(y_test, y_pred_adaboost)
mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)
rmse_adaboost = np.sqrt(mse_adaboost)
r2_adaboost = r2_score(y_test, y_pred_adaboost)

# Print Results
print(f"AdaBoost Regressor:\nMAE: {mae_adaboost}\nMSE: {mse_adaboost}\nRMSE: {rmse_adaboost}\nRÂ²: {r2_adaboost}")


# Train a Gradient Boosting Regressor
gbm_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_regressor.fit(X_train, y_train)

# Make predictions
y_pred_gbm = gbm_regressor.predict(X_test)

# Evaluate Performance
mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
mse_gbm = mean_squared_error(y_test, y_pred_gbm)
rmse_gbm = np.sqrt(mse_gbm)
r2_gbm = r2_score(y_test, y_pred_gbm)

# Print Results
print(f"Gradient Boosting Regressor:\nMAE: {mae_gbm}\nMSE: {mse_gbm}\nRMSE: {rmse_gbm}\nRÂ²: {r2_gbm}")


# Train a LightGBM Regressor
lgb_regressor = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=-1, random_state=42)
lgb_regressor.fit(X_train, y_train)

# Make predictions
y_pred_lgb = lgb_regressor.predict(X_test)

# Evaluate Performance
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mse_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)

# Print Results
print(f"LightGBM Regressor:\nMAE: {mae_lgb}\nMSE: {mse_lgb}\nRMSE: {rmse_lgb}\nRÂ²: {r2_lgb}")

