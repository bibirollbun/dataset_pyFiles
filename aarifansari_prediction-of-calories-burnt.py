import numpy as np
import pandas as pd
import os

# Model
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, probplot
from sklearn.decomposition import PCA
# import dask.dataframe as dd

from sklearn.metrics import mean_squared_log_error

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


df.info()


df.describe(include='all')


df = df.drop(columns=['id'])
df.head()


df.columns.tolist()


plt.style.use('seaborn')
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Duration', y = 'Calories', hue = 'Sex',style='Sex',s = 100)
plt.title('Calories vs Duration by Sex')
plt.xlabel('Duration (minutes)')
plt.ylabel('Calories')
plt.show()


avg_calories = df.groupby('Sex')['Calories'].mean().reset_index()
plt.figure(figsize=(6, 5))
sns.barplot(data=avg_calories, x='Sex', y='Calories', hue='Sex', palette='muted')
plt.title('Average Calories by Sex')
plt.xlabel('Sex')
plt.ylabel('Average Calories')
plt.show()


avg_calories = df.groupby('Sex')['Calories'].mean().reset_index()
plt.figure(figsize=(6, 5))
sns.barplot(data=avg_calories, x='Sex', y='Calories', hue='Sex', palette='muted')
plt.title('Average Calories by Sex')
plt.xlabel('Sex')
plt.ylabel('Average Calories')
plt.show()


plt.figure(figsize=(8, 6))
corr_matrix = df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


corr_matrix


df['Sex'] = df['Sex'].map({'male':1,'female':0})
X = df[['Sex','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']]
y = df['Calories']
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Linearity Check: Features vs Calories')
for i, col in enumerate(['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']):
    ax = axes[i//3, i%3]
    sns.scatterplot(data=df, x=col, y='Calories', hue='Sex', ax=ax)
    ax.set_title(f'Calories vs {col}')
plt.tight_layout()
plt.show()



# --- Linear Regression ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
y_pred_lr = np.maximum(0, y_pred_lr)
# Linear Regression Performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmsle_lr = np.sqrt(mean_squared_log_error(y_test, y_pred_lr))
print("\nLinear Regression Performance (Test Set):")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"R-squared: {r2_lr:.2f}")
print(f"RMSLE: {rmsle_lr:.2f}")
# Coefficients
coef_df_lr = pd.DataFrame(model_lr.coef_, X.columns, columns=['Coefficient'])
print("\nLinear Regression Coefficients:")
print(coef_df_lr)


# Check Assumption 3: Homoscedasticity (Residuals vs Predicted)
residuals = y_test - y_pred_lr
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_lr, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Homoscedasticity Check)')
plt.xlabel('Predicted Calories')
plt.ylabel('Residuals')
plt.show()


# Check Assumption 4: Normality (Q-Q Plot)
from scipy.stats import probplot
plt.figure(figsize=(8, 6))
probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Normality Check)')
plt.show()


# Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals (Normality Check)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# Shapiro-Wilk Test for Normality (with caution due to small sample)
shapiro_stat, shapiro_p = shapiro(residuals)
print("\nShapiro-Wilk Test for Normality:")
print(f"Statistic: {shapiro_stat:.2f}, p-value: {shapiro_p:.2f}")


# Check Assumption 5: Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)


# --- Ridge Regression ---
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)
y_pred_ridge = np.maximum(0.0,y_pred_ridge)
# Ridge Performance
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
rmsle_ridge = np.sqrt(mean_squared_log_error(y_test, y_pred_ridge))
print("\nRidge Regression Performance (Test Set):")
print(f"Mean Squared Error: {mse_ridge:.2f}")
print(f"R-squared: {r2_ridge:.2f}")
print(f"RMSLE: {rmsle_ridge}")

# Coefficients
coef_df_ridge = pd.DataFrame(model_ridge.coef_, X.columns, columns=['Coefficient'])
print("\nRidge Regression Coefficients:")
print(coef_df_ridge)


# --- PCA Model ---
X_correlated = X_train[['Duration', 'Heart_Rate', 'Body_Temp']]
pca = PCA(n_components=1)
X_pca_train = pca.fit_transform(X_correlated)
X_new_train = pd.concat([X_train[['Sex', 'Age','Height', 'Weight']].reset_index(drop=True),
                         pd.DataFrame(X_pca_train, columns=['PCA1'])], axis=1)

# Transform test set
X_correlated_test = X_test[['Duration', 'Heart_Rate', 'Body_Temp']]
X_pca_test = pca.transform(X_correlated_test)
X_new_test = pd.concat([X_test[['Sex', 'Age','Height', 'Weight']].reset_index(drop=True),
                        pd.DataFrame(X_pca_test, columns=['PCA1'])], axis=1)


# Fit PCA Model
model_pca = Ridge(alpha=1.0)
model_pca.fit(X_new_train, y_train)
y_pred_pca = model_pca.predict(X_new_test)
y_pred_pca = np.maximum(0.0,y_pred_pca)
# PCA Performance
mse_pca = mean_squared_error(y_test, y_pred_pca)
r2_pca = r2_score(y_test, y_pred_pca)
rmsle_pca = np.sqrt(mean_squared_log_error(y_test,y_pred_pca))
print("\nPCA Model Performance (Test Set):")
print(f"Mean Squared Error: {mse_pca:.2f}")
print(f"R-squared: {r2_pca:.2f}")
print(f"RMSLE: {rmsle_pca}")
# Coefficients
coef_df_pca = pd.DataFrame(model_pca.coef_, X_new_train.columns, columns=['Coefficient'])
print("\nPCA Model Coefficients:")
print(coef_df_pca)


# Coefficients
coef_df_pca = pd.DataFrame(model_pca.coef_, X_new_train.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coef_df_pca)


# --- Assumption Checks (Ridge Model, Test Set) ---
residuals_ridge_pca = y_test - y_pred_pca


# Shapiro-Wilk Test for Normality (with caution due to small sample)
shapiro_stat, shapiro_p = shapiro(residuals_ridge_pca)
print("\nShapiro-Wilk Test for Normality:")
print(f"Statistic: {shapiro_stat:.2f}, p-value: {shapiro_p:.2f}")


# Homoscedasticity
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_pca, y=residuals_ridge_pca)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values (Homoscedasticity Check, Ridge)')
plt.xlabel('Predicted Calories')
plt.ylabel('Residuals')
plt.show()


# Normality: Q-Q Plot
plt.figure(figsize=(8, 6))
probplot(residuals_ridge_pca, dist="norm", plot=plt)
plt.title('Q-Q Plot of Ridge Residuals (Normality Check)')
plt.show()


# Histogram
plt.figure(figsize=(8, 6))
sns.histplot(residuals_ridge_pca, kde=True, bins=30)
plt.title('Histogram of Ridge Residuals (Normality Check)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
# plt.savefig('residuals_histogram_ridge.png')
plt.show()


# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(residuals_ridge_pca[:5000])
print("\nShapiro-Wilk Test for Normality (Ridge, Test Set, sample):")
print(f"Statistic: {shapiro_stat:.2f}, p-value: {shapiro_p:.2f}")


# Check Assumption 6: Outliers (Standardized Residuals)
std_residuals = residuals_ridge_pca / np.std(residuals_ridge_pca)
print("\nStandardized Residuals:")
print(pd.DataFrame({'Std_Residual': std_residuals}))

# Visualize Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_pca, hue=X_test['Sex'], style=X_test['Sex'], s=100)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs Predicted Calories')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()


# Multicollinearity: VIF (Training Set)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_new_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_new_train.values, i) for i in range(X_new_train.shape[1])]
print("\nVariance Inflation Factor (VIF, Training Set):")
print(vif_data)


# Outliers: Standardized Residuals
std_residuals = residuals_ridge_pca / np.std(residuals_ridge_pca)
outliers = np.abs(std_residuals) > 3
print("\nNumber of Outliers (|Std Residual| > 3, Ridge, Test Set):", sum(outliers))
print("\nStandardized Residuals (Sample):")
print(pd.DataFrame({'Std_Residual': std_residuals}).head())


# Actual vs Predicted Plot (Ridge)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_pca, hue=X_test['Sex'], style=X_test['Sex'], s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Calories (Ridge)')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()



# Initialize ElasticNet model
model_enet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

# Fit model on PCA-transformed training data
model_enet.fit(X_new_train, y_train)

# Predict on PCA-transformed test data
y_pred_enet = model_enet.predict(X_new_test)

# Clip negative predictions if your target can't be negative
y_pred_enet = np.maximum(0, y_pred_enet)

# Evaluate performance
mse_enet = mean_squared_error(y_test, y_pred_enet)
r2_enet = r2_score(y_test, y_pred_enet)
rmsle_enet = np.sqrt(mean_squared_log_error(y_test, y_pred_enet))

print("\nElastic Net Regression Performance (Test Set):")
print(f"Mean Squared Error: {mse_enet:.2f}")
print(f"R-squared: {r2_enet:.2f}")
print(f"RMSLE: {rmsle_enet:.4f}")

# Show coefficients
coef_df_enet = pd.DataFrame(model_enet.coef_, X_new_train.columns, columns=['Coefficient'])
print("\nElastic Net Regression Coefficients:")
print(coef_df_enet)


# Create polynomial features 
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(X_new_train)
X_poly_test = poly.transform(X_new_test)

# Train linear regression on the polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Predict
y_pred_poly = poly_model.predict(X_poly_test)
y_pred_poly = np.maximum(0, y_pred_poly)  # Ensure non-negative predictions

# Evaluate
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
rmsle_poly = np.sqrt(mean_squared_log_error(y_test, y_pred_poly))

print("\nPolynomial Regression Performance (Test Set):")
print(f"Mean Squared Error: {mse_poly:.2f}")
print(f"R-squared: {r2_poly:.2f}")
print(f"RMSLE: {rmsle_poly:.4f}")

# Optional: check number of features after polynomial expansion
print(f"\nNumber of features after polynomial expansion: {X_poly_train.shape[1]}")


# Visualize Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_poly, hue=X_test['Sex'], style=X_test['Sex'], s=100)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs Predicted Calories')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()


# Train Bayesian Ridge
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)

# Predict
y_pred_bayes = bayesian_model.predict(X_test)
y_pred_bayes = np.maximum(0, y_pred_bayes)

# Evaluate RMSLE
rmsle_bayes = np.sqrt(mean_squared_log_error(y_test, y_pred_bayes))
print(f"ðŸ“‰ Bayesian Linear Regression RMSLE: {rmsle_bayes:.4f}")


# Actual vs Predicted Plot (Ridge)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_bayes, hue=X_test['Sex'], style=X_test['Sex'], s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted Calories (Ridge)')
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.show()



from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Define search space
kernels = ['linear', 'rbf', 'poly']
C_values = [0.01, 0.1, 1]

# List to store model info
model_results = []

# Loop through all combinations
for kernel in kernels:
    for C in C_values:
        # Create pipeline
        svr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel=kernel, C=C))
        ])

        # Fit model
        svr_pipeline.fit(X_new_train, y_train)

        # Predict on test set
        y_pred = svr_pipeline.predict(X_new_test)
        y_pred = np.maximum(0, y_pred)

        # Calculate RMSLE
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        # Store info
        model_info = {
            'model': svr_pipeline,
            'kernel': kernel,
            'C': C,
            'rmsle': rmsle
        }
        model_results.append(model_info)

        print(f"Kernel: {kernel}, C: {C} â†’ RMSLE: {rmsle:.4f}")

# Sort models by RMSLE
model_results = sorted(model_results, key=lambda x: x['rmsle'])

# Print best model info
best = model_results[0]
print(f"\nâœ… Best Model â†’ Kernel: {best['kernel']}, C: {best['C']}, RMSLE: {best['rmsle']:.4f}")



compute_rmsle = lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))


dt_model = DecisionTreeRegressor(random_state=0)
dt_model.fit(X_new_train, y_train)
y_pred_dt = dt_model.predict(X_new_test)

print(f"ðŸ“‰ Decision Tree RMSLE: {compute_rmsle(y_test, y_pred_dt):.4f}")


from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_new_train, y_train)
y_pred_knn = knn_model.predict(X_new_test)

print(f"ðŸ“‰ KNN (k=5) RMSLE: {compute_rmsle(y_test, y_pred_knn):.4f}")


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_new_train, y_train)
y_pred_rf = rf_model.predict(X_new_test)

print(f"ðŸ“‰ Random Forest RMSLE: {compute_rmsle(y_test, y_pred_rf):.4f}")


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
gb_model.fit(X_new_train, y_train)
y_pred_gb = gb_model.predict(X_new_test)

print(f"ðŸ“‰ Gradient Boosting RMSLE: {compute_rmsle(y_test, y_pred_gb):.4f}")

