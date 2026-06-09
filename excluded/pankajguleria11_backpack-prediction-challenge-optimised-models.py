import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train = pd.concat([train, train_extra], ignore_index=True)

##Since id is not important to removing id column

train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Checking dataset shape
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")

train.head()


import missingno as msno

# Visualizing the missing values
msno.bar(train, color='limegreen')

# Handling missing values
# Fill missing values in numerical columns using median
numerical_cols = train.select_dtypes(include=['number']).columns
train[numerical_cols] = train[numerical_cols].fillna(train[numerical_cols].median())

# Fill missing values in categorical columns using mode
categorical_cols = train.select_dtypes(exclude=['number']).columns
train[categorical_cols] = train[categorical_cols].fillna(train[categorical_cols].mode().iloc[0])

import matplotlib.pyplot as plt
import seaborn as sns

# Histograms for numerical columns
train.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.show()



# Count plots for all categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns

plt.figure(figsize=(12, 6))
for col in categorical_cols:
    sns.countplot(y=train[col], order=train[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()




##Checking price disrtibution with all categorical features using boxplots

cat_feature = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
plt.figure(figsize=(12, 12))

for i, col in enumerate(cat_feature, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=train[col], y=train["Price"], hue=train[col], palette="Dark2")
    plt.xticks(rotation=90)
    plt.ylabel("Price")
    plt.title(f"Price Distribution by {col}")

plt.tight_layout()
plt.show()


# Select numerical columns
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove Price if it's already in numeric_cols to avoid duplication
if 'Price' in numeric_cols:
    numeric_cols.remove('Price')

# Pairplot
sns.pairplot(train[[*numeric_cols, 'Price']])
plt.show()


# Select only numerical features for correlation analysis
numerical_features = train.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_matrix = numerical_features.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="Dark2", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


X = train.drop(columns=['Price'])  # Features
y = train['Price']  # Target


# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


# Handling Missing Values
num_imputer = SimpleImputer(strategy='median')  # Fill missing numerical values with median
cat_imputer = SimpleImputer(strategy='most_frequent')  # Fill missing categorical values with most frequent


preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', num_imputer), ('scaler', StandardScaler())]), numeric_cols),
    ('cat', Pipeline([('imputer', cat_imputer), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
])


# Transform training data, then spliting into training and validation sets and then reducing training size for faster processing.
X_processed = preprocessor.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)



# Define function to optimize Random Forest
def rf_evaluate(n_estimators, max_depth):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_val)
    return -mean_squared_error(y_val, y_pred)  # Minimize RMSE


# Define Bayesian Optimization for Random Forest
rf_bo = BayesianOptimization(
    f=rf_evaluate,
    pbounds={'n_estimators': (20, 100), 'max_depth': (3, 20)},
    random_state=42
)
rf_bo.maximize(init_points=5, n_iter=10)

# Get best Random Forest parameters
best_rf_params = rf_bo.max
best_rf = RandomForestRegressor(
    n_estimators=int(best_rf_params['params']['n_estimators']),
    max_depth=int(best_rf_params['params']['max_depth']),
    random_state=42,
    n_jobs=-1
)

# Define function to optimize XGBoost
def xgb_evaluate(n_estimators, learning_rate, max_depth):
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_val)
    return -mean_squared_error(y_val, y_pred)  # Minimize RMSE



# Define Bayesian Optimization for XGBoost
xgb_bo = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={'n_estimators': (50, 200), 'learning_rate': (0.01, 0.3), 'max_depth': (3, 10)},
    random_state=42
)
xgb_bo.maximize(init_points=5, n_iter=10)



# Get best XGBoost parameters
best_xgb_params = xgb_bo.max
best_xgb = XGBRegressor(
    n_estimators=int(best_xgb_params['params']['n_estimators']),
    learning_rate=best_xgb_params['params']['learning_rate'],
    max_depth=int(best_xgb_params['params']['max_depth']),
    random_state=42,
    n_jobs=-1
)



# Define models after Bayesian Optimization
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest (Optimized)": best_rf,
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "XGBoost (Optimized)": best_xgb,
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
}



# Train and evaluate models
rmse_scores = {}

for name, model in models.items():
    model.fit(X_train_sample, y_train_sample)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores[name] = rmse
    print(f"{name} RMSE: {rmse:.4f}")



# Select best model
best_model_name = min(rmse_scores, key=rmse_scores.get)
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name} with RMSE: {rmse_scores[best_model_name]:.4f}")

# Train best model on full dataset
best_model.fit(X_processed, y)



# Preprocess test data and predict prices
X_test_processed = preprocessor.transform(test)
test_predictions = best_model.predict(X_test_processed)

# Implement Stacking Regressor
stacking_model = StackingRegressor(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('gbr', GradientBoostingRegressor(n_estimators=50, random_state=42))
    ],
    final_estimator=LinearRegression(),
    cv=5
)




# Train and evaluate Stacking model
stacking_model.fit(X_train_sample, y_train_sample)
y_pred_stacked = stacking_model.predict(X_val)
rmse_stacked = np.sqrt(mean_squared_error(y_val, y_pred_stacked))
print(f"\nStacking RMSE: {rmse_stacked:.4f}")

# Use Stacking model if it performs better
if rmse_stacked < rmse_scores[best_model_name]:
    best_model = stacking_model
    print("\nUsing Stacking Model as Final Model")

# Train final model on full dataset
best_model.fit(X_processed, y)

# Predict on test data
final_predictions = best_model.predict(X_test_processed)


# Save predictions in submission format
submission = sample_submission.copy()
submission['Price'] = final_predictions
submission.to_csv('/kaggle/working/final_submission.csv', index=False)

print("Predictions saved as final_submission.csv")

