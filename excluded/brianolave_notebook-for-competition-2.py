import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error

# Load the data
train = pd.read_csv("/kaggle/input/gvu-spring-2025-data-454-project-1/train.csv")
test = pd.read_csv("/kaggle/input/gvu-spring-2025-data-454-project-1/test.csv")
sample_submission = pd.read_csv("/kaggle/input/gvu-spring-2025-data-454-project-1/sample_submission.csv")


#Exploratory Data Analysis
train.head()



test.head()



train.describe()



# Check for missing values
print("\nMissing Values:")
print(train.isnull().sum())



# Define target and features
target = "SALE_PRC"
features = [
    "LND_SQFOOT", "TOT_LVG_AREA", "SPEC_FEAT_VAL", "RAIL_DIST", "OCEAN_DIST", 
    "WATER_DIST", "CNTR_DIST", "SUBCNTR_DI", "HWY_DIST", "age", "avno60plus", 
    "structure_quality", "month_sold", "LATITUDE", "LONGITUDE"
]


# Extract feature matrix and target vector
X = train[features]
y = train[target]
X_test = test[features]


# Feature Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = train[features + [target]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple regression models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}


# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_predictions)
    print(f"{name} Validation MAE: {mae:.2f}")

# Choose the best model
best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
test_predictions = best_model.predict(X_test)



# Prepare submission file
test_ids = test["id"]
submission = pd.DataFrame({"id": test_ids, "SALE_PRC": test_predictions})
submission.to_csv("submission.csv", index=False)

print("Submission file created: submission.csv")

