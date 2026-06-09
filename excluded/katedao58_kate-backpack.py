import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


# Load datasets 
df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
extra_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


# Split the data into training and development sets:
X_train, X_dev, y_train, y_dev = train_test_split(df.drop(columns=['Price']), df['Price'],
                                                  test_size=0.2, random_state=1231)


# Split extra data
X_extra = extra_df.drop(columns=["Price"])
y_extra = extra_df["Price"]

# Append extra data to training set
X_train_combined = pd.concat([X_train, X_extra], ignore_index=True)
y_train_combined = pd.concat([y_train, y_extra], ignore_index=True)


# Define feature types
continuous_features = ["Compartments", "Weight Capacity (kg)"]
categorical_features = list(set(df.columns) - {"id", "Price"} - set(continuous_features))

# Define preprocessing steps
continuous_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", continuous_transformer, continuous_features),
    ("cat", categorical_transformer, categorical_features)
])


# GridSearchCV (using train.csv only)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
    
param_grid = {
    "model__max_depth": [4, 6, 8],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 3, 5],
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=1231))
])

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Evaluate on dev set
best_model = grid_search.best_estimator_
y_pred_dev = best_model.predict(X_dev)
rmse_dev = root_mean_squared_error(y_dev, y_pred_dev)

print("Best parameters:", grid_search.best_params_)
print(f"RMSE on Dev Set: {rmse_dev:.4f}")


# Create the model with specified hyperparameters
model = DecisionTreeRegressor( 
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=1231
)

# Create pipeline with preprocessing
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Fit the model on the combined training data
pipeline.fit(X_train_combined, y_train_combined)

# Predict on test data
X_test = test_df
y_pred_test = pipeline.predict(X_test)


# Create submission DataFrame
submission = pd.DataFrame({"id": test_df["id"], "Price": y_pred_test})


# Save to CSV
submission.to_csv("backpack_submission.csv", index=False)




