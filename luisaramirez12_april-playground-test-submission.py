import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

# Load training and test data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

# Drop unused text columns
X = train.drop(columns=["id", "Listening_Time_minutes", "Podcast_Name", "Episode_Title"])
y = train["Listening_Time_minutes"]
X_test = test.drop(columns=["id", "Podcast_Name", "Episode_Title"])



# Identify column types
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include="object").columns

# Pipelines for numeric and categorical features
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# Combine pipelines into a column transformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])


# Final modeling pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# Train the model
model.fit(X, y)

# Predict on test set
preds = model.predict(X_test)


# Generate submission DataFrame
submission = pd.DataFrame({
    "id": test["id"],
    "Listening_Time_minutes": preds
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved!")

