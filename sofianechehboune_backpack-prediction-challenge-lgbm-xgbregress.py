
import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Separate features and target
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


train.head()


train.info()


train.describe().transpose()


train.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from IPython.display import display, Image


train.columns


sns.barplot(data = train.head(), x = 'id' , y = 'Price')




# Set renderer for Kaggle
pio.renderers.default = "iframe"  # Essaie aussi "svg" ou "notebook"

# Load data
data = train

# Encode categorical columns
for col in ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]:
    data[col] = data[col].astype("category").cat.codes

# Creating the Parallel Coordinates graph
fig = px.parallel_coordinates(
    data,
    dimensions=data.columns,
    color="Price",
    color_continuous_scale=px.colors.sequential.Viridis
)

# Displaying the graph (Different solutions)
try:
    fig.show()
except:
    fig.write_image("plot.png")
    display(Image("plot.png"))



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# SÃ©parer les features et la cible
X = train.drop(columns=["Price", "id"])  # Supprime la cible et l'ID
y = train["Price"]
X_test = test.drop(columns=["id"])  # Supprime l'ID dans test


# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns


# Preprocessing pipelines with imputation and Replace NaNs with the median
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Replace NaNs with the median
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Remplace les NaN par la valeur la plus frÃ©quente
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Test XGBoost and LightGBM
models = {
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
}

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    print(f"\nðŸ”¹ EntraÃ®nement du modÃ¨le {name}...")



# Test XGBoost and LightGBM
models = {
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
}

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    print(f"\nðŸ”¹ EntraÃ®nement du modÃ¨le {name}...")

    # Pipeline with the model
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

   # Train the model
    full_pipeline.fit(X_train, y_train)

   # Evaluate the model
    y_pred = full_pipeline.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    print(f"âœ… {name} - MAE sur validation : {mae:.2f}")

    # Prediction on test.csv
    test_predictions = full_pipeline.predict(X_test)


# Generate Submission
    submission = pd.DataFrame({"id": test["id"], "price": test_predictions})
    submission.to_csv(f"submission_{name}.csv", index=False)



print(f"ðŸ“‚Generated submission file : submission_{name}.csv ({len(submission)} lignes)")

print("submission.csv")


