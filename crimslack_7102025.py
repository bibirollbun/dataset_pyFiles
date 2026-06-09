
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error




train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv") 
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv") 
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv") 


# Hedef değişken
y = train["accident_risk"]
X = train.drop(columns=["accident_risk", "id"])
X_test = test.drop(columns=["id"])


# Kategorik ve sayısal sütunları ayır
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()


# Ön işleme adımları
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])




# Base modeller
base_models = [
    ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method="hist", verbosity=0)),
    ("rf", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ("gb", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
]


# Meta model
meta_model = RidgeCV()



# Stacking regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5)



# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("stacking", stacking_model)
])


# Eğitim ve doğrulama
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)



# RMSE hesapla
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"Validation RMSE: {rmse:.5f}")


# Test tahminleri
test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 1)


# Submission dosyası
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_preds
})
submission.to_csv("submission.csv", index=False)

