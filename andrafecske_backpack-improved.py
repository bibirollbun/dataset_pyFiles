import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")


train["Weight Capacity (kg)"] = train["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())
test["Weight Capacity (kg)"] = test["Weight Capacity (kg)"].fillna(train["Weight Capacity (kg)"].median())


cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
for col in cat_cols:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")


features = cat_cols + ["Weight Capacity (kg)"]
target = "Price"

X = train[features]
y = train[target]
X_test = test[features]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


encoder = TargetEncoder(cols=cat_cols)
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)



model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    random_state=42
)
model.fit(X_train_encoded, y_train)


val_preds = model.predict(X_val_encoded)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print("Validation RMSE:", rmse)



import lightgbm as lgb
import matplotlib.pyplot as plt

# Plot feature importance directly from the model
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("Feature Importance (gain)")
plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Verteilung der Zielvariablen (Preis)
sns.histplot(train["Price"], kde=True)
plt.title("Verteilung der Preise")
plt.xlabel("Preis")
plt.ylabel("Anzahl")
plt.show()

# Durchschnittlicher Preis pro Brand
plt.figure(figsize=(12,6))
avg_prices = train.groupby("Brand")["Price"].mean().sort_values(ascending=False)
sns.barplot(x=avg_prices.index, y=avg_prices.values)
plt.title("Durchschnittlicher Preis pro Brand")
plt.xticks(rotation=90)
plt.ylabel("Durchschnittlicher Preis")
plt.tight_layout()
plt.show()



test_preds = model.predict(X_test_encoded)
submission["Price"] = test_preds
submission.to_csv("target_encoded_submission.csv", index=False)


