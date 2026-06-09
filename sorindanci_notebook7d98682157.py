import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Daten laden
df_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Zielvariable und Features
y = df_train["Price"]
X = df_train.drop(["Price", "id"], axis=1)
X_test_final = df_test.drop("id", axis=1)

# Kategorische Variablen kodieren
cat_cols = X.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in cat_cols:
    X[col] = encoder.fit_transform(X[col].astype(str))
    X_test_final[col] = encoder.transform(X_test_final[col].astype(str))

# Fehlende Werte mit Mittelwert ersetzen
X = X.fillna(X.mean(numeric_only=True))
X_test_final = X_test_final.fillna(X.mean(numeric_only=True))

# Trainings- und Validierungssets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelltraining
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Vorhersagen und Bewertung
y_pred = model.predict(X_val)
print("Mean Squared Error:", mean_squared_error(y_val, y_pred))
print("R² Score:", r2_score(y_val, y_pred))

# Finale Vorhersagen auf dem Testset
final_preds = model.predict(X_test_final)

# Submission-Datei erstellen
submission = pd.DataFrame({
    "id": df_test["id"],
    "Price": final_preds
})
submission.to_csv("submission_baseline.csv", index=False)


