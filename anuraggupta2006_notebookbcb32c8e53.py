# ============================================
# ğŸ“Œ Diabetes Risk Prediction - Kaggle Notebook
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# 1ï¸�âƒ£ Load Dataset
# ------------------------------

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

df.head()



# ------------------------------
# 2ï¸�âƒ£ Data Preprocessing
# - Drop: Outcome, Pregnancies, Insulin
# - Scale remaining features
# ------------------------------

X = df.drop(["Outcome", "Pregnancies", "Insulin"], axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled[:5]



# ------------------------------
# 3ï¸�âƒ£ Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



# ------------------------------
# 4ï¸�âƒ£ Train Random Forest Model
# ------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))



# ------------------------------
# 5ï¸�âƒ£ Confusion Matrix
# ------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



# ------------------------------
# 6ï¸�âƒ£ Feature Importance
# ------------------------------

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8,4))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance (Random Forest)")
plt.show()

importance_df



# ------------------------------
# 7ï¸�âƒ£ Predict on Custom Input
# (Manual Testing Example)
# ------------------------------

# Example input (You can modify these values)
input_data = np.array([[120, 70, 22, 27.5, 0.45, 32]])

# Scale input
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][1]

print("Prediction:", "High Risk" if prediction == 1 else "Low Risk")
print(f"Probability of Diabetes: {probability:.2%}")



# ------------------------------
# 8ï¸�âƒ£ Disclaimer
# ------------------------------
print("""
Note: This is a machine learning prediction model and should not be used 
as a substitute for professional medical diagnosis. Consult healthcare experts.
""")


