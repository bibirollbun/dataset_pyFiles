import shap
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


train["rainfall"].value_counts()


correlation = train.corr()["rainfall"].abs().sort_values(ascending=False)
print(correlation)


corr_threshold = 0.3 
selected_features = correlation[correlation > corr_threshold].index.tolist()
print(selected_features)


selected_features.remove("rainfall")


train_selected = train[selected_features]
print(train_selected)


scaler = StandardScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_selected), columns=train_selected.columns)
print(f"df_scaled" ,train_scaled)


vif_data = pd.DataFrame()
vif_data["Feature"] = train_scaled.columns
vif_data["VIF"] = [variance_inflation_factor(train_scaled.values, i) for i in range(train_scaled.shape[1])]
print("VIF DATA")
print(vif_data["VIF"])


vif_threshold = 5
low_vif_features = vif_data[vif_data["VIF"] < vif_threshold]["Feature"].tolist()
print("Final Selected Features:", low_vif_features)


train_final = train[low_vif_features + ["rainfall"]]
print(train_final)


X = train_final.drop(columns=["rainfall"])
y = train_final["rainfall"]


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
train_balanced = pd.DataFrame(X_resampled, columns=X.columns)
train_balanced["rainfall"] = y_resampled


print("Final Selected Features After VIF Filtering:\n", low_vif_features)


print("Dataset before SMOTE:", train_final["rainfall"].value_counts().to_dict())
print("Dataset after SMOTE:", train_balanced["rainfall"].value_counts().to_dict())


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


y_pred_proba = model.predict_proba(X_test)[:, 1]  
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nğŸ”� Model Evaluation Metrics:")
print(f"âœ… Accuracy: {accuracy:.4f}")
print(f"âœ… Precision: {precision:.4f}")
print(f"âœ… Recall: {recall:.4f}")
print(f"âœ… F1 Score: {f1:.4f}")
print(f"âœ… AUC-ROC Score: {roc_auc:.4f}")


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')  # Random model line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


import joblib
joblib.dump(model, "rainfall_prediction_model.pkl")
print("\nModel training complete. Trained model saved as 'rainfall_prediction_model.pkl'.")


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


best_model = joblib.load("rainfall_prediction_model.pkl")


test_ids = test["id"]
test = test.drop(columns=["id"], errors="ignore")
test = test.fillna(test.median())


scaler = StandardScaler()
original_features = ['cloud', 'sunshine', 'humidity'] 
test[original_features] = scaler.fit_transform(test[original_features])
X_test_final = test[original_features] 
predictions = best_model.predict(X_test_final)


submission["rainfall"] = predictions  
submission["id"] = test_ids  
submission.to_csv("submission.csv", index=False)

