import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns


data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


data = data.drop(['id'], axis=1)
print(data.shape)
data.head()


data.dtypes


data.isna().sum() / data.shape[0]


data['rainfall'].value_counts()


data.describe()


# Compute correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


df = data.copy()


# Create interaction features
df["cloud_humidity"] = df["cloud"] * df["humidity"]
df["sunshine_pressure"] = df["sunshine"] * df["pressure"]

# Show transformed data
df.head()


from sklearn.feature_selection import mutual_info_regression

# Select features and target
X = df.drop(columns=["rainfall"])  # Features
y = df["rainfall"]  # Target

# Compute Mutual Information
mi_scores = mutual_info_regression(X, y, random_state=42)

# Convert to DataFrame for better visualization
mi_df = pd.DataFrame({"Feature": X.columns, "Mutual Information": mi_scores})
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)

# Display MI scores
print(mi_df)


# Select features with MI > 0.02
selected_features = mi_df[mi_df["Mutual Information"] > 0.02]["Feature"].tolist()

# Print selected features
print("Selected Features:", selected_features)

# Filter the dataset with selected features
X_selected = X[selected_features]


X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.15, stratify=y, random_state=42
)


# Define parameter grid for tuning Random Forest
rf_param_grid = {
    "n_estimators": [100,200,250],
    "max_depth": [4, 6, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [5, 10,15],
    'max_features': [2,6,None],
    
}

# Initialize Random Forest model
rf_model = RandomForestClassifier(class_weight="balanced",criterion ='entropy',random_state=42)

# Perform Grid Search with Cross Validation
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)

# Best model
best_rf = rf_grid_search.best_estimator_

# Predict probabilities for AUC-ROC
y_rf_probs = best_rf.predict_proba(X_val)[:, 1]

# Compute ROC-AUC
rf_auc_score = roc_auc_score(y_val, y_rf_probs)

# Print results
print(f"Best Parameters: {rf_grid_search.best_params_}")
print(f"Random Forest AUC-ROC Score: {rf_auc_score:.4f}")


feature_importance = best_rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X_train.columns,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("RandomForest Feature Importance")
plt.gca().invert_yaxis()  
plt.show()


test.isna().sum() / test.shape[0]


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].mean())


df_test = test.copy()
df_test = df_test.drop(['id'], axis=1)
# Create interaction features
df_test["cloud_humidity"] = df_test["cloud"] * df_test["humidity"]
df_test["sunshine_pressure"] = df_test["sunshine"] * df_test["pressure"]

df_test = df_test[X_selected.columns]

# Show transformed data
df_test.head()


# Predict probabilities for AUC-ROC
y_pred_proba = best_rf.predict_proba(df_test)[:, 1]


submission = pd.DataFrame({"id": test['id'], "rainfall": y_pred_proba})  
submission.to_csv("submission.csv", index=False)
submission.head()

