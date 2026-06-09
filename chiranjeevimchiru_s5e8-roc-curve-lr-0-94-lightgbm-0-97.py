import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay

from lightgbm import LGBMClassifier





df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Quick check
print(df_train.head())
print(df_train['y'].value_counts(normalize=True))



# Target distribution
sns.countplot(x='y', data=df_train)
plt.title("Target Distribution")
plt.show()

# Numerical features distribution
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
df_train[num_cols].hist(bins=30, figsize=(15,10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_train[num_cols + ['y']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


X = df_train.drop(columns=['y'])
y = df_train['y']

categorical_cols = [col for col in X.columns if X[col].dtype == 'object' and col != 'id']
numerical_cols = [col for col in X.columns if X[col].dtype != 'object' and col != 'id']

# Define Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])



#  Logistic Regression Model with Hyperparameter Tuning

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=500, solver='liblinear'))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}

grid_lr = GridSearchCV(pipe_lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_lr.fit(X_train, y_train)

print("[Logistic Regression] Best Parameters:", grid_lr.best_params_)
print("[Logistic Regression] Best CV Score:", grid_lr.best_score_)



y_val_pred_proba_lr = grid_lr.predict_proba(X_val)[:,1]
auc_lr = roc_auc_score(y_val, y_val_pred_proba_lr)
print("Validation ROC-AUC (Logistic Regression):", auc_lr)


#  LightGBM Model

pipe_lgbm = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LGBMClassifier(n_estimators=200, random_state=42))
])

param_grid_lgbm = {
    'clf__num_leaves': [31, 50],
    'clf__learning_rate': [0.01, 0.1],
    'clf__max_depth': [-1, 10]
}

grid_lgbm = GridSearchCV(pipe_lgbm, param_grid_lgbm, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_lgbm.fit(X_train, y_train)

print("[LightGBM] Best Parameters:", grid_lgbm.best_params_)
print("[LightGBM] Best CV Score:", grid_lgbm.best_score_)




y_val_pred_proba_lgbm = grid_lgbm.predict_proba(X_val)[:,1]
auc_lgbm = roc_auc_score(y_val, y_val_pred_proba_lgbm)
print("Validation ROC-AUC (LightGBM):", auc_lgbm)




print("======================")
print(f"Logistic Regression AUC: {auc_lr:.4f}")
print(f"LightGBM AUC: {auc_lgbm:.4f}")
print("======================")

# ROC Curve Comparison
RocCurveDisplay.from_predictions(y_val, y_val_pred_proba_lr, name="Logistic Regression")
RocCurveDisplay.from_predictions(y_val, y_val_pred_proba_lgbm, name="LightGBM")
plt.title("ROC Curve Comparison")
plt.show()

# Bar chart comparison
plt.bar(["Logistic Regression", "LightGBM"], [auc_lr, auc_lgbm], color=["skyblue", "lightgreen"])
plt.ylabel("Validation AUC")
plt.title("Model AUC Comparison")
plt.show()


# Retrain Best Model on Full Train and Predict on Test
best_model = grid_lgbm if auc_lgbm > auc_lr else grid_lr
final_model = best_model.best_estimator_
final_model.fit(X, y)

y_test_pred_proba = final_model.predict_proba(df_test)[:,1]



submission = pd.DataFrame({
    'id': df_test['id'],
    'y': y_test_pred_proba
})

submission.to_csv("submission.csv", index=False)

submission.head(5)

