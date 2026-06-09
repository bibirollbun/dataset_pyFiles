import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# Quick check
print(df_train.head())
print(df_train['y'].value_counts(normalize=True))


# 2. EDA

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


# 3. Preprocessing

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


# 4. Logistic Regression Model with Hyperparameter Tuning

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=500, solver='liblinear'))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# Evaluate on validation set
y_val_pred_proba = grid.predict_proba(X_val)[:,1]
print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_pred_proba))

RocCurveDisplay.from_predictions(y_val, y_val_pred_proba)
plt.show()


# 5. Retrain on full training set and predict on test

final_model = grid.best_estimator_
final_model.fit(X, y)

y_test_pred_proba = final_model.predict_proba(df_test)[:,1]


# 6. Submission File

submission = pd.DataFrame({
    'id': df_test['id'],
    'y': y_test_pred_proba
})
submission.head(5)

submission.to_csv("submission.csv", index=False)


