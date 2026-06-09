import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")


RANDOM_STATE = 42

ROOT_DATA_DIR = Path("/kaggle/input/playground-series-s5e8/")
train_df = pd.read_csv(os.path.join(ROOT_DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(ROOT_DATA_DIR, 'test.csv'))

print("\nTrain data info:")
print(train_df.info())


X = train_df.drop(columns=['y'])
y = train_df['y']

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")
print("---" * 40)
print("---" * 40)

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")


from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

# Create CatBoost pools
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
val_pool = Pool(X_val, y_val, cat_features=categorical_features)

# Train final model with best parameters
print("\nTraining final model with optimized parameters...")
best_params = {'iterations': 1562, 
                   'learning_rate': 0.14819908161704315, 
                   'depth': 8, 
                    'l2_leaf_reg': 4.824627420355487, 
                    'subsample': 0.7763836264246322, 
                    'random_strength': 4.714158156417848}
best_params['eval_metric'] = 'AUC'
best_params['random_seed'] = RANDOM_STATE
best_params['verbose'] = 100
best_params['early_stopping_rounds'] = 50

catboost_model = CatBoostClassifier(**best_params)
catboost_model.fit(train_pool, eval_set=val_pool, use_best_model=True)


# Make predictions
y_pred_proba = catboost_model.predict_proba(X_val)[:, 1]
y_pred = catboost_model.predict(X_val)

# Calculate metrics
auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"\n=== MODEL PERFORMANCE METRICS ===")
print(f"AUC Score: {auc_score:.4f}")

# Classification report
print(f"\nClassification Report:")
print(classification_report(y_val, y_pred))

# Feature importance
feature_importance = catboost_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"\n=== TOP 15 MOST IMPORTANT FEATURES ===")
print(importance_df.head(15))


plt.figure(figsize=(10, 8))
top_features = importance_df.head(15)
sns.barplot(data=top_features, y='feature', x='importance')
plt.title('Top 15 Feature Importances - CatBoost Baseline')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CatBoost Baseline Model')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - CatBoost Baseline')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


print("Making predictions!")
print("---" * 20)

test_predictions = catboost_model.predict_proba(test_df)[:, 1]
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
submission["y"] = test_predictions
submission.to_csv('submission.csv', index=False)

print("---" * 20)
print("Submission success!")


submission.head()




