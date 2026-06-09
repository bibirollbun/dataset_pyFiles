import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score



train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


submission.head()


train_df.head()


train_df.info()


train_df.isna().sum()


train_df.duplicated().sum()


train_df.describe()


test_df.head()


test_df.info()


test_df.isna().sum()


test_df = test_df.fillna(test_df.mean())


test_df.duplicated().sum()


test_df.describe()


# Create correlation matrix
corr_matrix = train_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


X = train_df.drop(['id', 'rainfall'], axis=1)
y = train_df['rainfall']

# Define numeric features
numeric_features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
                   'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
                   'windspeed']

# Create preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create XGBoost pipeline with reasonable baseline parameters
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42
    ))
])

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(
    xgb_pipeline,
    X, 
    y,
    cv=kf,
    scoring='roc_auc'
)

# Print cross-validation results
print(f"5-Fold CV AUC-ROC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Fit the model on all training data
xgb_pipeline.fit(X, y)

# Calculate AUC-ROC on training data
y_prob = xgb_pipeline.predict_proba(X)[:, 1]
auc_roc = roc_auc_score(y, y_prob)
print(f"Training AUC-ROC: {auc_roc:.4f}")

# Get feature importances
feature_names = numeric_features
feature_importances = xgb_pipeline.named_steps['classifier'].feature_importances_

# Create DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.show()


# Make predictions on test data
X_test = test_df.drop(['id'], axis=1)  # Drop only the id column
test_predictions = xgb_pipeline.predict_proba(X_test)[:, 1]  # Get probability scores

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_predictions
})

# Save to CSV file
submission.to_csv('submission.csv', index=False)

