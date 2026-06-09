import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from jet_tag_utils import load_processed_data 
from jet_tag_plotting_utils import plot_confusion_matrix, plot_roc_curve


# Load data
# Create feature matrix and labels

X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, test_ids= load_processed_data()

X_train.shape


X_train.head()


# Initialize and train model
model = xgb.XGBClassifier(
    n_estimators=100,  # Number of boosting rounds
    max_depth=5,      # Maximum tree depth
    learning_rate=0.1, # Step size shrinkage
    objective='binary:logistic',  # Binary classification
    random_state=42
)

# Train the model
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          verbose=True)


# Make predictions

# this gives us probabilities for both categories - we only want for ttbar, so we select one column
# with a binary classification, the probability for one category implies the other 
y_pred = model.predict_proba(X_val)[:, 1]

# to test accuracy and confusion matrix, we need labels 0 and 1, so we set that based on a threshold
discrete_pred = np.where(y_pred > 0.5, 1, 0)
# Calculate accuracy
accuracy = accuracy_score(y_val, np.where(y_pred > 0.5, 1, 0))
print(f"Test Accuracy: {accuracy:.4f}")

# Plot confusion matrix
plot_confusion_matrix(y_val, discrete_pred)


plot_roc_curve(y_val, y_pred)


# Plot feature importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout() 
plt.show()


# test_predictions = model.predict_proba(X_test)[:, 1]
# solution = pd.DataFrame({'id':test_ids, 'label':test_predictions})
# solution.to_csv('solutions.csv', index=False)




