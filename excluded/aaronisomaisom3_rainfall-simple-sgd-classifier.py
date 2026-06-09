import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Load the dataset
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv', index_col='id')

# Display first few rows
display(train_df.head(10))
display("Train Shape", train_df.shape)
display("Test Shape", test_df.shape)

# Describe the data
display(train_df.describe())

# Display information about dtypes and missing values
display("Train Data Info:", train_df.info())

# Check target distribution
display("Target Distribution:", train_df['rainfall'].value_counts(normalize=True))

# Missing values
display("Train Missing Values:", train_df.isnull().sum().sum())
display("Test Missing Values:", test_df.isnull().sum().sum())

# Fix missing values in Test
test_df['winddirection'] = test_df['winddirection'].fillna(test_df['winddirection'].mean())

# Separate features and target
X = train_df.drop(columns=['rainfall'])
#test_df = test_df.drop(columns=['day'])
y = train_df['rainfall']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df)

model_sgd = SGDClassifier(max_iter=3000, early_stopping=True, 
                          random_state=42)
model = CalibratedClassifierCV(model_sgd, cv=10)
model.fit(X_scaled, y)

# Predict probabilities
y_pred_proba = model.predict_proba(X_scaled)[:, 1]

# Calculate AUC-ROC
auc_score = roc_auc_score(y, y_pred_proba)
print(f'Validation AUC-ROC Score: {auc_score:.4f}')

# Make predictions for submission
test_preds = model.predict_proba(X_test_scaled)[:, 1]

# Prepare submission file
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_preds
submission.to_csv('submission.csv', index=False)
print("Submission file saved!")
display(submission)

