# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os, time
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-whitegrid')
%matplotlib inline


# Load datasets and handle missing values
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Impute missing values in test (if any) for winddirection using the median
if test['winddirection'].isnull().sum() > 0:
    test['winddirection'].fillna(test['winddirection'].median(), inplace=True)

# Define feature set and target
X = train.drop(columns=['id', 'rainfall'])
y = train['rainfall']
X_test = test.drop(columns=['id'])
test_ids = test['id']


# List of columns that showed notable outliers in the images
outlier_cols = ['pressure', 'dewpoint', 'humidity', 'cloud', 'windspeed']

# If you want to treat all numeric columns, you can do:
# outlier_cols = X.columns.tolist()


def cap_outliers_iqr(df, columns, multiplier=1.5):
    df_capped = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df_capped[col] = df[col].clip(lower_bound, upper_bound)
    return df_capped

# Apply outlier capping to training data
X_capped = cap_outliers_iqr(X, outlier_cols)

# Apply the same capping approach to test data
# (Note: we compute IQR boundaries from train for a strictly correct approach)
# However, you can also compute each set's boundaries if you prefer.
X_test_capped = cap_outliers_iqr(X_test, outlier_cols)


feature = 'dewpoint'
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.boxplot(x=X[feature], color='lightgreen')
plt.title(f"Before Capping: {feature}")

plt.subplot(1,2,2)
sns.boxplot(x=X_capped[feature], color='lightblue')
plt.title(f"After Capping: {feature}")

plt.tight_layout()
plt.show()


# Apply MinMax scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_capped)
X_test_scaled = scaler.transform(X_test_capped)


# Train logistic regression and evaluate using cross-validation
log_reg = LogisticRegression(solver='liblinear', max_iter=1000, penalty='l2')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model using ROC AUC score
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=cv, scoring='roc_auc')
print("Cross-validated ROC AUC scores:", cv_scores)
print("Mean ROC AUC score:", cv_scores.mean())

# Fit the model on the full training set
log_reg.fit(X_scaled, y)

# Generate predictions for the test set (probability for class 1)
test_preds = log_reg.predict_proba(X_test_scaled)[:, 1]

# Create submission DataFrame
submission_df = pd.DataFrame({'id': test_ids, 'rainfall': test_preds})
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




