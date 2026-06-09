# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data=pd.read_csv("/kaggle/input/predict-loan-default/train.csv")
test_data=pd.read_csv("/kaggle/input/predict-loan-default/test.csv")
sub_data=pd.read_csv("/kaggle/input/predict-loan-default/sample_submmission.csv")


train_data


import matplotlib.pyplot as plt
import seaborn as sns


print(pd.crosstab(train_data['STATE'],train_data['Risk_Flag']))


# ðŸ”¹ Stacked Bar Plot
pd.crosstab(train_data['STATE'],train_data['Risk_Flag']).plot(kind='bar', stacked=True, colormap='viridis', figsize=(20,6))
                                                                                                                    
plt.title('Risk Flags Distribution by Profession')
plt.xlabel('STATE')
plt.ylabel('Count')
plt.legend(title="Risk Flags")
plt.xticks(rotation=45)
plt.show()


from category_encoders.target_encoder import TargetEncoder

# Define the encoder
encoder = TargetEncoder(cols=['Profession', 'CITY'])

# Fit on train_data using "Risk_Flag"
encoder.fit(train_data[['Profession', 'CITY']], train_data['Risk_Flag'])

# Transform train_data & test_data
train_data[['Profession', 'CITY']] = encoder.transform(train_data[['Profession', 'CITY']])
test_data[['Profession', 'CITY']] = encoder.transform(test_data[['Profession', 'CITY']])  # Apply the learned mapping

# Fill NaN values in test_data with train_data mean (handles unseen categories)
test_data[['Profession', 'CITY']] = test_data[['Profession', 'CITY']].fillna(train_data[['Profession', 'CITY']].mean())






train_data


test_data.head(20)


[train_data[col].unique() for col in train_data.columns if len(train_data[col].unique())<5]


from category_encoders.target_encoder import TargetEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder

# Define encoders
target_enc = TargetEncoder(cols=['Married.Single', 'House_Ownership', 'Car_Ownership'])
loo_enc = LeaveOneOutEncoder(cols=['Married.Single', 'House_Ownership', 'Car_Ownership'])

# Fit on train_data (with "Risk_Flag") and transform both datasets
train_data_encoded = target_enc.fit_transform(train_data[['Married.Single', 'House_Ownership', 'Car_Ownership']], train_data['Risk_Flag'])
test_data_encoded = target_enc.transform(test_data[['Married.Single', 'House_Ownership', 'Car_Ownership']])  # Transform test data

# Ensure missing categories are handled
test_data_encoded.fillna(train_data_encoded.mean(), inplace=True)

# Assign back to original DataFrame
train_data[['Married.Single', 'House_Ownership', 'Car_Ownership']] = train_data_encoded
test_data[['Married.Single', 'House_Ownership', 'Car_Ownership']] = test_data_encoded



test_data.head(20)


numeric_columns = ['Income', 'Age', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS']

plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=test_data[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


from scipy.stats import zscore

z_scores = train_data[numeric_columns].apply(zscore)  # Calculate Z-scores
outliers = (z_scores.abs() > 3).any(axis=1)  # Mark rows where any column has a Z-score > 3

print(f"Number of outliers: {outliers.sum()}")
df_outliers = train_data[outliers]  # Show the outlier rows
df_outliers.head()



train_data['Risk_Flag'].value_counts().unique



X_train=train_data.drop(columns=['Id','STATE','Risk_Flag'])
y_train= train_data['Risk_Flag']
X_test=test_data.drop(columns=['Id','STATE'])


X_test.info()


import catboost as cb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Identify categorical features
cat_features = ['Married.Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY']

# Handle missing categorical data
for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("Missing")
    X_test[col] = X_test[col].astype(str).fillna("Missing")

# Handle missing numerical data and apply log transformation
num_features = [col for col in X_train.columns if col not in cat_features]
for col in num_features:
    X_train[col] = X_train[col].apply(lambda x: np.log1p(x) if x > 0 else 0)
    X_test[col] = X_test[col].apply(lambda x: np.log1p(x) if x > 0 else 0)

# Train-validation split
X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Define CatBoost Model
cat_model = cb.CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function='Logloss',
    cat_features=cat_features,
    eval_metric='F1',
    early_stopping_rounds=150,
    auto_class_weights='Balanced',
    l2_leaf_reg=10,
    bagging_temperature=1.0,
    max_bin=64,
    verbose=200
)

# Train Model with Early Stopping
cat_model.fit(
    X_train_split, y_train_split,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=150,
    verbose=200
)

# Find the Best Threshold
y_valid_proba = cat_model.predict_proba(X_valid)[:, 1]
best_threshold = 0.5  # Default

for threshold in np.arange(0.1, 0.9, 0.05):
    y_valid_pred = (y_valid_proba > threshold).astype(int)
    score = f1_score(y_valid, y_valid_pred)
    if score > f1_score(y_valid, (y_valid_proba > best_threshold).astype(int)):
        best_threshold = threshold

print("Best Threshold:", best_threshold)

# Make Predictions with Best Threshold
y_test_pred = (cat_model.predict_proba(X_test)[:, 1] > best_threshold).astype(int)



submission = pd.DataFrame({
    'Id': sub_data['Id'],  # Use the 'Id' from the dataset
    'Risk_Flag': y_test_pred  # Predicted 'Risk_Flag' values
})

# Save to CSV without the index column
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file 'submission.csv' created successfully!")




