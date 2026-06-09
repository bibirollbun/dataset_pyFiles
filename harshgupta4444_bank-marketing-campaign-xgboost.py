# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


test.head()


train.shape


# Target distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='y', data=train)
plt.title('Distribution of Target Variable (y)')
plt.show()




# Calculate percentage
target_perc = train['y'].value_counts(normalize=True) * 100
print(f"Target distribution:\n{target_perc}")


# List of numerical features
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Summary statistics
print("Numerical Features Summary:")
train.describe()


# Plot distributions
fig, axes = plt.subplots(4, 2, figsize=(15, 15))
for i, col in enumerate(num_cols):
    sns.histplot(train[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Correlation matrix
plt.figure(figsize=(12, 8))
corr = train[num_cols + ['y']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


# List of categorical features
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


# Plot categorical distributions
for col in cat_cols:
    plt.figure(figsize=(12, 5))
    sns.countplot(x=col, data=train, order=train[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {col}')
    plt.show()


# Plot target rate by categories
for col in cat_cols:
    plt.figure(figsize=(12, 5))
    sns.barplot(x=col, y='y', data=train, order=train[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'Subscription Rate by {col}')
    plt.show()


# Label Encoding for categorical variables
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


# Prepare features and target
X = train.drop(['id', 'y'], axis=1)
y = train['y']
X_test = test.drop('id', axis=1)


# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize XGBoost classifier
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)



# Train model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)


# Validation predictions
val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print(f"\nValidation AUC: {auc:.4f}")



# Feature importance
xgb.plot_importance(model, max_num_features=15)
plt.title('Feature Importance')
plt.show()


# Predict on test set
test_preds = model.predict_proba(X_test)[:, 1]


# Create submission file
submission = pd.DataFrame({'id': test['id'], 'y': test_preds})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
submission.head()




