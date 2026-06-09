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


train, test, submission = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv'),pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv'),pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
train.head()


print('train nan values \n ', f'{train.isna().sum()}')
print('test nan values \n ', f'{test.isna().sum()}')


print(set(train['poutcome'].values))


import matplotlib.pyplot as plt
x = (train['y']==1).sum()
y = (train['y']==0).sum()
print(f'x:{x} , y:{y}')
# Create labels and values
labels = ['1s', '0s']
values = [x, y]

# Plot
plt.bar(labels, values)
plt.title("Class Counts in Train Dataset")
plt.xlabel("Class")
plt.ylabel("Count")

# show the value on top of each bar
for i, v in enumerate(values):
    plt.text(i, v + 10, str(v), ha='center')

plt.show()


train, test = train.drop(columns='id'), test.drop(columns='id')
train.describe()


test.describe()


train.groupby(["age", "job"])[["y"]].mean()


train.groupby(["age", "job"])[["y"]].sum()


students_data = train[train['job'] == 'student']
students_data.head()


x = (students_data['y']==1).sum()
y = (students_data['y']==0).sum()
print(f'x:{x} , y:{y}')
# Create labels and values
labels = ['1s', '0s']
values = [x, y]

# Plot
plt.bar(labels, values)
plt.title("Class Counts in Train Dataset")
plt.xlabel("Class")
plt.ylabel("Count")

# show the value on top of each bar
for i, v in enumerate(values):
    plt.text(i, v + 10, str(v), ha='center')

plt.show()


def y_plot_counter(data, job_name):
    x = (data['y']==1).sum()
    y = (data['y']==0).sum()
    print(f'Job: {job_name} - 1s:{x} , 0s:{y}')
    
    # Create labels and values
    labels = ['1s', '0s']
    values = [x, y]
    
    # Plot
    plt.figure(figsize=(6, 4))  # Create new figure for each plot
    plt.bar(labels, values)
    plt.title(f"Class Counts for {job_name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    # show the value on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 10, str(v), ha='center')
    
    plt.show()

def job_y_counter(data, jobs):
    jobs_data = []
    for job in jobs:
        job_data = data[data['job'] == job]
        jobs_data.append((job_data, job))  # Store both data and job name
    return jobs_data

# Get unique jobs if you don't have the 'jobs' list
jobs = train['job'].unique()

# Get filtered data for each job
jobs_counter = job_y_counter(train, jobs)

# Create plot for each job
for job_data, job_name in jobs_counter:
    if len(job_data) > 0:  # Only plot if there's data for this job
        y_plot_counter(job_data, job_name)


train.groupby(["housing", "loan"])[["y"]].count()


mean_balance_per_education = train.groupby("education")["balance"].mean()
print("Mean balance per education:")
mean_balance_per_education

# Then count y values per education
y_count_per_education = train.groupby("education")["y"].count()
print("\nY count per education:")
y_count_per_education


train['balance_bins'] = pd.cut(train['pdays'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
test['balance_bins'] = pd.cut(test['pdays'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
# Now group by education and balance bins
result = train.groupby(["education", "balance_bins"])[["y"]].count() 
# Group by education and balance_bins, then count 0s and 1s separately
result = train.groupby(["education", "balance_bins", "y"]).size().unstack(fill_value=0)

# Rename columns for clarity
result.columns = ['0', '1']

print(result)


train['pdays_bins'] = pd.cut(train['pdays'], bins=3, labels=['negative', 'positive', 'positive_high'])
test['pdays_bins'] = pd.cut(test['pdays'], bins=3, labels=['negative', 'positive', 'positive_high'])
result = train.groupby(["loan", "pdays_bins"])[["y"]].count() 
# Group by education and balance_bins, then count 0s and 1s separately
result = train.groupby(["loan", "pdays_bins", "y"]).size().unstack(fill_value=0)

# Rename columns for clarity
result.columns = ['0', '1']

print(result)


# Attempt to parse the date, coerce errors to NaT
train['date'] = pd.to_datetime(train['day'].astype(str) + '-' + train['month'] + '-2023', format='%d-%b-%Y', errors='coerce')
test['date'] = pd.to_datetime(test['day'].astype(str) + '-' + test['month'] + '-2023', format='%d-%b-%Y', errors='coerce')
# Check for any NaT values
invalid_dates = train[train['date'].isna()]
invalid_dates_test = test[test['date'].isna()]
print("Invalid dates:\n", invalid_dates)
print("Invalid test dates:\n", invalid_dates_test)
# Continue as needed
train['day_of_week'] = train['date'].dt.dayofweek
test['day_of_week'] = test['date'].dt.dayofweek

month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

train['month_num'] = train['month'].map(month_map)
test['month_num'] = test['month'].map(month_map)


train.head()


test.head()


# Select only string columns for encoding, excluding 'y' if it's a string (which it shouldn't be)
y = train['y']
# Step 2: Identify ALL categorical columns (object or category dtype) in the *original* train DataFrame
# that you intend to one-hot encode.
# Make sure the problematic column (e.g., 'month' if it has 'aug') is included here.
categorical_cols_to_encode = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
    'poutcome', 'balance_bins', 'pdays_bins'
]


# Step 3: Apply `pd.get_dummies` to the `train` DataFrame, specifying *all* columns to encode.
# This ensures that any string/object columns explicitly listed are converted.
train_encoded = pd.get_dummies(train, columns=categorical_cols_to_encode, drop_first=True) # drop_first helps avoid multicollinearity
test_encoded = pd.get_dummies(test, columns=categorical_cols_to_encode, drop_first=True)
# Step 4: Handle any remaining non-numeric columns that were NOT meant for one-hot encoding.
# This primarily targets columns like 'date' if they are strings or datetime objects.
remaining_non_numeric_after_encoding = train_encoded.select_dtypes(exclude=np.number).columns.tolist()
remaining_test_non_numeric_after_encoding = test_encoded.select_dtypes(exclude=np.number).columns.tolist()
# The 'y' column should already be numeric. We also drop 'y' from this check to avoid accidentally
# processing it if it was numeric but identified by `select_dtypes` due to some edge case.
if 'y' in remaining_non_numeric_after_encoding:
    remaining_non_numeric_after_encoding.remove('y')

if remaining_non_numeric_after_encoding:
    print(f"\nWarning: The following non-numeric columns still exist and will be dropped for correlation calculation: {remaining_non_numeric_after_encoding}, {remaining_test_non_numeric_after_encoding}")
    # Drop these columns as they cannot be directly included in correlation matrix.
    # If 'date' is here and you want to use it, convert it to a numerical representation (e.g., epoch timestamp)
    # or extract features like 'year', 'day_of_year' etc.
    train_encoded = train_encoded.drop(columns=remaining_non_numeric_after_encoding)
    test_encoded = test_encoded.drop(columns=remaining_test_non_numeric_after_encoding)
else:
    print("\nAll non-numeric columns successfully handled (either encoded or removed).")

# Step 5: Ensure 'y' is present and numeric (it should be handled by Step 1).
# This also re-adds 'y' if it was removed for processing by get_dummies (though get_dummies usually passes through non-encoded cols).
if 'y' not in train_encoded.columns:
    train_encoded['y'] = y # Re-add y if it was dropped during a previous step or implicitly.

# Step 6: Compute the correlation matrix
correlation_matrix = train_encoded.corr()

# Step 7: Display the correlation matrix for 'y'
print("\nCorrelation with 'y':")
print(correlation_matrix['y'].sort_values(ascending=False))


import seaborn as sns


corr_with_y = correlation_matrix['y'].sort_values(ascending=False).drop('y')

plt.figure(figsize=(10, len(corr_with_y) * 0.3)) # Adjust figure size dynamically
sns.barplot(x=corr_with_y.values, y=corr_with_y.index, palette='viridis')
plt.title('Correlation of Features with Target Variable "y"')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8) # Add a line at 0 for clarity
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# --- 1. Separate Features (X) and Target (y) ---
# X will be all columns in train_encoded except 'y'
X = train_encoded.drop('y', axis=1)
# y is the target column
y = train_encoded['y']

# --- 2. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y is important for imbalanced datasets, ensuring train/test sets have similar proportions of 'y' classes.

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- 3. Initialize and Train the XGBoost Model ---
# Since 'y' is binary (0 or 1), this is a classification problem.
# Use XGBClassifier for classification tasks.
# 'objective' 'binary:logistic' is for binary classification with probability output.
# 'eval_metric' 'logloss' is a common metric for classification.
# You can tune parameters like n_estimators, learning_rate, max_depth, etc.
model = xgb.XGBClassifier(objective='binary:logistic',
                          eval_metric='logloss',
                          use_label_encoder=False, # Suppress a common warning
                          n_estimators=100,      # Number of boosting rounds
                          learning_rate=0.1,     # Step size shrinkage to prevent overfitting
                          max_depth=5,           # Maximum depth of a tree
                          random_state=42)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate Model Performance (Optional but recommended) ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# --- 5. Extract and Visualize Feature Importance ---
# XGBoost provides feature importance scores. 'gain' is often a good metric.
# 'gain' indicates the average gain across all splits where the feature is used.
feature_importances = model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance_df.head(20)) # Display top 20 features

# Plotting the top N most important features
plt.figure(figsize=(12, 8))
top_n = 20 # You can adjust this number
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette='viridis')
plt.title(f'Top {top_n} Most Important Features (XGBoost)')
plt.xlabel('Feature Importance (Gain)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# --- Alternative Plot: Using XGBoost's built-in plot_importance ---
plt.figure(figsize=(12, 8))
xgb.plot_importance(model, importance_type='gain', max_num_features=top_n, ax=plt.gca(), grid=False)
plt.title(f'Top {top_n} Most Important Features (XGBoost - Built-in Plot)')
plt.xlabel('Feature Importance (Gain)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    subsample=0.6,
    use_label_encoder=False,
    reg_lambda=10,
    reg_alpha=0,
    n_estimators=800,      
    learning_rate=0.1,     
    max_depth=5,
    gamma = 0.3,
    colsample_bytree=0.6,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

test_pred = xgb_model.predict(test_encoded)
test_pred_prob = xgb_model.predict_proba(test_encoded)[:,1]
# submission['y'] = test_pred
submission['y'] = test_pred_prob
# submission = submission.drop(columns='y_prob')
submission.to_csv('submission.csv', index=False)
submission.head()


from sklearn.metrics import roc_curve, auc, roc_auc_score

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'XGBoost (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


test_encoded['y'] = test_pred
train_encoded['y'] = train['y']
new_data = pd.concat([train_encoded, test_encoded])
print(new_data)


y = new_data['y']
X = new_data.drop(columns='y')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    subsample=0.6,
    use_label_encoder=False,
    reg_lambda=10,
    reg_alpha=0,
    n_estimators=800,      
    learning_rate=0.1,     
    max_depth=5,
    gamma = 0.3,
    colsample_bytree=0.6,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
test_encoded = test_encoded.drop(columns='y')
#test_pred = xgb_model.predict(test_encoded)
test_pred_prob = xgb_model.predict_proba(test_encoded)[:,1]
# submission['y'] = test_pred
submission['y'] = test_pred_prob
# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC AUC Score: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'XGBoost (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# submission = submission.drop(columns='y_prob')
submission.to_csv('submission.csv', index=False)
submission.head()




