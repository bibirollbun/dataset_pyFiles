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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original = pd.read_csv("/kaggle/input/personality-data/personality_dataset.csv")
original_data = pd.read_csv("/kaggle/input/personality-dataset/personality_datasert.csv")


print("train_data :", train_data.shape)
print("test_data :", test_data.shape)
print("original :", original.shape)
print("original_data :", original_data.shape)
print("sample_submission :", sample_submission.shape)


train_data.head()


train_data['Personality'].value_counts()


train_data.info()


# Calculate missing values
missing_values = train_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


original_data.head()


# Calculate missing values
missing_values = original_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


test_data.isna().sum().sort_values(ascending=False)


# Calculate missing values
missing_values = test_data.isnull().mean() * 100

# Plot
missing_values.plot(kind='bar', figsize=(8, 4), color='skyblue')
plt.title('Percentage of Missing Values by Feature')
plt.ylabel('Percentage')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.show()


# Categorical columns to plot
cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']

# Set up 2x2 grid for subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 6))
axes = axes.flatten()  # Flatten to iterate easily

# Generate pie charts
for i, col in enumerate(cat_cols):
    train_data[col].value_counts().plot.pie(
        ax=axes[i],
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        shadow=True
    )
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_ylabel("")  # Remove y-label for cleaner plot

plt.tight_layout()
plt.show()


# Define a helper function
def plot_pie_bar(column_name, title_prefix=""):
    counts = test_data[column_name].value_counts()
    labels = counts.index
    values = counts.values
    colors = plt.cm.tab20.colors[:len(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Pie chart
    axes[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    axes[0].set_title(f"{title_prefix}{column_name} Distribution (Pie Chart)")
    axes[0].axis('equal')
     # Bar chart
    axes[1].bar(labels, values, color=colors)
    axes[1].set_title(f"{title_prefix}{column_name} Distribution (Bar Chart)")
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# Plot for Soil Type
plot_pie_bar('Stage_fear', title_prefix=" ")

# Plot for Crop Type
plot_pie_bar('Drained_after_socializing', title_prefix=" ")


train_data = train_data.drop("id", axis=1)
#train_data = pd.concat([train_data, original], ignore_index=True)
#train_data = pd.concat([train_data, original_data], ignore_index=True)
train_data = train_data.drop_duplicates()
print("shape of the data :",train_data.shape)
train_data.head()


num_cols = list(train_data.select_dtypes(exclude=['object']).columns)
cat_cols = list(train_data.select_dtypes(include=['object']).columns.difference(['Personality']))

num_cols_test = list(test_data.select_dtypes(exclude=['object']).columns)
cat_cols_test = list(test_data.select_dtypes(include=['object']).columns)


# Set up the layout
n_cols = 2
n_rows = (len(num_cols) + 1) // n_cols
plt.figure(figsize=(n_cols * 5, n_rows * 3))

for i, col in enumerate(num_cols):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.kdeplot(train_data[col], label='Train', fill=True, color='blue', linewidth=2)
    sns.kdeplot(test_data[col], label='Test', fill=True, color='orange', linewidth=2)
    plt.title(f'Distribution of {col}')
    plt.legend()

plt.tight_layout()
plt.show()


# Plot distribution for each numeric column
plt.figure(figsize=(15, len(num_cols)*3))

for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols), 1, i)
    sns.histplot(train_data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.tight_layout()

plt.show()


#train_data = train_data.dropna()
train_data.shape


#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
labelencoder = LabelEncoder()
encoder = OrdinalEncoder()
#for col_name in cat_cols:
    #train_data[col_name]=labelencoder.fit_transform(train_data[col_name]).astype(int)
        
#for col_name in cat_cols_test:
    #test_data[col_name]=labelencoder.transform(test_data[col_name]).astype(int)


train_data[cat_cols]=encoder.fit_transform(train_data[cat_cols])
test_data[cat_cols]=encoder.transform(test_data[cat_cols])
target_le = LabelEncoder()
train_data['Personality'] = target_le.fit_transform(train_data['Personality'])


from sklearn.model_selection import train_test_split
X = train_data.drop(['Personality'], axis=1)
y = train_data['Personality']
test = test_data.drop(['id'],axis=1)


#Best Accuracy: 0.9694991299481467
params = {'n_estimators': 16540, 'max_depth': 10, 'learning_rate': 0.1589449383523801, 'subsample': 0.8350862629912426, 'colsample_bytree': 0.9334695012599407, 'gamma': 1.2725909269297682, 'reg_alpha': 0.5961699412092507, 'reg_lambda': 2.365625768379546}


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Define your model
model = XGBClassifier(**params,
    objective = 'binary:logistic',
    eval_metric='logloss',
    random_state=42)

# Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
test_preds = np.zeros((test.shape[0], len(np.unique(y))))  # for prob-based voting

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    accuracy_scores.append(acc)
    #print(f"Fold {fold} Accuracy: {acc:.4f}")

    # Predict on test set (probabilities)
    test_preds += model.predict_proba(test) / skf.n_splits
# Mean Accuracy
print(f"\n✅ Mean Accuracy: {np.mean(accuracy_scores):.4f}")
# Final test predictions as most probable class
final_preds = np.argmax(test_preds, axis=1)

final_preds_labels = target_le.inverse_transform(final_preds)
submission = pd.DataFrame({
    "id": sample_submission["id"],  
    "Personality": final_preds_labels  
})

submission.to_csv("submission_xgb.csv", index=False)
submission.head()


from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# Repeated Stratified K-Fold setup
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

# Initialize arrays
accuracy_scores = []
test_preds = np.zeros((test.shape[0], len(np.unique(y))))  # for class probabilities

# Train across folds and average test predictions
for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y), start=1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        use_label_encoder=False,
        objective = 'binary:logistic',
        subsample = 0.8,
        colsample_bytree = 0.8,
        eval_metric='logloss',
        random_state=fold  # vary seed for robustness
    )

    model.fit(X_train, y_train)
    
    # Evaluate accuracy
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    accuracy_scores.append(acc)
    #print(f"Fold {fold} Accuracy: {acc:.4f}")

    # Predict on test set
    test_preds += model.predict_proba(test) / rskf.get_n_splits()

print(f"\n✅ Mean Accuracy: {np.mean(accuracy_scores):.4f}")
# Final test predictions as most probable class
final_preds = np.argmax(test_preds, axis=1)

final_preds_labels = target_le.inverse_transform(final_preds)
submission = pd.DataFrame({
    "id": sample_submission["id"],  
    "Personality": final_preds_labels  
})

submission.to_csv("submissionxgb.csv", index=False)
submission.head()


#!pip install imbalanced-learn
#!pip install -U scikit-learn==1.2.2 imbalanced-learn==0.10.1 --quiet


# After training your model
importances = model.feature_importances_

# Create a DataFrame for better plotting
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

