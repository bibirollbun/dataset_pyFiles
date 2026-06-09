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

from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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


# Calculate missing values
missing_values = original.isnull().mean() * 100

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


# Fill missing values
train_data[train_data.select_dtypes(include=['number']).columns] = train_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
train_data[train_data.select_dtypes(include=['object', 'category']).columns] = train_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))

# Fill missing values
test_data[test_data.select_dtypes(include=['number']).columns] = test_data.select_dtypes(include=['number']).apply(lambda x: x.fillna(x.median()))
test_data[test_data.select_dtypes(include=['object', 'category']).columns] = test_data.select_dtypes(include=['object', 'category']).apply(lambda x: x.fillna("missing"))


#  object datatype columns encoding:
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col_name in cat_cols:
    train_data[col_name]=labelencoder.fit_transform(train_data[col_name]).astype(int)
        
#for col_name in cat_cols_test:
    test_data[col_name]=labelencoder.transform(test_data[col_name]).astype(int)

target_le = LabelEncoder()
train_data['Personality'] = target_le.fit_transform(train_data['Personality'])


# Social Engagement Score (interaction term)
train_data['Social_score'] = (train_data['Social_event_attendance'] + train_data['Going_outside'] + train_data['Friends_circle_size'])
# Introvert-Tendency Proxy
train_data['Introvert_score'] = (train_data['Time_spent_Alone'] * train_data['Drained_after_socializing'])

# Social Engagement Score (interaction term)
test_data['Social_score'] = (test_data['Social_event_attendance'] + test_data['Going_outside'] + test_data['Friends_circle_size'])
# Introvert-Tendency Proxy
test_data['Introvert_score'] = (test_data['Time_spent_Alone'] * test_data['Drained_after_socializing'])



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])


# Get correlation matrix (default is Pearson correlation)
correlation_matrix = train_data.corr()
# Display
#print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


from sklearn.model_selection import train_test_split
X = train_data.drop(['Personality'], axis=1)
y = train_data['Personality']
test = test_data.drop(['id'],axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='error', random_state=42)),
    ('lgb', LGBMClassifier(verbosity=-1,random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svc', SVC(probability=True, random_state=42))
]

# Meta model
meta_model = LogisticRegression(max_iter=1000)

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=True  # optional: gives meta model access to original features
)

# Train
stacking_clf.fit(X_train, y_train)

# Evaluate
val_preds = stacking_clf.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"✅ Stacked Metamodel Accuracy: {acc:.4f}")


# ✅ Predict on test set
test_preds = stacking_clf.predict(test)

final_preds_labels = target_le.inverse_transform(test_preds)
submission = pd.DataFrame({
    "id": sample_submission["id"],  
    "Personality": final_preds_labels  
})

submission.to_csv("submission.csv", index=False)
submission.head()



# Count each class
class_counts = submission['Personality'].value_counts()

# Plot
plt.figure(figsize=(5, 3))
plt.pie(
    class_counts.values,
    labels=class_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Paired.colors
)
plt.title("Distribution of Predicted Personality Classes")
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.show()


