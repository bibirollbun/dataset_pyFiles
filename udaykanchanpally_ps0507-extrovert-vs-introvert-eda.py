# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


original  = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
data = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")


val = original.shape[0]
end= data.shape[0] +val

ids = [i for i in range(val, end)]
data["id"] = ids


train = total_train = pd.concat([original, data], ignore_index=True)


train 


train.info()


train.isnull().sum()




# Copy the dataset


# Calculate missing values per column
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

# Display missing values in each column
print("ğŸ”� Missing Values per Column:\n")
print(missing)

# Plot missing values per column
plt.figure(figsize=(10, 5))
sns.barplot(x=missing.values, y=missing.index, palette="flare")
plt.title("Missing Values in TRAIN_DATA")
plt.xlabel("Missing Count")
plt.ylabel("Missing Column Names")
plt.tight_layout()
plt.show()




# Copy the dataset


# Calculate missing values per column
missing = test.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

# Display missing values in each column
print("ğŸ”� Missing Values per Column:\n")
print(missing)

# Plot missing values per column
plt.figure(figsize=(10, 5))
sns.barplot(x=missing.values, y=missing.index, palette="flare")
plt.title("Missing Values in TEST_DATA")
plt.xlabel("Missing Count")
plt.ylabel("Missing Column Names")
plt.tight_layout()
plt.show()



from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(strategy='mean')
imputer_mode = SimpleImputer(strategy='most_frequent')



train_numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
train_categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
test_categorical_cols = test.select_dtypes(include=['object', 'category']).columns.tolist()
test_numerical_cols = test.select_dtypes(include=['int64', 'float64']).columns.tolist()


train_numerical_cols


for i in train_numerical_cols:
    train[i] = imputer_mean.fit_transform(train[[i]])
   
for i in train_categorical_cols:
    train[i] = imputer_mode.fit_transform(train[[i]]).ravel()


for i in test_numerical_cols:
    test[i] = imputer_mean.fit_transform(test[[i]])
for i in test_categorical_cols:
    test[i] = imputer_mode.fit_transform(test[[i]]).ravel()



train.isnull().sum()


train.info()


test.info()


train.describe()


n = len(test_categorical_cols)
print("Number of categorical columns in test data:", n)
for i in test_categorical_cols:
    print(i, "-->", test[i].unique())

print("////////////////////////////////////////////////////")


n = len(train_categorical_cols)
print("Number of categorical columns in train data:", n)
for i in train_categorical_cols:
    print(i, "-->", train[i].unique())  


categorical_vals = {
    "Yes":1,"No":0,
    "Extrovert":1,"Introvert":0 
           }


for i in train_categorical_cols:
    if i != 'Personality':
        train[i] = train[i].map({"Yes":1,"No":0})
    else:
        train[i] = train[i].map({"Extrovert":1,"Introvert":0})
        
for i in test_categorical_cols:
    test[i] = test[i].map({"Yes":1,"No":0})


train = train.round().astype(int)
test = test.round().astype(int)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train)
scaler.fit(test)



import math

# Set Seaborn style and palette
sns.set(style="darkgrid")
sns.set_palette("muted")

# Select numerical columns except ID and target
train_numerical_cols1 = [col for col in train.select_dtypes(include='number').columns if col != 'id']

# Grid size (e.g., 2 columns per row)
n_cols = 2
n_rows = math.ceil(len(train_numerical_cols1) / n_cols)

# Set up figure
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten()

# Plot KDEs
for i, col in enumerate(train_numerical_cols1):
    sns.kdeplot(data=train, x=col, hue='Personality', fill=True, ax=axes[i])
    axes[i].set_title(f" {col} by Personality", fontsize=12)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



train



test


corr = train.select_dtypes(include='number').corr()

# Set figure size and style
plt.figure(figsize=(10, 8))
sns.set(style="darkgrid")

# Create heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)

plt.title('Correlation Heatmap')
plt.show()


test



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],               # Regularization strength
    'penalty': ['l2'],                          # 'l1' requires 'liblinear' or 'saga'
    'solver': ['lbfgs', 'saga'],                # Solvers that support 'l2'
    'class_weight': [None, 'balanced'],         # Try with and without class balancing
    'max_iter': [100, 500, 1000]                # In case solver needs more time
}



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',           # You can also try 'f1', 'roc_auc', etc.
    verbose=1,
    n_jobs=-1                     # Use all CPU cores
)



X = train.drop("Personality", axis=1)  # Features
y = train["Personality"]               # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,            # 20% test, 80% train
    random_state=42,          # For reproducibility
    stratify=y                # Maintain class distribution
)


grid_search.fit(X, y)



"""best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))"""


"""
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()"""








# Predict on test data
y_pred = best_model.predict(test)

# Map numerical prediction to labels
categorical_vals = {
    1: "Extrovert",
    0: "Introvert"
}
pred_labels = pd.Series(y_pred).map(categorical_vals)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': pred_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Successfully model predicted")












