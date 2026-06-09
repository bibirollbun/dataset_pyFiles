# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df_train


# from google.colab import drive
# drive.mount('/content/drive')


## use basic techniques to analyze the data
print(df_train.shape)
print(df_train.describe())


print(df_train.dtypes)


## turn categorical data to numerical data (one hot encoding)

## listed all categories below
cat_cols = ['default', 'marital', 'education', 'job', 'loan', 'housing', 'contact', 'month', 'poutcome']

df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=False)
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=False)

# Convert boolean columns to integers (True to 1 and False to 0)
bool_cols_train = df_train.select_dtypes(include='bool').columns
df_train[bool_cols_train] = df_train[bool_cols_train].astype(int)
bool_cols_test = df_test.select_dtypes(include='bool').columns
df_test[bool_cols_test] = df_test[bool_cols_test].astype(int)


df_train.describe()


## correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns
orig_cols = ['age','balance','day','duration','campaign','pdays','previous', 'y']
corr = df_train[orig_cols].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()


# ## can drop features id (just an identifier) and other weak features
df_train.drop(columns=['age', 'day', 'campaign', 'pdays'], inplace=True)
df_test.drop(columns=['age', 'day', 'campaign', 'pdays'], inplace=True)

df_train



# replacing outliers
import scipy.stats as stats
extreme_value_cols = ['balance', 'duration', 'previous']
for col in extreme_value_cols:
    df_train[col + '_winsorized'] = stats.mstats.winsorize(df_train[col], limits=[0.01, 0.01])
    df_train.drop([col], axis=1, inplace=True)
    df_test[col + '_winsorized'] = stats.mstats.winsorize(df_test[col], limits=[0.01, 0.01])
    df_test.drop([col], axis=1, inplace=True)
df_train


nan_count = np.sum(df_train.isnull(), axis=0)
print(nan_count) ## for small missing counts, we can just impute with mean but otherwise, we should drop feature?
## no need to handle missing values


main_cols = ['balance_winsorized', 'duration_winsorized', 'previous_winsorized']
df_train[main_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# use min max scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
for min_max_col in main_cols:
    df_train[min_max_col] = min_max_scaler.fit_transform(df_train[[min_max_col]])
    df_test[min_max_col] = min_max_scaler.fit_transform(df_test[[min_max_col]])


df_train ## now its all cleaned and preprocessed
df_test


df_test.columns


X = df_train.drop('y', axis = 1)
y = df_train['y']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1234, test_size = 0.2)


print(y_train.shape)


#from imblearn.over_sampling import SMOTE
class_counts = y_train.value_counts()
max_count = class_counts.max()
resampled_indices =[]
for cls in class_counts.index:
    cls_indices = y_train[y_train == cls].index
    # Randomly sample with replacement
    oversampled_cls_indices = np.random.choice(cls_indices, size=max_count, replace=True)
    resampled_indices.extend(oversampled_cls_indices)
np.random.shuffle(resampled_indices)
X_train_resampled = X_train.loc[resampled_indices]
y_train_resampled = y_train.loc[resampled_indices]
    
# because of class imbalance
    

# this initializes the resampler
#smote = SMOTE(random_state=42)

# resample just train st
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#print("Before:", y_train.value_counts())
#print("After:", y_train_resampled.value_counts())


## train logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

## fine tune parameters for logistic regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Example parameters to tune
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

model = grid_search.best_estimator_
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)


## evaluate model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc_score = accuracy_score(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc_score)
print("\nClassification Report:\n", class_rep)
print("\nConfusion Matrix:\n", cm)



from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()


## random forest is an ensemble model



from sklearn.ensemble import RandomForestClassifier
# # Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42,n_estimators = 100, max_depth = 10)

# Train the model
rf_model.fit(X_train_resampled, y_train_resampled)

# # Make predictions
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)


# Evaluate the Random Forest model
acc_score_rf = accuracy_score(y_test, y_pred_rf)
class_rep_rf = classification_report(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("Random Forest Accuracy:", acc_score_rf)
print("\nRandom Forest Classification Report:\n", class_rep_rf)
print("\nRandom Forest Confusion Matrix:\n", cm_rf)

disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()


df_test.shape

# df_train.shape

# y_proba.shape



import pandas as pd

y_proba_test = model.predict_proba(df_test)

# Example: if your test set has an 'id' column
# and you already calculated prediction probabilities in y_pred_proba

samp_sub = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')

submission = pd.DataFrame({
    "id": samp_sub["id"],   # replace 'test' with your test dataframe name
    "y": y_proba_test[:, 1]   # replace with your prediction probabilities array
})

# Save to CSV in the format the competition wants
submission.to_csv("submission.csv", index=False)

print("✅ Submission file created: submission.csv")


