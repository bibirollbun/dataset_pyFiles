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


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns", None)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train


test


submission


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


train.describe()


cat_cols = []
for col in train.columns:
    if train[col].dtypes == "object":
        cat_cols.append(col)

for i in cat_cols:
    print(train[i].value_counts())
    print("-"*30)


num_cols = []
for col in train.columns:
    if train[col].dtypes == "float64":
        num_cols.append(col)


plt.figure(figsize = (15, 15))
for i, col in enumerate(train.columns[1:], 1):
    plt.subplot(3, 3, i)
    sns.histplot(x = train[col])
    plt.title(f"Histogram of {col} Data")
    plt.tight_layout()
    plt.plot()


for i in num_cols:
    print(train[i].value_counts())
    print("-"*30)


df1 = train[train.isna().any(axis = 1)]
df2 = train[~train.isna().any(axis = 1)]


df1_filled = df1.copy()

# Separate with two groups(numerical/categorical)
numerical_cols = train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\n Numerical Ffeatures: {numerical_cols}")
print(f"Categorical features: {categorical_cols}")

# Fill NaN on numerical columns
for col in numerical_cols:
    if df1_filled[col].isnull().any():
        fill_value = df2[col].mean()
        df1_filled[col] = df1_filled[col].fillna(fill_value)

# Fill NaN on categorical columns
for col in categorical_cols:
    if df1_filled[col].isnull().any():
        fill_value = df2[col].mode()[0]
        df1_filled[col] = df1_filled[col].fillna(fill_value)

df1_filled.isnull().sum()


for col in ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]:
    df1_filled[col] = df1_filled[col].round()
    df1_filled[col] = df1_filled[col].astype("int")
df1_filled


# Add two groups on new_train
new_train = pd.concat([df2, df1_filled])
new_train


plt.figure(figsize = (15, 15))
for i, col in enumerate(new_train.columns[1:], 1):
    plt.subplot(3, 3, i)
    sns.histplot(x = new_train[col])
    plt.title(f"Histogram of {col} Data")
    plt.tight_layout()
    plt.plot()


train_temp = new_train.copy()
le = LabelEncoder()
for col in cat_cols:
    new_train[col] = le.fit_transform(new_train[col])

train_corr = new_train.corr()
plt.figure(figsize = (7, 7))
plt.title("Heatmap of New Train")
sns.heatmap(train_corr, fmt = ".3f", annot = True, cmap = "RdPu")
plt.show()


new_train


t1 = new_train.pivot_table(index = "Personality", columns = "Time_spent_Alone", values = "Going_outside", aggfunc = lambda x : len(x)/sum(x)).unstack().reset_index()
t1 = t1.rename(columns = {0 : "Going_outside_ratio"})
t1


t2 = new_train.pivot_table(index = "Drained_after_socializing", columns = "Friends_circle_size", values = "Social_event_attendance", aggfunc = lambda x : len(x)/sum(x)).unstack().reset_index()
t2 = t2.rename(columns = {0 : "Social_event_attendance_ratio"})
t2


t3 = new_train.pivot_table(index = "Stage_fear", columns = "Personality", values = "Post_frequency", aggfunc = lambda x : len(x)/sum(x)).unstack().reset_index()
t3 = t3.rename(columns = {0 : "Post_frequency_ratio"})
t3


new_train_2 = pd.merge(new_train, t1, on = ["Personality", "Time_spent_Alone"], how = "left")
new_train_2 = pd.merge(new_train_2, t2, on = ["Drained_after_socializing", "Friends_circle_size"], how = "left")
new_train_2 = pd.merge(new_train_2, t3, on = ["Stage_fear", "Personality"], how = "left")
new_train_2


new_cat_cols = ["Stage_fear", "Drained_after_socializing"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(new_train_2[new_cat_cols])
encoded_columns = encoder.get_feature_names_out(new_cat_cols)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
new_train_3 = pd.concat([new_train_2, encoded_df], axis=1)
new_train_3 = new_train_3.drop(new_cat_cols, axis=1)
new_train_3


train_temp = new_train_3.copy()
new_train_3.drop("Personality", axis = 1, inplace = True)
new_train_3 = pd.concat([new_train_3, train_temp["Personality"]], axis = 1)
new_train_3


X = new_train_3.iloc[:,1:-1]
y = new_train_3.iloc[:, -1]
skf = StratifiedKFold(n_splits = 20, shuffle = True, random_state = 42)
fold_accuracies = []
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n--- {fold+1}th Fold Start ---")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(f"Train set class distribution : {np.bincount(y_train)}")
    print(f"Test set class distribution : {np.bincount(y_test)}")
    dtc = DecisionTreeClassifier(random_state = 42)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    print(f" {fold+1}th Fold accuracy : {accuracy:.4f}")

print("\n--- All Fold finished ---")
print("Each Fold accuracy : {fold_accuracies}")
print(f"Average accuracy :{np.mean(fold_accuracies):.4f}")

