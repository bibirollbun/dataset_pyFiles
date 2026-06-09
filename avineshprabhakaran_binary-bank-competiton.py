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


# ================================================================
# Import Libraries 
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Plotting library
import seaborn as sns # Statistical plots built on matplotlib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


# ================================================================
# Load Data
# ================================================================
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")



train


test


print('train data shape',train.shape )
print('test data shape',test.shape )


print("train datatype ")
print(train.dtypes)# Data types of each column



print("test datatype ")
print(test.dtypes) # Data types of each column


print("First 5 rows of train")
print( train.head()) # first 5 rows at the data


print("First 5 rows of train")
print( test.head()) # first 5 rows at the data


# Missing values check (column-wise counts)
print("Missing values in train:", train.isnull().sum())
print("Missing values in test:", test.isnull().sum())



#Target distribution (class imbalance insight)
# Count raw numbers
print(train['y'].value_counts())

# Count proportions (percentages)
print(train['y'].value_counts(normalize=True))



# Train
# Bar chart
sns.countplot(x='y', data=train)
plt.title("Target Variable Distribution (y)")
plt.show()

# Pie chart
train['y'].value_counts().plot.pie(autopct='%1.1f%%', labels=['No (0)','Yes (1)'])
plt.title("Subscription Rate")
plt.ylabel("")
plt.show()


# Identify numeric and categorical columns based on the known schema
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"] # numerical features
cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"] # categorical features


#train dataset
# Histograms + KDE to inspect distribution, skewness, and potential outliers
#Loops through each column in your list of numerical features (like age, balance, duration, etc.).
for col in num_cols:
    plt.figure(figsize=(7,4)) # Set figure size
    sns.histplot(train[col], bins=30, kde=True) # Histogram with density curve
    plt.title(f"Distribution of {col}") # Chart title
    plt.xlabel(col) # X axis label
    plt.ylabel("Count") # Y axis label
    plt.tight_layout() # Neat layout
    plt.show() # Render plot


#test dataset
# Histograms + KDE to inspect distribution, skewness, and potential outliers
#Loops through each column in your list of numerical features (like age, balance, duration, etc.).
for col in num_cols:
    plt.figure(figsize=(7,4)) # Set figure size
    sns.histplot(test[col], bins=30, kde=True) # Histogram with density curve
    plt.title(f"Distribution of {col}") # Chart title
    plt.xlabel(col) # X axis label
    plt.ylabel("Count") # Y axis label
    plt.tight_layout() # Neat layout
    plt.show() # Render plot


# train dataset
# Bar charts showing frequency of each category (ordered by count)
for col in cat_cols:
    plt.figure(figsize=(9,4)) # Set figure size
    order = train[col].value_counts().index # Order bars by frequency
    sns.countplot(data=train, x=col, order=order) # Bar plot of category counts
    plt.title(f"Counts of {col}") # Title
    plt.xlabel(col) # X label
    plt.ylabel("Count") # Y label
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.tight_layout() # Neat layout
    plt.show() # Render


# test dataset
# Bar charts showing frequency of each category (ordered by count)
for col in cat_cols:
    plt.figure(figsize=(9,4)) # Set figure size
    order = test[col].value_counts().index # Order bars by frequency
    sns.countplot(data=test, x=col, order=order) # Bar plot of category counts
    plt.title(f"Counts of {col}") # Title
    plt.xlabel(col) # X label
    plt.ylabel("Count") # Y label
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.tight_layout() # Neat layout
    plt.show() # Render


# Boxplots help visualize how numeric distributions differ by class y
for col in num_cols:
    plt.figure(figsize=(7,4)) # Figure size
    sns.boxplot(data=train, x='y', y=col) # Boxplot y~col by class
    plt.title(f"{col} vs Target y") # Title
    plt.xlabel("y (0=No, 1=Yes)") # X label
    plt.ylabel(col) # Y label
    plt.tight_layout() # Neat layout
    plt.show() # Render


# Subscription rate (mean of y) per category for each categorical feature
for col in cat_cols:
# Compute mean target per category (i.e., conversion rate)
    rate = train.groupby(col)['y'].mean().sort_values(ascending=False)
    plt.figure(figsize=(9,4)) # Figure size
    sns.barplot(x=rate.index, y=rate.values) # Bar chart of rates
    plt.title(f"Subscription Rate by {col}") # Title
    plt.xlabel(col) # X label
    plt.ylabel("Mean of y (rate)") # Y label
    plt.xticks(rotation=45, ha='right') # Readable ticks
    plt.tight_layout() # Neat layout
    plt.show() # Render


#train
#  Correlation analysis (numerical features only) ----
# Pearson correlations among numerical features and the target
corr = train[num_cols + ['y']].corr() # Compute correlation matrix
plt.figure(figsize=(10,6)) # Figure size
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0) # Heatmap with values
plt.title("Correlation Heatmap (numeric + target)") # Title
plt.tight_layout() # Neat layout
plt.show() # Render


#test
#  Correlation analysis (numerical features only) ----
# Pearson correlations among numerical features and the target
corr = test[num_cols ].corr() # Compute correlation matrix
plt.figure(figsize=(10,6)) # Figure size
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0) # Heatmap with values
plt.title("Correlation Heatmap (numeric )") # Title
plt.tight_layout() # Neat layout
plt.show() # Render


#train data set 
# Store IQR outlier results in a DataFrame
iqr_outliers = []
for col in num_cols:
    q1, q3 = train[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outliers = ((train[col] < lower) | (train[col] > upper)).sum()
    iqr_outliers.append([col, outliers, lower, upper])

train_outlier_df = pd.DataFrame(iqr_outliers, columns=["Column", "Outlier_Count", "Lower_Bound", "Upper_Bound"])
print(train_outlier_df)  


#test data set
# Store IQR outlier results in a DataFrame
iqr_outliers = []
for col in num_cols:
    q1, q3 = test[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outliers = ((test[col] < lower) | (test[col] > upper)).sum()
    iqr_outliers.append([col, outliers, lower, upper])

test_outlier_df = pd.DataFrame(iqr_outliers, columns=["Column", "Outlier_Count", "Lower_Bound", "Upper_Bound"])
print(test_outlier_df) 


#train data set 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 1. Get categorical columns automatically (object or category dtype)
cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", cat_cols)

# 2. Create encoder
encoder = OneHotEncoder(sparse=False, drop="first")

# 3. Fit + transform categorical data
encoded = encoder.fit_transform(train[cat_cols])

# 4. Get new column names
encoded_cols = encoder.get_feature_names_out(cat_cols)

# 5. Convert to dataframe
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=train.index)

# 6. Drop original categorical columns and join new ones
train_encoded = train.drop(columns=cat_cols).join(encoded_df)

print(train_encoded.head())



train_encoded=train_encoded.drop("id", axis=1)


X=train_encoded


#test data encoding 
# step 2
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# 1. Get categorical columns automatically (object or category dtype)
cat_cols = test.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns:", cat_cols)

# 2. Create encoder
encoder = OneHotEncoder(sparse=False, drop="first")

# 3. Fit + transform categorical data
encoded = encoder.fit_transform(test[cat_cols])

# 4. Get new column names
encoded_cols = encoder.get_feature_names_out(cat_cols)

# 5. Convert to dataframe
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=test.index)

# 6. Drop original categorical columns and join new ones
test_encoded = test.drop(columns=cat_cols).join(encoded_df)

print(test_encoded.head())


test_encoded=test_encoded.drop("id", axis=1)


X1=test_encoded



#X = train.drop(columns=["y"])
#y = train["y"]  # binary 0/1


#train
from sklearn.preprocessing import StandardScaler
# --- Step 1: Scale ALL features (since all are numeric now) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 2: Convert back to DataFrame (keep same column names) ---
X = pd.DataFrame(X_scaled, columns=X.columns)

print(X.head())


#test
from sklearn.preprocessing import StandardScaler
# --- Step 1: Scale ALL features (since all are numeric now) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X1)

# --- Step 2: Convert back to DataFrame (keep same column names) ---
X1 = pd.DataFrame(X_scaled, columns=X1.columns)

print(X1.head())


#train
#: Separate features & target ---
X = train_encoded.drop(columns=["y"])
y = train_encoded["y"]

print(X.shape, y.shape)


from sklearn.model_selection import train_test_split
#Train
# X and y already prepared from your train dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)


#Test
# X and y already prepared from your train dataset
#X_test, X_val, y_test, y_val = train_test_split(
#    X, y, test_size=0.2, random_state=42, stratify=y
#)

#print("Train shape:", X_test.shape, y_test.shape)
#print("Validation shape:", X_val.shape, y_val.shape)


#from sklearn.utils import class_weight

# use your actual target variable 'y'
#weights = class_weight.compute_class_weight(
 #   class_weight='balanced',
  #  classes=np.unique(y),
   # y=y
#)

#print("Class Weights:", dict(zip(np.unique(y), weights)))



from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Use class weights from earlier
#class_weights = [weights[0], weights[1]]

cat_model = CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=8,l2_leaf_reg=5,random_seed=42,eval_metric="AUC",verbose=200)
#class_weights=class_weights,
# Train
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# Predictions (probabilities for AUC)
cat_pred_proba = cat_model.predict_proba(X_val)[:, 1]
cat_pred = cat_model.predict(X_val)

# Metrics
print("ROC AUC Score (CatBoost):", roc_auc_score(y_val, cat_pred_proba))
print("F1 Score (CatBoost):", f1_score(y_val, cat_pred))



y_pred = cat_model.predict(X1)


submission=pd.DataFrame({'id':test['id'], 'y':y_pred})
print(submission)


submission.to_csv('submission_cat.csv',index=False)


y_pred = cat_model.predict(X1)




