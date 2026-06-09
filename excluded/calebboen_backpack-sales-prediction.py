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


# import libraries

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

from lightgbm import LGBMRegressor

import warnings
warnings.filterwarnings('ignore')


# load the datasets

train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv").drop(["id"], axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv").drop(["id"], axis=1)

train.head()


train.shape, test.shape, train_extra.shape


# merge the two training sets

train = pd.concat([train, train_extra], ignore_index=True)
train.shape


def analyze_df(df):
    print("DataFrame Information:")
    print("----------------------")
    display(df.info(verbose=True, show_counts=True))
    print("\n")

    print("Number of Null Values:")
    print("----------------------")
    display(df.isnull().sum())
    print("\n")

    print("Number of Duplicated Rows:")
    print("--------------------------")
    display(df.duplicated().sum())
    print("\n")

    print("Number of Unique Values:")
    print("------------------------")
    display(df.nunique())
    print("\n")

analyze_df(train)


# let's examine the price column

# first we plot a histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(x = train["Price"], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Price Distribution")
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Frequency")

# then we use a boxplot to see if there any outliers
sns.boxplot(x = train["Price"], ax=axes[1], color="lightcoral")
axes[1].set_title("Boxplot of Price")
axes[1].set_xlabel("Price")

plt.tight_layout()
plt.show()


# check the price summary
price_summary = train["Price"].reset_index().describe().round(4).style.format(precision=2).background_gradient(cmap="Blues")
display(price_summary)


# let's split our variables into categorical and numeric for easier EDA using the details from
# train.info

cat_cols = [
    "Brand",
    "Material",
    "Size",
    "Laptop Compartment",
    "Waterproof",
    "Style",
    "Color",
]

num_cols = ["Compartments", "Weight Capacity (kg)", "Price"]


# Visualizing Numeric Columns

plt.figure(figsize=(14, len(num_cols) * 3))
for i , col in enumerate(num_cols):
    plt.subplot(len(num_cols)//2+1,2,i+1)
    sns.histplot(x=col, data=train, bins=30, kde=True, palette="pastel")
    plt.title(col)
    plt.tight_layout()


# check for outliers in the numeric columns

plt.figure(figsize=(10, 6))
for i , col in enumerate(num_cols,1):
    plt.subplot(2,2,i)
    sns.boxplot(x=col, y = "Brand", data=train, palette="Dark2")
    plt.title(col)
    plt.tight_layout()


# Visualiziing categorical columns

plt.figure(figsize=(10,8))
for i ,col in enumerate(cat_cols ,1):
    plt.subplot(3,3,i)
    sns.countplot(y=col, data=train)
    plt.title(col)
    
plt.tight_layout()
plt.show()


# create a weight capacity pipeline

weight_capacity_pipe=Pipeline(steps=[('scaler',StandardScaler())])
preprocessor = ColumnTransformer(
    transformers=[
        ('weight_capacity_pipe', weight_capacity_pipe, ['Weight Capacity (kg)']),
        ('cat_pipeline', Pipeline(steps=[
            ('encoder', OneHotEncoder())
        ]), cat_cols)
    ],
    remainder='passthrough'
)


# split data into features and targe

X_train,X_test,y_train,y_test= train_test_split(train.drop(columns='Price'),train['Price'],test_size=0.2)


# finnish creating a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor())
])

# Perform cross-validation
rmse_scorer = make_scorer(mean_squared_error, squared=False)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=rmse_scorer)

print(f'Cross-Validation RMSE: {np.mean(scores):.4f}')

# Fit the pipeline on the training set
pipeline.fit(X_train, y_train)

# Test set predictions
y_pred = pipeline.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test Set RMSE: {test_rmse:.4f}')


# Make predictions on the test set
predictions = pipeline.predict(test)

# Create the submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],         # Ensure 'id' exists in the test set
    'Price': predictions      # Use predictions on the test set
})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

