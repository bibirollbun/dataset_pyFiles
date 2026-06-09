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


train_df = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv")


train_df.head()


train_df["loan_status"].value_counts()


train_df.head()


((train_df.isnull().sum())/len(train_df) )*100


train_df['loan_status'].value_counts()


train_df.shape


X = train_df.drop("loan_status", axis=1)
y = train_df['loan_status'].copy()


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=2)


from sklearn.preprocessing import RobustScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.compose import make_column_selector as mcs


num_trf = Pipeline([('impute', SimpleImputer()),
                   ('scl', RobustScaler())])

cat_trf = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown="ignore"))])

ct = ColumnTransformer([
    ('num', num_trf, mcs(dtype_exclude='object')),
    ("cat", cat_trf, mcs(dtype_include='object'))
])


ct.fit(X_train, y_train)

X_train_trf = ct.transform(X_train)
X_test_trf = ct.transform(X_test)


ct


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
    "Natural language processing is a part of AI",
    "AI and machine learning are revolutionizing the world"
]

df = pd.DataFrame({"Text": documents})
df.head()


vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

tfidf_df.head()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.DataFrame({
    "age": [25, 30, 35, 40],
    "salary": [50000, 60000, 70000, 80000],
    "gender": ["Male", "Female", "Male", "Female"],
    "review": ["Great product", "Not bad", "Could be better", "Amazing experience"]
})

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'salary']),
    ('cat', OneHotEncoder(sparse_output=False), ['gender']),
    ('text', TfidfVectorizer(), 'review')  
]) 

pipeline = Pipeline([
    ('preprocessor', preprocessor)
    
])

transformed_data = pipeline.named_steps['preprocessor'].fit_transform(data)
print(transformed_data)



import numpy as np
import pandas as pd
from scipy.stats import zscore

data = pd.DataFrame({
    'A': [10, 12, 14, 15, 100, 18, 20, 22, 24, 200],  
    'B': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
})

z_scores = np.abs(zscore(data))

threshold = 2

data_no_outliers = data[(z_scores < threshold).all(axis=1)]

print("Original Data:\n", data)
print("\nData after Removing Outliers using Z-Score:\n", data_no_outliers)



import numpy as np
import pandas as pd

data = pd.DataFrame({
    'A': [10, 12, 14, 15, 100, 18, 20, 22, 24, 200],  
    'B': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
})

# Q1 (25th percentile) and Q3 (75th percentile)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

print("Original Data:\n", data)
print("\nData after Removing Outliers using IQR:\n", data_no_outliers)



from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_trf, y_train)


y_res.value_counts()


import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


sns.countplot(x=y_train, ax=axes[0])
axes[0].set_title("Before SMOTE")
axes[0].set_xlabel("Class Label")
axes[0].set_ylabel("Count")


sns.countplot(x=y_res, ax=axes[1])
axes[1].set_title("After SMOTE")
axes[1].set_xlabel("Class Label")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


from sklearn.ensemble import IsolationForest
isf = IsolationForest(random_state=2)
isf.fit(X_train_trf) #notice, here we are passing only the feature, not the label

mask = isf.predict(X_train_trf)
print(mask)


import numpy as np

mask_bool = mask == 1  

X_train_clean = X_train_trf[mask_bool]  
y_train_clean = y_train[mask_bool]  

print(f"Before filtering: {X_train_trf.shape}, {y_train.shape}")
print(f"After filtering: {X_train_clean.shape}, {y_train_clean.shape}")




