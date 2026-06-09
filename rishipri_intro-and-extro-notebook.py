# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e7'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings('ignore')
import os
os.environ["LGBM_LOGLEVEL"] = "error"


import pandas as pd
df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df.shape


df.info()


df.isnull().sum()


# Numerical: fill missing with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill missing with mode
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values filled using fillna().")




import matplotlib.pyplot as plt
import seaborn as sns


num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure(figsize=(6, 3))
    sns.countplot(data=df, x=col)
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()



for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=df, x='Personality', y=col)
    plt.title(f'{col} distribution by Personality')
    plt.show()



# Encode categorical if needed for correlation
df_corr = df.copy()
df_corr[cat_cols] = df_corr[cat_cols].astype('category').apply(lambda x: x.cat.codes)

plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()



from scipy.stats import skew

for col in num_cols:
    print(f"{col} skewness: {skew(df[col].dropna()):.2f}")



df['Personality'].value_counts(normalize=True)



# One-Hot Encoding for nominal categories
df_encoded = pd.get_dummies(df, drop_first=True)



X = df_encoded.drop('Personality_Introvert', axis=1)  # or 'Personality' if not one-hot encoded
y = df_encoded['Personality_Introvert']




from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_val_scaled)

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))



model.fit(X_train_scaled, y_train)



# Load test data
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

# Preprocess the same way:
test_encoded = pd.get_dummies(test)

# Align columns with training set
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

# Scale
X_test_scaled = scaler.transform(test_encoded)




# 1. Load test data
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# 2. Fill missing values (same logic as training set)
# Fill categorical with mode
for col in test.select_dtypes(include='object').columns:
    test[col] = test[col].fillna(test[col].mode()[0])

# Fill numerical with median
for col in test.select_dtypes(include=['int64', 'float64']).columns:
    test[col] = test[col].fillna(test[col].median())

# 3. Encode same as training
test_encoded = pd.get_dummies(test)

# 4. Align with training features (add missing columns with 0s)
test_encoded = test_encoded.reindex(columns=X.columns, fill_value=0)

# 5. Scale using training scaler
X_test_scaled = scaler.transform(test_encoded)

# 6. Predict
test_preds = model.predict(X_test_scaled)



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume df is your cleaned and encoded training dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)



y_pred = model.predict(X_val_scaled)



acc = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", acc)



from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))



submission = pd.DataFrame({
    "id": test["id"],  
    "Personality": np.where(test_preds == 1, "Introvert", "Extrovert")  
})


submission.to_csv("submission.csv", index=False)

print("✅ Submission file 'submission.csv' created.")
submission.head()





