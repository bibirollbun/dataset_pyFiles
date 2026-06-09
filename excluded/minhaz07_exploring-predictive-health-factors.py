# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')


df.head()


df.shape


df.isnull().sum()


# removing the unnecessery columns
df.drop(columns=['ID'],inplace=True)


# Handling the age column
df['Age'].unique()


df[['a', 'b']] = df['Age'].str.extract(r'(\d+)-?(\d+)?')

# Convert 'a' and 'b' to numeric, setting errors='coerce' to handle non-numeric cases
df[['a', 'b']] = df[['a', 'b']].apply(pd.to_numeric, errors='coerce')

# Compute row-wise mean
df['Mean_Age'] = df[['a', 'b']].mean(axis=1)


# removing the age,a and b column
df = df.drop(columns=['Age','a','b'])


df.shape


df.head()


df['Weight_kg'] = df['Weight_kg'].fillna(df['Weight_kg'].mean())


for col in df.columns:
    print(col,':',len(df[col].unique()))


from sklearn.impute import SimpleImputer
impute = SimpleImputer(strategy='most_frequent')
arr = impute.fit_transform(df)
df = pd.DataFrame(arr,columns=df.columns)


df.isnull().sum()


df['PCOS'].value_counts()


mapping = {
    'No, Yes, not diagnosed by a doctor':0,
    'Yes Significantly':1,
    'Yes':1,
    'No':0
}
df['Hormonal_Imbalance'] = df['Hormonal_Imbalance'].replace(mapping)


df['Hormonal_Imbalance'].value_counts()


df.head()


cat = df[['Hyperandrogenism','Hirsutism', 'Conception_Difficulty', 'Insulin_Resistance']]


for col in cat.columns:
    plt.figure(figsize=(8, 4))  # Set figure size for better visibility
    cat[col].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Value Counts of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)  # Rotate labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid for better readability
    plt.show() 


df[['Hirsutism','Conception_Difficulty','Insulin_Resistance']].value_counts()


# Correcting the dictionary
custom_mapping = {
    'PCOS':{'No':0,'Yes':1},
    'Hyperandrogenism':{'No':0,'Yes':1},
    'Hirsutism': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 1},
    'Conception_Difficulty': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 0, 'Yes, diagnosed by a doctor': 1},
    'Insulin_Resistance': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 0}
}

# Apply mapping only to relevant columns
for col, mapping in custom_mapping.items():
    if col in df.columns:  # Ensure the column exists in the DataFrame
        df[col] = df[col].map(mapping)


df.head()


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Exercise_Frequency'] = label_encoder.fit_transform(df['Exercise_Frequency'])
df['Sleep_Hours'] = label_encoder.fit_transform(df['Sleep_Hours'])
df['Exercise_Duration'] = label_encoder.fit_transform(df['Exercise_Duration'])


df.head(2)


ohe_df = pd.get_dummies(df[['Exercise_Type','Exercise_Benefit']],dtype = np.int8,drop_first=True)


df.drop(columns=['Exercise_Type','Exercise_Benefit'],inplace=True)


df = pd.concat([df,ohe_df],axis=1)


df.head()


x = df.drop(columns=['PCOS'])
y = df['PCOS']


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

# Sample data preparation (replace with your actual dataset)
# Assuming df has feature columns and 'target' is the label column

# df = pd.read_csv('your_dataset.csv')  # Example of how to load your dataset
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scaling features (important for models like SVM and KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of classifiers to test
classifiers = [
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Support Vector Classifier', SVC()),
    ('Naive Bayes', GaussianNB()),
    ('Dummy Classifier', DummyClassifier(strategy='most_frequent'))  # baseline classifier
]

# Store results
results = []

# Run classifiers and track accuracy
for name, clf in tqdm(classifiers, desc='Running Models'):
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store the results
    results.append({
        'Model': name,
        'Accuracy': accuracy
    })

# Create a DataFrame to show results
results_df = pd.DataFrame(results)
print(results_df)



results_df.sort_values(by='Accuracy',ascending=False)































