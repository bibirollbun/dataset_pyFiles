import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd

# Replace with your actual file path
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Show the first few rows
df.head(10)


df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')


# Shape of the dataset
print("Shape of dataset:", df.shape)

# Column names
print("Column names:", df.columns.tolist())

# Data types and non-null counts
df.info()

# Statistical summary of numerical columns
df.describe()

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check for duplicate rows
print("Duplicate rows:", df.duplicated().sum())


# Check unique values in the target column
print(df['satisfaction'].value_counts())

# Visualize it
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data=df, x='satisfaction', palette='Set2')
plt.title('Target Variable Distribution')
plt.show()


categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']  # example columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='satisfaction')
    plt.title(f'{col} vs Satisfaction')
    plt.xticks(rotation=45)
    plt.show()


from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
le = LabelEncoder()

# Loop through columns and encode if dtype is 'object' (string)
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Now all columns are numeric — safe to use .corr()
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 10))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']  # example columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='satisfaction')
    plt.title(f'{col} vs Satisfaction')
    plt.xticks(rotation=45)
    plt.show()


numerical_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes']  # example columns

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, kde=True, hue='satisfaction')
    plt.title(f'{col} Distribution by Satisfaction')
    plt.show()


print("Shape of dataset:", df.shape)
print("Column Names:\n", df.columns.tolist())
df.describe(include='all').T  # Shows stats for both numeric and categorical


import missingno as msno

msno.matrix(df)
plt.title("Missing Data Visualization")
plt.show()

# Count missing values
df.isnull().sum().sort_values(ascending=False)


df['satisfaction'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, 
                                           colors=['lightblue', 'lightgreen'], 
                                           explode=[0, 0.1])
plt.ylabel('')
plt.title("Satisfaction Distribution")
plt.show()


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['id', 'Unnamed: 0']]

df[numerical_cols].hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='satisfaction', y=col)
    plt.title(f'{col} vs Satisfaction')
    plt.show()


categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('satisfaction')  # already used

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='satisfaction')
    plt.title(f'{col} vs Satisfaction')
    plt.xticks(rotation=45)
    plt.show()


# Only for small number of features (slow if too many)
sns.pairplot(df_encoded[['Age', 'Flight Distance', 'Inflight wifi service', 'satisfaction']], hue='satisfaction')
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Gender', hue='Customer Type', palette='Set2')
plt.title("Gender vs Customer Type")
plt.ylabel("Number of Customers")
plt.show()


# Group by gender and customer type
gender_ct = df.groupby(['Gender', 'Customer Type']).size().unstack()

# Pie chart for Male
gender_ct.loc['Male'].plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
plt.title("Male: Loyal vs Disloyal")
plt.ylabel('')
plt.show()

# Pie chart for Female
gender_ct.loc['Female'].plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
plt.title("Female: Loyal vs Disloyal")
plt.ylabel('')
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Class', hue='Customer Type', palette='Set2')
plt.title("Customer Type by Travel Class")
plt.ylabel("Number of Customers")
plt.show()


class_ct = df.groupby(['Class', 'Customer Type']).size().unstack()

for c in class_ct.index:
    class_ct.loc[c].plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'{c} Class: Loyal vs Disloyal')
    plt.ylabel('')
    plt.show()


print("Loyal/Disloyal % by Class:\n")
print(df.groupby('Class')['Customer Type'].value_counts(normalize=True).mul(100).round(2))


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Type of Travel', hue='Customer Type', palette='Set1')
plt.title("Customer Type by Type of Travel")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)
plt.show()


travel_ct = df.groupby(['Type of Travel', 'Customer Type']).size().unstack()

for t in travel_ct.index:
    travel_ct.loc[t].plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'{t}: Loyal vs Disloyal')
    plt.ylabel('')
    plt.show()


print("Loyal/Disloyal % by Type of Travel:\n")
print(df.groupby('Type of Travel')['Customer Type'].value_counts(normalize=True).mul(100).round(2))


bins = [0, 18, 30, 40, 50, 60, 70, 100]
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '70+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)



delay_by_age = df.groupby('Age Group')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean().reset_index()



plt.figure(figsize=(10, 6))
sns.barplot(data=delay_by_age, x='Age Group', y='Departure Delay in Minutes', color='skyblue', label='Departure')
sns.barplot(data=delay_by_age, x='Age Group', y='Arrival Delay in Minutes', color='lightgreen', alpha=0.6, label='Arrival')
plt.title("Average Flight Delay (in Minutes) by Age Group")
plt.ylabel("Average Delay (minutes)")
plt.xlabel("Age Group")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Age Group', y='Departure Delay in Minutes')
plt.title("Boxplot of Departure Delay by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Departure Delay (minutes)")
plt.show()


print("Correlation between Age and Delay:")
print(df[['Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']].corr())

