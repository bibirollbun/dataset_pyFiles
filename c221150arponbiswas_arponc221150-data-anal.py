import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")


# Corrected file paths
train_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/train_dataset.csv")
test_df = pd.read_csv("/kaggle/input/machine-learning-and-data-mining-lab-exam-7-dm/test_dataset_exam.csv")

# Show basic structure
print("Train Dataset Shape:", train_df.shape)
print("Test Dataset Shape:", test_df.shape)

train_df.head()


print("Train Dataset Info:")
train_df.info()

print("\nTest Dataset Info:")
test_df.info()


print("Missing values in Train Data:\n", train_df.isnull().sum())
print("\nMissing values in Test Data:\n", test_df.isnull().sum())


print("Duplicate rows in Train:", train_df.duplicated().sum())
print("Duplicate rows in Test:", test_df.duplicated().sum())


print("Train Dataset Columns:\n", train_df.columns.tolist())


cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index, palette="Set2")
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()


num_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=train_df, x=col, y='Age', hue='satisfaction')
    plt.title(f'{col} vs Age by Satisfaction')
    plt.xticks(rotation=45)
    plt.show()


# Average age of satisfied vs neutral/dissatisfied customers
train_df.groupby('satisfaction')['Age'].mean()

# Which class has the highest average flight distance
train_df.groupby('Class')['Flight Distance'].mean()

# Average delay by travel type
train_df.groupby('Type of Travel')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean()


# Average flight distance by customer type
train_df.groupby('Customer Type')['Flight Distance'].mean()

# Average inflight service ratings by satisfaction
train_df.groupby('satisfaction')[['Inflight service', 'On-board service', 'Cleanliness']].mean()

# Satisfaction count by gender
train_df.groupby(['Gender', 'satisfaction']).size().unstack()

# Median age by class and travel type
train_df.groupby(['Class', 'Type of Travel'])['Age'].median()


# Top 10 passengers with highest flight distance
train_df.sort_values(by='Flight Distance', ascending=False).head(10)

# Passengers with longest departure delay
train_df.sort_values(by='Departure Delay in Minutes', ascending=False).head(10)


# Passengers who are satisfied and traveled in Business class
train_df[(train_df['satisfaction'] == 'satisfied') & (train_df['Class'] == 'Business')]

# Female passengers under 25 who were neutral or dissatisfied
train_df[(train_df['Gender'] == 'Female') & (train_df['Age'] < 25) & (train_df['satisfaction'] == 'neutral or dissatisfied')]


# Satisfaction across travel type and class
pd.crosstab(train_df['Type of Travel'], train_df['Class'], margins=True)

# Heatmap of satisfaction rate by gender and class
ct = pd.crosstab(train_df['Gender'], train_df['Class'], values=(train_df['satisfaction'] == 'satisfied').astype(int), aggfunc='mean')
sns.heatmap(ct, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Satisfaction Rate by Gender and Class")
plt.show()

