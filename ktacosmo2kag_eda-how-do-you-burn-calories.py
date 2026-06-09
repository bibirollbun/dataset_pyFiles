# More discussion is here: https://www.kaggle.com/competitions/playground-series-s5e5/discussion/575986


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv', index_col='id')

colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#FFB366', '#FF99FF']
plt.style.use('seaborn-v0_8')


print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nTrain Info:")
train.info()
print("\nTest Info:")
test.info()


#Add feature
train['BMI'] = train['Weight'] / (train['Height'] / 100) ** 2
test['BMI'] = test['Weight'] / (test['Height'] / 100) ** 2

# Sex mapping
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})


print("\nTrain Describe:")
train.describe()


plt.figure(figsize=(10, 6))
sns.histplot(train['Calories'], bins=50, kde=True)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train['Calories']), bins=50, kde=True)
plt.title('Distribution of Log(Calories + 1)')
plt.xlabel('Log(Calories + 1)')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix', fontsize=15, pad=20)
plt.show()


num_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
plt.figure(figsize=(20, 15))
for i, feature in enumerate(num_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=train, x=feature, kde=True, color=colors[i-1])
    plt.title(f'Distribution of {feature}', fontsize=12)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Feature distribution by gender
plt.figure(figsize=(20, 15))
for i, feature in enumerate(num_features[:-1], 1):  # Excluding Calories
    plt.subplot(3, 3, i)
    sns.boxplot(data=train, x='Sex', y=feature, palette=['#FF9999', '#66B2FF'])
    plt.title(f'{feature} Distribution by Gender', fontsize=12)
    plt.xticks([0, 1], ['Female', 'Male'])
plt.tight_layout()
plt.show()


# Scatter plots of features vs Calories
plt.figure(figsize=(20, 15))
for i, feature in enumerate(num_features[:-1], 1):  # Excluding Calories
    plt.subplot(3, 3, i)
    sns.scatterplot(data=train, x=feature, y='Calories', hue='Sex', palette=['#FF9999', '#66B2FF'])
    plt.title(f'Calories vs {feature}', fontsize=12)
plt.tight_layout()
plt.show()


# Analysis by age groups
train['Age_Group'] = pd.cut(train['Age'], 
                           bins=[0, 20, 30, 40, 50, 60, 100], 
                           labels=['Under 20', '20-30', '30-40', '40-50', '50-60', 'Over 60'])
plt.figure(figsize=(15, 6))
sns.boxplot(data=train, x='Age_Group', y='Calories', palette=colors)
plt.title('Calorie Expenditure by Age Group', fontsize=15)
plt.xticks(rotation=45)
plt.show()


# Analysis by BMI categories
train['BMI_Category'] = pd.cut(train['BMI'], 
                              bins=[0, 18.5, 25, 30, 100],
                              labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='BMI_Category', y='Calories', palette=colors[:4])
plt.title('Calorie Expenditure by BMI Category', fontsize=15)
plt.show()


# Relationship between Duration and Calories (colored by Heart Rate)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(train['Duration'], train['Calories'], 
                     c=train['Heart_Rate'], cmap='viridis', 
                     alpha=0.6)
plt.colorbar(scatter, label='Heart Rate')

# 線形回帰のフィット（切片なし）
slope = np.sum(train['Duration'] * train['Calories']) / np.sum(train['Duration'] ** 2)
x_line = np.array([train['Duration'].min(), train['Duration'].max()])
y_line = slope * x_line
plt.plot(x_line, y_line, 'r--', label=f'Linear Fit (y = {slope:.2f}x)')

plt.title('Duration vs Calories (Colored by Heart Rate)', fontsize=15)
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.legend()
# plt.savefig('duration_vs_calories.pdf', bbox_inches='tight')
plt.show()

