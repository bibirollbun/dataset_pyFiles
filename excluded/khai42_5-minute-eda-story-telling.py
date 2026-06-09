import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import scipy.stats as stats


train_df=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


print('Train Data\n')
train_df.head(3)


train_df.drop(columns=['id']).groupby('Sex').mean()


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
sns.histplot(train_df['Duration'], bins=30, kde=False)
plt.title('Histogram of Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df['Duration'])
plt.title('Boxplot of Duration')
plt.xlabel('Duration (minutes)')
plt.tight_layout()
plt.show()


print('Mean of Duration: ',train_df['Duration'].mean())
print('Median of Duration: ',train_df['Duration'].median())
print('First quartile of Duration: ',train_df['Duration'].quantile(0.25))
print('Third quartile of Duration: ',train_df['Duration'].quantile(0.75))
print('Skewness of Duration :',train_df['Duration'].skew())


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
sns.histplot(train_df['Heart_Rate'], bins=30, kde=False)
plt.title('Histogram of Heart_Rate')
plt.xlabel('Heart_Rate')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df['Heart_Rate'])
plt.title('Boxplot of Heart_Rate')
plt.xlabel('Heart_Rate')
plt.tight_layout()
plt.show()


Q1 = train_df['Heart_Rate'].quantile(0.25)
Q3 = train_df['Heart_Rate'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = train_df[(train_df['Heart_Rate'] < lower_bound) | (train_df['Heart_Rate'] > upper_bound)]
print(len(outliers))
print(outliers['Sex'].value_counts())
print(train_df['Heart_Rate'].skew())


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
sns.histplot(train_df['Body_Temp'], bins=30, kde=False)
plt.title('Histogram of Body_Temp')
plt.xlabel('Body_Temp')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df['Body_Temp'])
plt.title('Boxplot of Body_Temp')
plt.xlabel('Body_Temp')
plt.tight_layout()
plt.show()


Q1 = train_df['Body_Temp'].quantile(0.25)
Q3 = train_df['Body_Temp'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = train_df[(train_df['Body_Temp'] < lower_bound) | (train_df['Body_Temp'] > upper_bound)]
print(len(outliers))
print(outliers['Sex'].value_counts())
print(train_df['Body_Temp'].skew())


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
sns.histplot(train_df['Calories'], bins=30, kde=False)
plt.title('Histogram of Calories')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
sns.boxplot(x=train_df['Calories'])
plt.title('Boxplot of Calories')
plt.xlabel('Calories')
plt.tight_layout()
plt.show()


Q1 = train_df['Calories'].quantile(0.25)
Q3 = train_df['Calories'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = train_df[(train_df['Calories'] < lower_bound) | (train_df['Calories'] > upper_bound)]
print(len(outliers))
print(outliers['Sex'].value_counts())
print(train_df['Calories'].skew())


plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Height', y='Weight', hue='Sex', alpha=0.6,)
plt.title('Height vs Weight by Sex')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Duration', y='Calories')
plt.title('Duration & Calories')
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.tight_layout()
plt.show()


corr = train_df['Duration'].corr(train_df['Calories'])
corr


plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Heart_Rate', y='Calories')
plt.title('Heart_Rate & Calories')
plt.xlabel('Heart_Rate')
plt.ylabel('Calories')
plt.tight_layout()
plt.show()


corr = train_df['Heart_Rate'].corr(train_df['Calories'])
corr


hr_bins = [66, 77, 87, 97, 108, 118, 128]
hr_labels = ['67-77', '78-87', '88-97', '98-108', '109-118', '119-128']
train_df['HR_Bin'] = pd.cut(train_df['Heart_Rate'], bins=hr_bins, labels=hr_labels, include_lowest=True)
bin_label = train_df.groupby('HR_Bin')['Calories'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=bin_label, x='HR_Bin', y='Calories', marker='o')
plt.title('Average Calories by Heart Rate Bin')
plt.xlabel('Heart Rate Bin')
plt.ylabel('Average Calories')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='Body_Temp', y='Calories')
plt.title('Body_Temp & Calories')
plt.xlabel('Body_Temp')
plt.ylabel('Calories')
plt.tight_layout()
plt.show()


corr = train_df['Body_Temp'].corr(train_df['Calories'])
corr


body_temp_bins = [37, 37.8, 38.6, 39.3, 40.0, 40.8, 41.5]
body_temp_labels = ['37-37.8', '37.8-38.6', '38.6-39.3', '39.3-40.0', '40.0-40.8', '40.8-41.5']
train_df['Body_Temp_Bin'] = pd.cut(train_df['Body_Temp'], bins=body_temp_bins, labels=body_temp_labels, include_lowest=True)
temp_bin = train_df.groupby('Body_Temp_Bin')['Calories'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=temp_bin, x='Body_Temp_Bin', y='Calories')
plt.title('Average Calories Burned by Body Temperature Bin')
plt.xlabel('Body Temperature Bin')
plt.ylabel('Average Calories')
plt.tight_layout()
plt.show()


train_df['Calories_Bin'] = pd.qcut(train_df['Calories'], 5)
bin_edges = train_df['Calories_Bin'].cat.categories
bin_midpoints = bin_edges.map(lambda x: (x.left + x.right) / 2)
bin_midpoint_map = dict(zip(bin_edges, bin_midpoints))
train_df['Calories_Bin_Mid'] = train_df['Calories_Bin'].map(bin_midpoint_map)
avg_duration_by_sex_calories = train_df.groupby(['Calories_Bin_Mid', 'Sex'])['Duration'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_duration_by_sex_calories, x='Calories_Bin_Mid', y='Duration', hue='Sex')
plt.title('Average Duration by Calories Bins and Sex')
plt.xlabel('Calories bin')
plt.ylabel('Average Duration')
plt.tight_layout()
plt.show()


if 'Calories' in train_df.columns:
    age_counts = train_df.groupby(['Calories', 'Sex']).size().reset_index(name='Count')
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")
    custom_palette = {'female': 'blue', 'male': 'darkgray'}
    sns.lineplot(data=age_counts, x='Calories', y='Count', hue='Sex', palette=custom_palette, linewidth=3)
    plt.title('Calories distribution by Sex')
    plt.xlabel('Calories')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
else:
    print("The 'Calories' column does not exist in the dataset.")


train_df['Duration_Group'] = pd.cut(train_df['Duration'], bins=[0, 20, train_df['Duration'].max()], labels=['<=20 min', '>20 min'], include_lowest=True)
avg_calories_duration = train_df.groupby(['Duration_Group', 'Sex'])['Calories'].mean().reset_index()
print(avg_calories_duration)

