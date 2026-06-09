import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train.head()


train.isnull().sum()


train.info()


train['Sex'].unique()


train['sex_numeric'] = train['Sex'].map({'male': 1, 'female': 0})


train.info()


train.describe()


train.head()


train_num = train.select_dtypes(exclude = [object])


plt.figure(figsize = (12,8))
df_corr = train_num.corr()
sns.heatmap(df_corr, annot = True, cmap = 'coolwarm')


for col in train.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


train_no_outliers = remove_outliers_iqr(train, train_num)

print("Original shape:", train.shape)
print("Shape after outlier removal:", train_no_outliers.shape)


for col in train_no_outliers.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train_no_outliers[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


train_no_outliers.describe()


train_no_outliers.head()


train_num_outlier = train_no_outliers.select_dtypes(exclude = [object])


sex_counts = train["Sex"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Sex")
plt.show()


plt.figure(figsize=(18, 16))
for i, col in enumerate(train_num_outlier, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train_num_outlier[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()



train_no_outliers['BMI'] = train_no_outliers['Weight'] / ((train_no_outliers['Height'] / 100) ** 2)

bins = [19, 29, 39, 49, 59, 69, 79]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
train_no_outliers['Age_Group'] = pd.cut(train_no_outliers['Age'], bins=bins, labels=labels)

train_no_outliers['Intensity'] = pd.cut(train_no_outliers['Heart_Rate'], bins=[0, 90, 110, 200], labels=['Low', 'Moderate', 'High'])


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_no_outliers, x='Sex', y='Calories')
plt.title('Calories Burned by Gender')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_no_outliers, x='Age_Group', y='Calories')
plt.title('Calories Burned by Age Group')
plt.show()


plt.figure(figsize=(12, 6))
sns.scatterplot(data=train_no_outliers, x='Duration', y='Calories', hue='Sex', alpha=0.5)
plt.title('Duration vs Calories Burned by Gender')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=train_no_outliers, x='Intensity', y='Calories')
plt.title('Calories Burned by Workout Intensity')
plt.show()




