# Import Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# remove future warnings

import warnings

# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)




train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


data_card <- tribble(
    
)


# quick check

train.head()


train.info()


train.describe().round()


# Shape of train dataset
train.shape


# let's check for null values
train.isnull().sum()


# identify numeric columns

features_numerical = train.select_dtypes(include = 'number').columns
features_numerical


# identify categorical columns

features_categorical = train.select_dtypes(include = ['object', 'category']).columns
features_categorical



gender_counts = train['gender'].value_counts()

gender_counts


# Plot with seaborn

sns.barplot(
    x=gender_counts.index, 
    y=gender_counts.values,
    palette='magma'
)


plt.title('Gender Distribution')

plt.show()





ethnicity_counts = train['ethnicity'].value_counts()

ethnicity_counts


# Plot with seaborn

sns.barplot(
    x=ethnicity_counts.index, 
    y=ethnicity_counts.values,
    palette='magma'
)


plt.title('Ethnicity Distribution')

plt.show()


counts = train['age'].value_counts()

counts


sns.barplot(
    x=counts.index, 
    y=counts.values,
    palette='magma'
)


plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()


sns.histplot(
    data=train, 
    x='age', 
    kde=True, 
    bins=100, 
    color='purple'
)


sns.violinplot(x=train['age'], color='purple')

plt.title('Age Density')
plt.xlabel('Age')
plt.show()


counts_alcohol = train['alcohol_consumption_per_week'].value_counts()

counts_alcohol


sns.barplot(
    x=counts_alcohol.index, 
    y=counts_alcohol.values,
    palette='magma'
)


plt.title('Alcohol Consumption Per Week')
plt.xlabel('Units')
plt.show()


for col in features_categorical:
    sns.countplot(
        train,
        x=col,
        palette='magma',
        hue='diagnosed_diabetes'
    )
    plt.title(col)
    plt.show()


# Feature vs Target Analysis

for col in features_numerical[1:]:
    plt.figure(figsize=(8,5))
    sns.kdeplot(
        data=train, 
        x=col, 
        hue="diagnosed_diabetes", 
        fill=True, 
        palette="magma",       
        alpha=0.15            
    )
    plt.title(f"KDE Plot of {col} by diagnosed_diabetes")
    plt.xlabel(col)
    plt.ylabel("Density")

    # Manually update the legend, to be a bit nicer
    plt.legend(title="Diagnosis", labels=["No Diabetes", "Diabetes"])
    
    plt.show()



