import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


Train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
Test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
Original = pd.read_csv('/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv')


print('Shape of Train data is: ', Train.shape)
print('Shape of Test data is: ', Test.shape)
print('Shape of Original data is:', Original.shape)


Train.head()


Test.head()


Train = Train.drop(['id'] , axis=1)
print('Shape of Train data is: ', Train.shape)

Test = Test.drop(['id'] , axis=1)
print('Shape of Test data is: ', Test.shape)


Train.info()


Original.head()


Train_Original = pd.concat([Train, Original], ignore_index=True)
print('Shape of Train data is : ' , Train.shape)


#Identify columns with numerical and non-numerical data
Cat_Features = Train_Original.select_dtypes(include=['object']).columns
Numeric_Feature = Train_Original.select_dtypes(exclude=['object']).columns


Train_Original[Numeric_Feature].corr()


sns.heatmap(Train_Original[Numeric_Feature].corr(),annot= True)
plt.rcParams['figure.figsize'] = (20,7)
plt.show()


Train_Original.select_dtypes(include=[np.number]).describe().T


Train_Original.describe(include=object)


Train_Original = Train_Original.drop_duplicates().reset_index(drop=True)
Train_Original.duplicated().sum()


Train_Original.duplicated().sum()


Test.duplicated().sum()


Train_Original.isnull().sum()


Train_Original = Train_Original.dropna()
Train_Original.isna().sum()


Test = Test.dropna()
Test.isna().sum()


def Plot_Boxplot(df, columns):
    fig, axes = plt.subplots(len(columns), 1, figsize=(5, 3 * len(columns)))

    if len(columns) == 1:
        axes = [axes]  

    for ax, col in zip(axes, columns):
        sns.boxplot(x=df[col], ax=ax, color='lightblue')  
        ax.set_title(f'Boxplot of {col}', fontsize=12)
        ax.set_xlabel(col, fontsize=10)

    plt.tight_layout()
    plt.show()

# اجرای تابع فقط برای "Price" و "Weight Capacity (kg)"
Plot_Boxplot(Train_Original, ['Price', 'Weight Capacity (kg)','Compartments'])




def Plot_Specific_Numerical_Data(df, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 3))

    if len(columns) == 1:
        axes = [axes]  

    for ax, col in zip(axes, columns):
        sns.histplot(df[col], kde=True, bins=30, color='blue', ax=ax)
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


Plot_Specific_Numerical_Data(Train_Original, ['Price', 'Weight Capacity (kg)'])



plt.figure(figsize=(5,3))
sns.countplot(x=Train_Original['Compartments'], palette='coolwarm')
plt.title('Count of Compartments')
plt.xlabel('Number of Compartments')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


#Plotting Histograms for Categorical Columns

def Plot_Categorical_Distributions(data, n_cols=3):
    Categorical_Col = data.select_dtypes(include=['object']).columns.tolist()

    n_rows = int(np.ceil(len(Categorical_Col) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()
    
    for idx, col in enumerate(Categorical_Col):
        value_counts = data[col].value_counts()

        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Count")

        if data[col].nunique() > 10:
            axes[idx].set_xticks([])
            axes[idx].set_xlabel(f"{col} (>10 unique values)")
        else:
            axes[idx].tick_params(axis='x', rotation=45)

    for i in range(len(Categorical_Col), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

Plot_Categorical_Distributions(Train_Original)


sns.set(style="whitegrid")

def Fix_Skewness(df, threshold=0.5):
    numerical_cols = df.select_dtypes(include=['float64']).columns
    
    Skewness_Results = {}

    for col in numerical_cols:
        Skewness = skew(df[col])
        Skewness_Results[col] = Skewness
        
        if abs(Skewness) > threshold:
            if Skewness > 0:  
                df[col] = np.log1p(df[col])   
                Skewness_Results[col] = "Log Transformed"
            else:  
                df[col] = np.sqrt(df[col])  
                Skewness_Results[col] = "Square Root Transformed"

    return df, Skewness_Results

train_original_fixed, Skewness_info = Fix_Skewness(Train_Original)

print("Skewness Results:")
for column, result in Skewness_info.items():
    print(f"{column}: {result}")

def Plot_Numerical_Data_Fixed(df):
    numerical_cols = df.select_dtypes(include=['float64']).columns
    num_columns = len(numerical_cols)
    
    fig, axes = plt.subplots(1, num_columns, figsize=(5*num_columns, 4))   
    if num_columns == 1:
        axes = [axes]
    
    for ax, col in zip(axes, numerical_cols):
        sns.histplot(df[col], kde=True, bins=30, color='blue', ax=ax)
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    plt.tight_layout()   
    plt.show()

Plot_Numerical_Data_Fixed(train_original_fixed)

