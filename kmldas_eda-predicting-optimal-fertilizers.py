import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


import warnings
# Suppress only the specific warning from seaborn/pandas
warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")

# Set up for better plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df




def analyze_numeric_features(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'id']  # Skip ID column

    for col in numeric_cols:
        print(f"\nðŸ“Š Feature: {col}")
        print(df[col].describe())
        
        # Histogram + KDE
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
        
        # Boxplot
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f'Boxplot of {col}')
        plt.show()
analyze_numeric_features(df)




def analyze_categorical_features(df):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        print(f"\nðŸ”¤ Feature: {col}")
        print(df[col].value_counts())

        plt.figure(figsize=(12, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette="Set2")
        plt.title(f'Frequency Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()


analyze_categorical_features(df)



def bivariate_analysis(df, target_col='Fertilizer Name'):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id']]  # Exclude ID
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_col]  # Exclude target itself

    # --- Numeric Features vs Target ---
    for col in numeric_cols:
        plt.figure(figsize=(12, 5))
        sns.boxplot(x=target_col, y=col, data=df, palette="Set3")
        plt.title(f'{col} vs {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # --- Categorical Features vs Target ---
    for col in categorical_cols:
        plt.figure(figsize=(12, 5))
        sns.countplot(data=df, x=col, hue=target_col, palette="Set1")
        plt.title(f'{col} by {target_col}')
        plt.xticks(rotation=45)
        plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


bivariate_analysis(df)




def multivariate_analysis(df):
    numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

    # --- Correlation Heatmap ---
    plt.figure(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.show()



multivariate_analysis(df)


