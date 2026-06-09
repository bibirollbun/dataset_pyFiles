# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# 1. Data Loading: Reading the CSV file into a pandas DataFrame
file_path = '/kaggle/input/patient-data-csv/patient_data.csv'

# Reading the CSV file
df = pd.read_csv(file_path)

# Confirmation message after successful submission
print("Submission file run successfully.")





# Displaying the first few rows of the dataset to confirm loading
df.head()


# 3. Data Exploration: Descriptive Statistics
print("\nBasic Statistical Summary:")
print(df.describe(include='all'))



# 4. Count of Treatments Recommended (How many times each treatment is recommended)
treatment_counts = df['Treatment_Recommended'].value_counts()
print("\nCount of each Treatment Recommended:")
print(treatment_counts)



import matplotlib.pyplot as plt
import seaborn as sns
# 5. Visualization 1: Bar plot for Treatments Recommended
plt.figure(figsize=(10,6))
sns.barplot(x=treatment_counts.index, y=treatment_counts.values, palette='viridis')
plt.title("Distribution of Recommended Treatments for Patients", fontsize=16)
plt.xlabel("Treatment Type", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.xticks(rotation=45)
plt.show()


# 6. Visualization 2: Age Distribution by Condition
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Condition', y='Age', palette='coolwarm')
plt.title("Age Distribution Across Different Conditions", fontsize=16)
plt.xlabel("Medical Condition", fontsize=12)
plt.ylabel("Age of Patients", fontsize=12)
plt.show()



# 7. Visualization 3: Heatmap of Condition vs Comorbidities (Cross-tabulation)
condition_comorbidity = pd.crosstab(df['Condition'], df['Comorbidities'])
plt.figure(figsize=(10,6))
sns.heatmap(condition_comorbidity, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Condition vs Comorbidity Heatmap", fontsize=16)
plt.xlabel("Comorbidity", fontsize=12)
plt.ylabel("Primary Condition", fontsize=12)
plt.show()



# 8. Visualization 4: Age Distribution by Treatment Recommended
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Treatment_Recommended', y='Age', palette='Set2')
plt.title("Age Distribution for Each Treatment Recommended", fontsize=16)
plt.xlabel("Treatment Recommended", fontsize=12)
plt.ylabel("Age of Patients", fontsize=12)
plt.xticks(rotation=45)
plt.show()



# 9. Advanced Analysis: Treatment by Condition and Average Age
treatment_condition_age = df.groupby(['Condition', 'Treatment_Recommended']).agg({'Age': ['mean', 'std']}).reset_index()
print("\nTreatment by Condition and Average Age:")
print(treatment_condition_age)


# 10. Advanced Analysis: Correlation between Age and Treatment Recommendation
# Encoding Treatments numerically for correlation analysis
df['Treatment_Recommended_Code'] = df['Treatment_Recommended'].astype('category').cat.codes

# Correlation analysis
correlation_matrix = df[['Age', 'Treatment_Recommended_Code']].corr()
print("\nCorrelation matrix between Age and Treatment Recommended:")
print(correlation_matrix)

