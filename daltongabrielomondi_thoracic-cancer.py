#importing relevant libraries for analysis and visualizaiton
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import os

#Ensure plots display in Colab
%matplotlib inline

#set seaborn style for clean visualizations
sns.set(style='whitegrid')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# define path to training data
data_path = '/kaggle/input/grand-xray-slam-division-b/train2.csv'

#load training dataset with error handling
try:
    train_df = pd.read_csv(data_path)
    print(f"Successfully loaded train.csv with shape: {train_df.shape}")

except FileNotFoundError:
    print(f"Error: {data_path} not found. Please check the file path")
    raise


train_df.head()


print("Dataset Info: ")
print(train_df.info())


#summary key metrics
total_images = len(train_df)
total_patients = train_df['Patient_ID'].nunique()
total_studies= train_df['Study'].nunique()
print(f"Total Images: {total_images}")
print(f"Total Patients: {total_patients}")
print(f"Total Studies: {total_studies}")


#Check for missing values
print("\nMissing Values: ")
print(train_df.isnull().sum())


#Define the 14 condition columns
label_columns = ['No Finding', 'Lung Opacity', 'Support Devices', 'Atelectasis', 
                'Cardiomegaly', 'Pleural Effusion', 'Enlarged Cardiomediastinum',
                'Edema', 'Consolidation', 'Pneumonia', 'Fracture', 'Lung Lesion',
                'Pneumothorax', 'Pleural Other']

#calculate counts and percentages for each condition
label_counts = train_df[label_columns].sum()
label_percentages = (label_counts / total_images * 100).round(2)
prevalence_df = pd.DataFrame({
    'Condition': label_counts.index,
    'Count': label_counts.values,
    'Percent (%)': label_percentages.values
})

#display prevalence table
print("label prevalence")
print(prevalence_df)


plt.figure(figsize=(12, 6))
sns.barplot(x='Count', y='Condition', data=prevalence_df, palette='viridis', hue=None )
plt.title('Label counts (Number of Positive Cases)')
plt.xlabel('Count')
plt.ylabel('Condition')
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.savefig('/content/label_counts_barplot.jpg')
plt.show()


# donut chart for label percentages
plt.figure(figsize=(8,8))
colors = sns.color_palette('viridis', len(prevalence_df))
plt.pie(prevalence_df['Count'], labels=prevalence_df['Condition'],
       autopct=lambda pct: f'{pct:.1f}%', )

