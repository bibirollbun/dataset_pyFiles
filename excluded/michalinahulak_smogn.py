# pip install smogn
# import smogn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


# df_smogn = smogn.smoter(
#     data=train,
#     y='Calories', 
#     k=5,           
#     samp_method='extreme',  
#     rel_thres=0.8,          
#     rel_method='auto',     
#     under_samp=0.5          
# )


# print("Oryginalny rozmiar:", train.shape)
# print("Po SMOGN:", df_smogn.shape)


# df_smogn.to_csv("train_smogn.csv", index=False)


df_smogn = pd.read_csv('/kaggle/input/smogn-predict-calorie-expenditure-playground-s5e5/train_smogn.csv')
df_smogn = df_smogn.round(1)

df_smogn_male = df_smogn[df_smogn['Sex'] == 'male']
df_smogn_female = df_smogn[df_smogn['Sex'] == 'female']


def plot_distributions_by_sex_train_test(train, test, column):
    plt.figure(figsize=(20, 5))

    # Train: Histogram + KDE
    plt.subplot(1, 4, 1)
    sns.histplot(data=train, x=column, hue='Sex', kde=True, bins=30, palette='Set2', element='step')
    plt.title(f'{column} Distribution (Train)')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True)

    # Train: Boxplot
    plt.subplot(1, 4, 2)
    sns.boxplot(data=train, x='Sex', y=column, palette='Set2')
    plt.title(f'{column} Boxplot (Train)')
    plt.xlabel('Sex')
    plt.ylabel(column)
    plt.grid(True)

    # df_smogn: Histogram + KDE
    plt.subplot(1, 4, 3)
    sns.histplot(data=test, x=column, hue='Sex', kde=True, bins=30, palette='Set2', element='step')
    plt.title(f'{column} Distribution (df_smogn)')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.grid(True)

    # df_smogn: Boxplot
    plt.subplot(1, 4, 4)
    sns.boxplot(data=test, x='Sex', y=column, palette='Set2')
    plt.title(f'{column} Boxplot (df_smogn)')
    plt.xlabel('Sex')
    plt.ylabel(column)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


columns_to_plot = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

for col in columns_to_plot:
    plot_distributions_by_sex_train_test(train, df_smogn, col)

