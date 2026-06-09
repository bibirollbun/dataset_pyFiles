import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = "/kaggle/input/playground-series-s5e6"
test = pd.read_csv(path + "/test.csv")
train = pd.read_csv(path + "/train.csv")


train.head()


train["Fertilizer Name"].unique()


#TemparatureとFertilizer Nameの関係
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fertilizer Name', y='Temparature', data=train, palette='viridis')
plt.xlabel('Fertilizer Name')
plt.ylabel('Temparature')
plt.xticks(rotation=45, ha='right') # ラベルが重なるのを防ぐ
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#HumidityとFertilizer Nameの関係
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fertilizer Name', y='Humidity', data=train, palette='viridis')
plt.xlabel('Fertilizer Name')
plt.ylabel('Humidity')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#MoistureとFertilizer Nameの関係
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fertilizer Name', y='Moisture', data=train, palette='viridis')
plt.xlabel('Fertilizer Name')
plt.ylabel('Moisture')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Soil TypeとFertilizer Nameの関係(データ数)
cross_soil_fertilizer = pd.crosstab(train["Soil Type"], train['Fertilizer Name'])
plt.figure(figsize=(8, 6))
sns.heatmap(cross_soil_fertilizer, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5, cbar_kws={'label': 'Count'})
plt.xlabel('Fertilizer Name')
plt.ylabel('Soil Type')
plt.tight_layout()
plt.show()


#Soil TypeとFertilizer Nameの関係(各Soil Typeに対する割合)
cross_soil_fertilizer = pd.crosstab(train["Soil Type"], train['Fertilizer Name'], normalize="index")
plt.figure(figsize=(8, 6)) 
sns.heatmap(cross_soil_fertilizer, annot=True, cmap='YlGnBu', linewidths=.5, cbar_kws={'label': 'Count'})
plt.xlabel('Fertilizer Name')
plt.ylabel('Soil Type')
plt.tight_layout()
plt.show()

