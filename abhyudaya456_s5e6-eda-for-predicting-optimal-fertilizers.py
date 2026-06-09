import pandas as pd
import warnings
warnings.filterwarnings('ignore')



df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()



df.shape


df.info()



df.describe()


import seaborn as sns
import matplotlib.pyplot as plt



plt.figure(figsize=(12,6))
sns.countplot(y='Fertilizer Name', data=df, order=df['Fertilizer Name'].value_counts().index)
plt.title("Distribution of Fertilizer Labels")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.show()


plt.figure(figsize=(10,4))
sns.countplot(data=df, x='Soil Type', order=df['Soil Type'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Soil Type Distribution")
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(data=df, x='Crop Type', order=df['Crop Type'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Crop Type Distribution")
plt.show()



numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
df[numerical_cols].describe()



df[numerical_cols].hist(figsize=(15,10), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()



plt.figure(figsize=(10,6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numerical Features")
plt.show()



plt.figure(figsize=(12, 6))
sns.boxplot(x='Fertilizer Name', y='Nitrogen', data=df)
plt.xticks(rotation=45)
plt.title("Nitrogen Level vs Fertilizer")
plt.show()



df.groupby('Fertilizer Name')[numerical_cols].mean().style.background_gradient(cmap='YlGnBu')



sns.countplot(data=df, x='Soil Type', hue='Fertilizer Name')



sns.countplot(data=df, y='Crop Type', hue='Fertilizer Name')



sns.pairplot(df.sample(5000), hue='Fertilizer Name', vars=['Nitrogen', 'Phosphorous', 'Potassium'])



sns.boxplot(data=df[['Nitrogen', 'Phosphorous', 'Potassium']])



from sklearn.preprocessing import LabelEncoder
df_encoded = df.copy()
df_encoded['Soil Type'] = LabelEncoder().fit_transform(df['Soil Type'])
df_encoded['Crop Type'] = LabelEncoder().fit_transform(df['Crop Type'])
df_encoded['Fertilizer Name'] = LabelEncoder().fit_transform(df['Fertilizer Name'])
df_encoded.head()





