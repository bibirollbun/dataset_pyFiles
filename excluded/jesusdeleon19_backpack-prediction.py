import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
df.head()


df.isna().sum()


df_test.isna().sum()


fig,axs = plt.subplots(2,3,figsize=(12,9))

Size = df['Size'].value_counts()
axs[0,0].pie(Size,labels=Size.index,autopct='%1.1f%%')
axs[0,0].set_title('Size Distribution')

Color = df['Color'].value_counts()
axs[0,1].pie(Color,labels=Color.index,autopct='%1.1f%%')
axs[0,1].set_title('Color Distribution')

Material = df['Material'].value_counts()
axs[1,0].pie(Material,labels=Material.index,autopct='%1.1f%%')
axs[1,0].set_title('Material Distribution')

Brand = df['Brand'].value_counts()
axs[1,1].pie(Brand,labels=Brand.index,autopct='%1.1f%%')
axs[1,1].set_title('Brand Distribution')

lc = df['Laptop Compartment'].value_counts()
axs[0,2].pie(lc,labels=lc.index,autopct='%1.1f%%')
axs[0,2].set_title('Laptop Compartment Distribution')

Waterproof = df['Waterproof'].value_counts()
axs[1,2].pie(Waterproof,labels=Waterproof.index,autopct='%1.1f%%')
axs[1,2].set_title('Waterproof Distribution')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2,2,figsize=(14,9))

sns.histplot(df['Compartments'], ax=axs[0,0], kde=True)
axs[0,0].set_title('Compartments Distribution')

sns.histplot(df['Weight Capacity (kg)'], ax=axs[0,1], kde=True)
axs[0,1].set_title('Weight Capacity Distribution')

sns.histplot(df['Style'], ax=axs[1,0], kde=True)
axs[1,0].set_title('Style Distribution')

sns.histplot(df['Price'], ax=axs[1,1], kde=True)
axs[1,1].set_title('Price Distribution')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2,2,figsize=(14,9))

sns.boxplot(df['Price'], ax=axs[0,0], color='red')
axs[0,0].set_title('Price Boxplot');

sns.boxplot(df['Weight Capacity (kg)'], ax=axs[0,1])
axs[0,1].set_title('Weight Capacity Boxplot')

sns.boxplot(df['Color'].value_counts(), ax=axs[1,0])
axs[1,0].set_title('Color Boxplot')

sns.boxplot(df['Material'].value_counts(), ax=axs[1,1])
axs[1,1].set_title('Material Boxplot')

plt.tight_layout()
plt.show()


oe = OrdinalEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = oe.fit_transform(df[[col]])

df.info()


# Use MICE to Impute Missing Data

mice_imputer = IterativeImputer(random_state=42)
df = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
df.isna().sum()


# Test Data

oe = OrdinalEncoder()
for col in df_test.select_dtypes(include='object').columns:
    df_test[col] = oe.fit_transform(df_test[[col]])

df_test.info()


# Use MICE on Test Data
mice_imputer = IterativeImputer(random_state=42)
df_test = pd.DataFrame(mice_imputer.fit_transform(df_test), columns=df_test.columns)
df_test.isna().sum()


# Heatmap of Data Relationship
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True, fmt='.1%', cmap='coolwarm')
plt.show()

