import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')


df.head()


df.shape


df['PCOS'].unique()
df['Exercise_Type'].unique()


df_copy = df.copy()


df_copy['PCOS'] = df_copy['PCOS'].replace({'No':0, 'Yes':1})
df_copy['Hormonal_Imbalance'] = df_copy['Hormonal_Imbalance'].replace({'No':0, 'Yes':1})
df_copy['Hyperandrogenism'] = df_copy['Hyperandrogenism'].replace({'No':0, 'Yes':1})
df_copy['Hirsutism'] = df_copy['Hirsutism'].replace({'No':0, 'Yes':1})
df_copy['Conception_Difficulty'] = df_copy['Conception_Difficulty'].replace({'No':0, 'Yes':1})
df_copy['Insulin_Resistance'] = df_copy['Insulin_Resistance'].replace({'No':0, 'Yes':1})


df_copy.describe()


print(df_copy.isnull().sum())


plt.hist(df['Weight_kg'])


sns.heatmap(df_copy.isnull(), cmap='viridis')


df_copy_1 = df_copy.drop(['Exercise_Type'], axis=1)



cat_cols = df_copy_1.select_dtypes(include=['object']).columns 
encoders = {} 

for col in cat_cols:
    le = LabelEncoder()
    df_copy_1[col] = le.fit_transform(df_copy_1[col].astype(str)) 
    encoders[col] = le  


imputer = KNNImputer(n_neighbors=3) 
df_imputed = pd.DataFrame(imputer.fit_transform(df_copy_1), columns=df_copy_1.columns)


for col in cat_cols:
    df_imputed[col] = encoders[col].inverse_transform(df_imputed[col].astype(int))  


print(df_imputed.isnull().sum())


plt.figure(figsize=(10,4))
sns.heatmap(df_imputed.isnull(), cmap='viridis')


df_copy = df_imputed.drop(['ID'], axis=1)


plt.figure(figsize=(4,4))
sns.boxplot(df_copy['Weight_kg'], color="lightblue", flierprops={'markerfacecolor': 'r', 'marker': 'o'})




