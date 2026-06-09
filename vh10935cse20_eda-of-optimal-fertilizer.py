import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


df_train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


df_train.head(3)


df_test.head(3)


print("Size of the Training Dataset :",df_train.shape)
print("Size of the Testing Dataset :",df_test.shape)


df_train.info() #info of the training dataset


df_test.info()#info of the testing dataset


df_train.columns #printing the columns


df_test.columns #printing the columns


#Checking for null values on training dataset
df_train.isna().sum()


#Checking for null values on testing dataset
df_train.isna().sum()


Fertilizer_count=df_train['Fertilizer Name'].value_counts().reset_index()
Fertilizer_count.columns = ['Fertilizer Name', 'Count']
Fertilizer_count


Soil_count=df_train['Soil Type'].value_counts().reset_index()
Soil_count.columns = ['Soil Type', 'Count']
Soil_count


plt.figure(figsize=(10, 5))
sns.countplot(data=df_train,x="Fertilizer Name" ,
              order=df_train["Fertilizer Name"].value_counts().index,palette="Set1")
plt.title("Fertilizer Count")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.show()


plt.figure(figsize=(12, 7))
sns.countplot(data=df_train, x='Soil Type', hue='Fertilizer Name', palette='tab10')
plt.title('Fertilizer Name Distribution by Soil Type')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



cat_cols=['Soil Type','Crop Type']
for i in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=df_train,x=i,order=df_train[i].value_counts().index)
    plt.title(f"{i} distribution")  
    plt.show()


num_cols=df_train.select_dtypes(include="number").columns.drop("id")
ncols = 3
nrows = int(np.ceil(len(num_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    sns.histplot(df_train[col], kde=True, ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()
    


#num_cols=df_train.drop(list(cat_cols)+['id'],axis=1)
vc = df_train['Fertilizer Name'].value_counts().sort_values(ascending=False)
for i in num_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df_train,x="Fertilizer Name",y=i,
               order=vc.index)
    plt.xticks(rotation=45)
    plt.show()


corr=df_train[num_cols].corr(method="spearman")
sns.heatmap(corr,cmap="coolwarm",annot=True,square=True)
plt.title("Spearman Correlation")
plt.show


sns.pairplot(
    df_train[num_cols.union(['Fertilizer Name'])],
     hue="Fertilizer Name", corner=True, diag_kind="kde",
    height=1.5, plot_kws=dict(alpha=.3, linewidth=0)
)
plt.show()

